import torch
import torch.multiprocessing as mp
import os
from datetime import datetime
import swanlab
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def _train_single_fold(rank: int, model, dataset, train_idx, val_idx, args: dict):
    """子进程：只训练 + 收集每轮指标，返回 best model + 完整日志历史"""
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=args["batch_size"], shuffle=False)

    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args["lr_decay_gamma"])

    best_val_loss = float("inf")
    best_model = None

    # 收集每轮的日志（主进程统一上传 swanlab）
    history = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}

    for epoch in range(1, args["epochs"] + 1):
        model.train()
        train_loss = 0.0
        grad_norms = []

        for date, stock_code, features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
            optimizer.step()
            train_loss += loss.item()
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0).item())

        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for date, stock_code, features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs.squeeze(), labels.squeeze()).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        args["logger"].info(
            f"Fold {rank} Epoch [{epoch}/{args["epochs"]}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Grad Norm: {avg_grad_norm:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

        # 记录到 history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["grad_norm"].append(avg_grad_norm)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        fold_model_dir = os.path.join(args["model_save_dir"], f"{args['project_name']}_{args['timestamp']}_period_{args['period_index']}_fold{rank}_model")
        os.makedirs(os.path.dirname(fold_model_dir), exist_ok=True)

        # 更新最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            fold_model_dir = os.path.join(args["model_save_dir"], f"{args['project_name']}_{args['timestamp']}_period_{args['period_index']}_fold{rank}_model")
            os.makedirs(fold_model_dir, exist_ok=True)
            best_model = model
            best_model_path = os.path.join(
                fold_model_dir, f"{args['project_name']}_{args['timestamp']}_period_{args['period_index']}_fold{rank}_epoch{epoch}.pt"
            )
        torch.save(best_model, best_model_path)

        # # 保存检查点
        # if args["save_model"] and epoch % args["model_save_frequency"] == 0:
        #     fold_model_dir = os.path.join(args["model_save_dir"], f"{args['project_name']}_{args['timestamp']}_period_{args['period_index']}_fold{rank}_model")
        #     os.makedirs(fold_model_dir, exist_ok=True)
        #     model_path = os.path.join(fold_model_dir, f"{args['project_name']}_{args['timestamp']}_period_{args['period_index']}_fold{rank}_epoch{epoch}.pt")
        #     torch.save(model, model_path)

    return best_model_path, best_val_loss, history


def _worker(rank: int, queue: mp.Queue, model, dataset, train_idx, val_idx, args: dict):
    best_model_path, best_val_loss, history = _train_single_fold(rank, model, dataset, train_idx, val_idx, args)
    queue.put((rank, best_model_path, best_val_loss, history))
    queue.close()


def train_neural_network_model_parallel(
    model,
    dataset,
    logger,
    epochs: int = 200,
    learning_rate: float = 1e-4,
    model_save_dir: str = "/home/user0/results/models/",
    save_model: bool = True,
    project_name: str = "StockPredictor",
    period_index: int = 0,
    model_save_frequency: int = 5,
    use_swanlab: bool = True,
    k_folds: int = 4,
    lr_decay_gamma: float = 0.99,
    batch_size: int = 32,
    timestamp: str = None,
):
    assert k_folds == 4
    assert torch.cuda.device_count() >= 4

    args = {
        "logger": logger,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "model_save_dir": model_save_dir,
        "save_model": save_model,
        "project_name": project_name,
        "timestamp": timestamp,
        "period_index": period_index,
        "model_save_frequency": model_save_frequency,
        "lr_decay_gamma": lr_decay_gamma,
        "batch_size": batch_size,
    }

    if use_swanlab:
        swanlab.init(project=project_name, config=args)
        logger.info("SwanLab initialized")

    logger.info("Starting 4-fold parallel training on 4 GPUs")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_splits = list(kf.split(range(len(dataset))))

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes = []

    for fold in range(k_folds):
        train_idx, val_idx = fold_splits[fold]
        p = mp.Process(target=_worker, args=(fold, queue, model, dataset, train_idx, val_idx, args))
        p.start()
        processes.append(p)

    # 等待所有进程结束并收集结果
    results = [None] * k_folds
    histories = [None] * k_folds

    for _ in range(k_folds):
        rank, best_model_path, best_val_loss, history = queue.get()
        results[rank] = (best_model_path, best_val_loss)
        histories[rank] = history
        logger.info(f"Fold {rank+1} finished, best val loss: {best_val_loss:.6f}")

    for p in processes:
        p.join()

    # 所有 fold 结束后统一上传 SwanLab
    if use_swanlab:
        for epoch in range(args["epochs"]):
            log_dict = {}
            for rank in range(k_folds):
                h = histories[rank]
                log_dict.update(
                    {
                        f"train_loss/fold_{rank+1}": h["train_loss"][epoch],
                        f"val_loss/fold_{rank+1}": h["val_loss"][epoch],
                        f"grad_norm/fold_{rank+1}": h["grad_norm"][epoch],
                        f"lr/fold_{rank+1}": h["lr"][epoch],
                    }
                )
            swanlab.log(log_dict, step=epoch + 1)

    best_model_path = [r[0] for r in results]

    best_models = []
    for rank in range(k_folds):
        best_model = torch.load(results[rank][0], weights_only=False)
        best_models.append(best_model)

    logger.info("All 4 folds completed & logged to SwanLab")

    return best_models
