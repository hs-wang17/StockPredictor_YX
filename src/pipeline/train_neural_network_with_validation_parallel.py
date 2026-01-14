import torch
import torch.multiprocessing as mp
import os
import swanlab
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import time
import utils.neural_network_model as utils_neural_network_model
import tqdm


def fast_collate_fn(batch):
    dates, stock_codes, features, labels, masks = zip(*batch)
    return (torch.stack(dates, dim=0), torch.stack(stock_codes, dim=0), torch.stack(features, dim=0), torch.stack(labels, dim=0), torch.stack(masks, dim=0))


def _train_single_fold(rank: int, model_param_dict, dataset, train_idx, val_idx, args: dict):
    """子进程：只训练 + 收集每轮指标，返回 best model + 完整日志历史"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda:0"

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args["batch_size"], shuffle=True, collate_fn=fast_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args["batch_size"], shuffle=False, collate_fn=fast_collate_fn, pin_memory=True)

    model = utils_neural_network_model.neural_network_model(
        input_dim=model_param_dict["input_dim"],
        hidden_dim=model_param_dict["hidden_dim"],
        output_dim=model_param_dict["output_dim"],
        model_type=model_param_dict["model_type"],
    )
    model.to(device)

    if args["criterion"] == "mae":
        criterion = torch.nn.L1Loss(reduction="none")
    elif args["criterion"] == "huber":
        criterion = torch.nn.SmoothL1Loss(reduction="none")
    elif args["criterion"] == "ic":
        criterion = utils_neural_network_model.ICLoss()
    elif args["criterion"] == "weighted_mse":
        criterion = utils_neural_network_model.WeightedMSELoss(alpha=1.0)
    else:
        criterion = torch.nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args["lr_decay_gamma"])

    best_val_loss = float("inf")
    best_model = None

    # 收集每轮的日志（主进程统一上传 swanlab）
    history = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}

    iterator = tqdm.trange(1, args["epochs"] + 1) if rank == 0 else range(1, args["epochs"] + 1)
    for epoch in iterator:
        # training
        model.train()
        train_loss = 0.0
        for _, _, features, labels, mask in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(features)
            if isinstance(outputs, (tuple, list)):
                pred_labels, score = outputs
                score = score.squeeze(-1)
                loss_horizon_raw = criterion(pred_labels, labels)
                loss_horizon = (loss_horizon_raw.mean(dim=-1) * mask).sum() / mask.sum()
                loss_horizon.backward(retain_graph=True)
                label_score = labels[:, :, -1].squeeze(-1)  # (B, N)
                loss_score_raw = criterion(score, label_score)  # (B, N)
                loss = (loss_score_raw * mask).sum() / mask.sum()
            else:
                loss_raw = criterion(outputs.squeeze(-1), labels.squeeze(-1))
                loss = (loss_raw * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for _, _, features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                outputs = model(features)
                if isinstance(outputs, (tuple, list)):
                    pred_labels, score = outputs
                    score = score.squeeze(-1)
                    label_score = labels[:, :, -1].squeeze(-1)  # (B, N)
                    loss_score_raw = criterion(score, label_score)  # (B, N)
                    loss = (loss_score_raw * mask).sum() / mask.sum()
                else:
                    loss_raw = criterion(outputs.squeeze(-1), labels.squeeze(-1))
                    loss = (loss_raw * mask).sum() / mask.sum()
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 记录到 history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
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

    return best_model_path, best_val_loss, history


def _worker(rank: int, queue: mp.Queue, model_param_dict, dataset, train_idx, val_idx, args: dict):
    print(f"[Worker {rank}] Entered _worker", flush=True)
    t0 = time.time()
    best_model_path, best_val_loss, history = _train_single_fold(rank, model_param_dict, dataset, train_idx, val_idx, args)
    print(f"[Worker {rank}] Training finished, dt={time.time()-t0:.2f}s", flush=True)
    queue.put((rank, best_model_path, best_val_loss, history))
    queue.close()


def train_neural_network_model_parallel(
    model_param_dict,
    dataset,
    logger,
    epochs: int = 200,
    learning_rate: float = 1e-4,
    criterion: str = "mse",
    model_save_dir: str = "/home/haris/results/models/",
    save_model: bool = True,
    project_name: str = "StockPredictor",
    period_index: int = 0,
    model_save_frequency: int = 5,
    use_swanlab: bool = True,
    k_folds: int = 4,
    lr_decay_gamma: float = 0.99,
    batch_size: int = 32,
    timestamp: str = "None",
):
    """
    Train a neural network model with parallel processing on 4 GPUs using KFold cross-validation.

    Parameters:
        model_param_dict (dict): Dictionary containing model parameters.
        dataset (Dataset): Dataset for training data (should yield date, stock_code, features, labels).
        logger (logging.Logger): Logger for process tracking.
        epochs (int): Number of epochs for training (default 200).
        learning_rate (float): Learning rate for optimizer (default 1e-4).
        criterion (str): Loss function to use (default "mse").
        model_save_dir (str): Directory to save trained models.
        save_model (bool): Whether to save model checkpoints after each epoch.
        project_name (str): Name of the project for logging and saving.
        period_index (int): Optional index for logging different prediction periods.
        model_save_frequency (int): Frequency (in epochs) to save the model.
        use_swanlab (bool): Whether to use SwanLab for logging results.
        k_folds (int): Number of folds for KFold cross-validation (default 4).
        lr_decay_gamma (float): Learning rate decay gamma.
        batch_size (int): Batch size for training (default 32).
        timestamp (str): Timestamp string for file naming.

    Returns:
        list of best torch.nn.Module: List of best models, one per fold.
    """
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
        "criterion": criterion,
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
        p = mp.Process(target=_worker, args=(fold, queue, model_param_dict, dataset, train_idx, val_idx, args))
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
                        f"train_loss/period_{period_index}_fold_{rank+1}": h["train_loss"][epoch],
                        f"val_loss/period_{period_index}_fold_{rank+1}": h["val_loss"][epoch],
                        f"lr/period_{period_index}_fold_{rank+1}": h["lr"][epoch],
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
