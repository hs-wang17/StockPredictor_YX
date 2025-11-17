import torch
from torch.utils.data import DataLoader, Subset
import tqdm
import os
from datetime import datetime
import json
import swanlab
import numpy as np
import utils.neural_network_model as utils_neural_network_model
from sklearn.model_selection import KFold


def train_neural_network_model(
    model,
    dataset,
    logger,
    epochs: int = 200,
    learning_rate: float = 1e-5,
    model_save_dir: str = "/home/user0/results/models/",
    save_model: bool = True,
    device: str = "cuda",
    project_name: str = "StockPredictor",
    period_index: int = 0,
    model_save_frequency: int = 5,
    use_swanlab: bool = True,
    k_folds: int = 4,
    lr_decay_gamma: float = 0.99,
    batch_size: int = 32,
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
) -> tuple:
    """
    Train the model with K-fold cross-validation, validation loss logging, and learning rate decay.
    Returns:
        best_models: list of best model state_dict per fold
        fold_results: list of training history dict per fold
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training started on device: {device}")
    logger.info(f"Total epochs: {epochs}, Learning rate: {learning_rate}, Save model: {save_model}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_models = []  # 保存每折的最佳模型

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), start=1):
        logger.info(f"Starting fold {fold_idx}/{k_folds}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 每折创建新的模型、优化器和调度器
        model = model.to(device)
        # criterion = utils_neural_network_model.ICLoss()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            grad_norms = []

            for date, stock_code, features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()

                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(grad_norm**0.5)

                optimizer.step()
                train_loss += loss.item()

            scheduler.step()  # 学习率衰减

            avg_train_loss = train_loss / len(train_loader)
            avg_grad_norm = float(np.mean(grad_norms))

            # 验证集
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for date, stock_code, features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    val_loss += criterion(outputs.squeeze(), labels.squeeze()).item()
            avg_val_loss = val_loss / len(val_loader)

            logger.info(
                f"Fold {fold_idx} Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Grad Norm: {avg_grad_norm:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}"
            )

            if use_swanlab:
                swanlab.log(
                    {
                        f"train_loss/period_{period_index}_fold_{fold_idx}": avg_train_loss,
                        f"val_loss/period_{period_index}_fold_{fold_idx}": avg_val_loss,
                        f"train_grad_norm/period_{period_index}_fold_{fold_idx}": avg_grad_norm,
                    }
                )

            # 保存每折最优模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model

            # 保存 checkpoint
            if save_model and (epoch % model_save_frequency == 0 or epoch == epochs):
                fold_model_dir = os.path.join(model_save_dir, f"{project_name}_{timestamp}_fold{fold_idx}_model")
                os.makedirs(fold_model_dir, exist_ok=True)
                model_path = os.path.join(fold_model_dir, f"{project_name}_{timestamp}_fold{fold_idx}_epoch{epoch}.pt")
                torch.save(model, model_path)
                logger.info(f"Model saved: {model_path}")

                checkpoint_path = os.path.join(model_save_dir, f"{project_name}_{timestamp}_fold{fold_idx}_checkpoint.json")
                if os.path.exists(checkpoint_path):
                    with open(checkpoint_path, "r") as f:
                        checkpoint_list = json.load(f)
                else:
                    checkpoint_list = []
                checkpoint_list.append(
                    {
                        "project_name": project_name,
                        "timestamp": timestamp,
                        "fold": fold_idx,
                        "period_index": period_index,
                        "epochs_completed": epoch,
                        "latest_model_path": model_path,
                    }
                )
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_list, f, indent=4)
                logger.info(f"Checkpoint updated: {checkpoint_path}")

        best_models.append(best_model)

    logger.info("Training with K-fold cross-validation completed successfully.")
    return best_models
