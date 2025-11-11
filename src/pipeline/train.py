import torch
from torch.utils.data import DataLoader
import tqdm
import os
from datetime import datetime

def train_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
        logger,
        epochs: int = 10,
        learning_rate: float = 0.001,
        model_save_dir: str = "/home/user0/results/models/",
        save_model: bool = True,
        device: str = 'cuda',
        project_name: str = 'StockPredictor',
        period_index: int = 0,
        model_save_frequency: int = 5,
    ) -> torch.nn.Module:
    """
    Train the given model using the provided DataLoader and optionally save checkpoints.

    Parameters:
        model (torch.nn.Module): Model to be trained
        dataloader (DataLoader): Data loader providing (date, stock_code, features, labels)
        logger: Python logger for recording training process
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        save_dir (str): Directory to save trained models
        save_model (bool): Whether to save model checkpoints after each epoch
        device (str): Device to use ('cuda' or 'cpu'), defaults to auto-detect
        project_name (str): Name of the project/experiment for logging purposes
        period_index (int): Index of the current training period
        model_save_frequency (int): Frequency (in epochs) to save the model

    Returns:
        torch.nn.Module: Trained model
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Training started on device: {device}")
    logger.info(f"Total epochs: {epochs}, Learning rate: {learning_rate}, Save model: {save_model}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for date, stock_code, features, labels in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", ncols=80):
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            logger.info(f"Epoch [{epoch}/{epochs}] - Average Loss: {avg_loss:.6f}")

            # Save model checkpoint
            if save_model and (epoch % model_save_frequency == 0 or epoch == epochs):
                os.makedirs(os.path.join(model_save_dir, f"{project_name}_{timestamp}_model"), exist_ok=True)
                model_path = os.path.join(model_save_dir, f"{project_name}_{timestamp}_model/{project_name}_{timestamp}_model_period_{period_index}_epoch{epoch}_loss{avg_loss:.6f}.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"Model saved: {model_path}")

        logger.info("Training completed successfully.")

    except Exception as e:
        logger.error(f"Training interrupted due to error: {e}", exc_info=True)

    return model
