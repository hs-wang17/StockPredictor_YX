import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import tqdm

def train_model(model: torch.nn.Module, dataloader: DataLoader, logger, epochs: int = 10, learning_rate: float = 0.1) -> torch.nn.Module:
    """
    Train the given model using the provided DataLoader.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in tqdm.trange(epochs):
        for date, stock_code, features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model
