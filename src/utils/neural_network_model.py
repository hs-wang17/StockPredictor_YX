import torch

def neural_network_model(input_dim: int, hidden_dim: int, output_dim: int, model_type: str = 'mlp') -> torch.nn.Module:
    """
    Define a simple neural network model.
    """
    if model_type == 'mlp':
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == 'cnn':
        model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim * input_dim, output_dim)
        )
    elif model_type == 'rnn':
        model = torch.nn.Sequential(
            torch.nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == 'transformer':
        model = torch.nn.Sequential(
            torch.nn.Transformer(d_model=input_dim, nhead=4, num_encoder_layers=2),
            torch.nn.Linear(input_dim, output_dim)
        )
    elif model_type == 'lstm':
        model = torch.nn.Sequential(
            torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

def save_neural_network_model(model: torch.nn.Module, file_path: str):
    """
    Save the trained neural network model to a file.
    """
    torch.save(model.state_dict(), file_path)

def load_neural_network_model(model: torch.nn.Module, file_path: str) -> torch.nn.Module:
    """
    Load a trained neural network model from a file.
    """
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model
