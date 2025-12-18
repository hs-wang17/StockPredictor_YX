import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        numerator = (pred_mean * target_mean).sum()
        denominator = torch.sqrt((pred_mean**2).sum() * (target_mean**2).sum() + 1e-8)
        ic = numerator / denominator
        loss = 1 - ic
        return loss


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)  # LayerNorm 替代 BatchNorm
        self.linear2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.ln1(self.linear1(x)))
        x = self.ln2(self.linear2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResNetModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim * 4)
        self.input_ln = nn.LayerNorm(hidden_dim * 4)
        self.res_blocks = nn.Sequential(ResBlock(hidden_dim * 4, hidden_dim))
        self.output = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, output_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        x = self.output(x)
        return x


def neural_network_model(input_dim: int, hidden_dim: int, output_dim: int, model_type: str = "mlp") -> torch.nn.Module:
    """
    Define a simple neural network model.
    """
    if model_type == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    elif model_type == "resnet":
        model = ResNetModel(input_dim, hidden_dim, output_dim)
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
