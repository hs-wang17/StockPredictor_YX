import torch
import torch.nn as nn
import torch.nn.functional as F


# Custom Loss Functions
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


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        weight = (pred.abs().detach() + self.eps) ** self.alpha
        return (weight * base_loss).mean()


class RankWeightedMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    @staticmethod
    def _rank_normalize(x):
        ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()  # argsort twice <=> rank
        max_rank = x.size(-1) - 1
        return ranks / max(max_rank, 1)

    def forward(self, pred, target, mask=None):
        base_loss = self.mse(pred, target)
        with torch.no_grad():
            rank = self._rank_normalize(target)
            weight = 1.0 + 2.0 * torch.abs(rank - 0.5)
            if mask is not None:
                weight = weight * mask
        loss = weight * base_loss
        if mask is not None:
            return loss.sum() / (mask.sum() + self.eps)
        else:
            return loss.mean()


class HybridWeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, threshold=1.28, penalty_weight=5.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.penalty_weight = penalty_weight
        self.eps = eps

    def forward(self, pred, target):
        base_loss = 0.5 * (pred - target) ** 2
        # 权重 A: 预测值绝对值越大，权重越高 (鼓励模型关注大幅波动的预测)
        weight_pred = (pred.abs().detach() + self.eps) ** self.alpha
        # 权重 B: 真正的高收益股票，权重越高 (强制模型学准头部样本)
        weight_target = torch.where(target > self.threshold, float(self.penalty_weight), 1.0)
        # 合并权重
        final_loss = base_loss * weight_pred * weight_target
        return final_loss.mean()


# Neural Network Models
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


class ResNetBackboneModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, label_dim=4):
        super(ResNetBackboneModel, self).__init__()
        # ===== Backbone =====
        self.input_linear = nn.Linear(input_dim, hidden_dim * 4)
        self.input_ln = nn.LayerNorm(hidden_dim * 4)
        self.res_blocks = nn.Sequential(ResBlock(hidden_dim * 4, hidden_dim))
        # ===== Head 1: label prediction =====
        self.horizon_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, label_dim))
        # ===== Head 2: score aggregation =====
        self.score_head = nn.Linear(label_dim, output_dim)
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
        """
        x: (B, N, input_dim)
        """
        x = self.input_linear(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        horizon_pred = self.horizon_head(x)  # (B, N, label_dim)
        score = self.score_head(horizon_pred)  # (B, N, output_dim)
        return horizon_pred, score.squeeze(-1)


class SEBlock(torch.nn.Module):
    def __init__(self, dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(dim, dim // reduction), torch.nn.ReLU(), torch.nn.Linear(dim // reduction, dim), torch.nn.Sigmoid())

    def forward(self, x):
        # x: (batch, dim)
        w = self.fc(x)
        return x * w


class ResBlockWithSE(torch.nn.Module):
    def __init__(self, dim):
        super(ResBlockWithSE, self).__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()
        self.se = SEBlock(dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.se(out)
        return self.relu(out + residual)


class ResNetAttentionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(ResNetAttentionModel, self).__init__()
        self.input_fc = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(*[ResBlockWithSE(hidden_dim) for _ in range(num_blocks)])
        self.output_fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.blocks(x)
        return self.output_fc(x)


class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


# Model Utility Functions
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
    elif model_type == "resnet_backbone":
        model = ResNetBackboneModel(input_dim, hidden_dim, output_dim)
    elif model_type == "resnet_attention":
        model = ResNetAttentionModel(input_dim, hidden_dim, output_dim)
    elif model_type == "gru":
        model = GRUModel(input_dim, hidden_dim, output_dim)
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
