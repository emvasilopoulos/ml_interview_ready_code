import torch
import mnn.hyperparameters.config


def build_optimizer_from_config(
    optimizer_config: mnn.hyperparameters.config.OptimizerConfiguration,
):
    if optimizer_config.name == "Adam":
        return torch.optim.Adam(
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.name == "AdamW":
        return torch.optim.AdamW(
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
        )
    elif optimizer_config.name == "SGD":
        return torch.optim.SGD(
            lr=optimizer_config.learning_rate,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config.name}")
