from dataclasses import dataclass
from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-3


class Trainer:
    """Minimal training loop for classification."""

    def __init__(self, model: nn.Module, device: torch.device, cfg: TrainConfig) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def fit(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.to(self.device)
        self.model.train()
        last_loss = 0.0
        for _ in range(self.cfg.epochs):
            running = 0.0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                running += loss.item()
            last_loss = running / max(len(train_loader), 1)
        return {"loss": last_loss}
