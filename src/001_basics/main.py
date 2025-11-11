import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        return self.a + self.b * x


class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def make_train_step(model, loss_fn, optimizer):
    def train_step(x: Tensor, y: Tensor):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return train_step


if __name__ == "__main__":
    # Data Generation
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    device = "mps" if torch.mps.is_available() else "cpu"

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [80, 20])
    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    torch.manual_seed(42)

    models = {
        "manual": ManualLinearRegression().to(device),
        "layer": LayerLinearRegression().to(device),
        "seq": nn.Sequential(nn.Linear(1, 1)).to(device),
    }

    lr = 1e-1
    n_epochs = 1000

    optimizers = {
        name: optim.SGD(model.parameters(), lr=lr) for name, model in models.items()
    }

    loss_fn = nn.MSELoss(reduction="mean")

    for model_name in models.keys():
        model = models[model_name]
        optimizer = optimizers[model_name]
        train_step = make_train_step(model, loss_fn, optimizer)

        losses = []
        val_losses = []

        for epoch in range(n_epochs):
            for x_batch, y_batch in train_loader:
                # the dataset "lives" in the CPU, so do our mini-batches
                # therefore, we need to send those mini-batches to the
                # device where the model "lives"
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                loss = train_step(x_batch, y_batch)
                losses.append(loss)

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        model.eval()
                        yhat = model(x_val)
                        val_loss = loss_fn(y_val, yhat)
                        val_losses.append(val_loss.item())

        print(f"Model: {model_name}")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.item():.4f}")
            print(f"  Final Training Loss: {losses[-1]:.4f}")
            print(f"  Final Validation Loss: {val_losses[-1]:.4f}")
