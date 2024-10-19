import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import os

PROFILING = int(os.getenv("PROFILING", "0"))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    # two moons
    inputs, targets = make_moons(n_samples=100, noise=0.2)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    EPOCHS = 500
    lr = 0.1

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    epochs = []
    losses = []

    for epoch in range(EPOCHS + 1):
        epoch_loss = 0.0
        for i in range(inputs.size(0)):
            input_data = inputs[i].unsqueeze(0)
            target = targets[i].unsqueeze(0)

            output = model(input_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % (EPOCHS // 10) == 0:
            avg_loss = epoch_loss / inputs.size(0)
            print(f"epoch={epoch}, loss={avg_loss:.4f}")
            epochs.append(epoch)
            losses.append(avg_loss)

            with torch.no_grad():
                outputs = model(inputs).round()
                accuracy = (outputs.eq(targets).sum().item() / len(targets)) * 100
                print(f"Accuracy: {accuracy:.2f}%")
                print("".join(["🔥" if pred == target else " " for pred, target in zip(outputs, targets)]))
