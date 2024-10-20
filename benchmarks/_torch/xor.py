import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    # XOR
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    EPOCHS = 5000
    lr = 0.05

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

            outputs = model(inputs).round()
            for i, (input_data, output) in enumerate(zip(inputs, outputs)):
                correct = "🔥" if output.item() == targets[i].item() else ""
                print(f"{input_data.tolist()} XOR = {output.item():.2f} {correct}")
