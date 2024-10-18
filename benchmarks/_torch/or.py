import torch
import torch.nn as nn
import torch.optim as optim


class Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # OR
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

    EPOCHS = 1000
    lr = 0.0001
    epochs = []
    losses = []
    model = Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS + 1):
        for i, input in enumerate(inputs):
            optimizer.zero_grad()
            output = model(input.unsqueeze(0))
            loss = criterion(output, targets[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

        if epoch % (EPOCHS // 10) == 0:
            print(f"{epoch=}", f"{loss.item()}")
            epochs.append(epoch)
            losses.append(loss.item())
            print(inputs[0])
            out0 = round(model(inputs[0].unsqueeze(0)).item())
            out1 = round(model(inputs[1].unsqueeze(0)).item())
            out2 = round(model(inputs[2].unsqueeze(0)).item())
            out3 = round(model(inputs[3].unsqueeze(0)).item())
            print("0 OR 0 = ", round(model(inputs[0].unsqueeze(0)).item(), 2), "🔥" if out0 == 0 else "")
            print("0 OR 1 = ", round(model(inputs[1].unsqueeze(0)).item(), 2), "🔥" if out1 == 1 else "")
            print("1 OR 0 = ", round(model(inputs[2].unsqueeze(0)).item(), 2), "🔥" if out2 == 1 else "")
            print("1 OR 1 = ", round(model(inputs[3].unsqueeze(0)).item(), 2), "🔥" if out3 == 1 else "")
