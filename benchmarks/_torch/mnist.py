from os import getenv

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


model = Model(28 * 28, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

if not PROFILING:
    num_row = 3
    num_col = 4
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 7))
    test_iter = iter(test_loader)
    images, labels = next(test_iter)

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)

    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i].squeeze(), cmap="viridis")
        ax.set_title(f"Pred: {predictions[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
