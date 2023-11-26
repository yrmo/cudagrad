from random import random

from cudagrad.tensor import Tensor # type: ignore
import matplotlib.pyplot as plt

def mse(predicted: Tensor, actual: Tensor) -> Tensor:
    return (predicted - actual) * (predicted - actual)

class Linear:
    def __init__(self, inputs: int, outputs: int):
        self.w = Tensor([outputs, inputs], [random() for _ in range(outputs * inputs)])
        self.b = Tensor([outputs], [random() for _ in range(outputs)])

    def __call__(self, x: Tensor) -> Tensor:
        return (self.w @ x) + self.b


if __name__ == "__main__":
    # OR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 1]
    
    EPOCHS = 300
    lr = 0.001
    epochs = []
    losses = []
    model = Linear(2, 1)
    for epoch in range(EPOCHS + 1):
        for i, input in enumerate(inputs):
            model.w.zero_grad()
            model.b.zero_grad()
            loss = mse(Tensor([1], [targets[i]]), model(Tensor([2, 1], input)))
            loss.backward()
            model.w.data[[0, 0]] = model.w.data[[0, 0]].item() + (-lr * model.w.grad[0])
            model.w.data[[0, 1]] = model.w.data[[0, 1]].item() + (-lr * model.w.grad[1])
            model.b.data[[0, 0]] = model.b.data[[0, 0]].item() + (-lr * model.b.grad[0])
        if epoch % (EPOCHS // 100) == 0:
            print(f"{epoch=}", f"{loss.item()}")
            epochs.append(epoch)
            losses.append(loss.item())
            print(model(Tensor([2, 1], inputs[0])).item(), round(model(Tensor([2, 1], inputs[0])).item()) == targets[0])
            print(model(Tensor([2, 1], inputs[1])).item(), round(model(Tensor([2, 1], inputs[1])).item()) == targets[1])
            print(model(Tensor([2, 1], inputs[2])).item(), round(model(Tensor([2, 1], inputs[2])).item()) == targets[2])
            print(model(Tensor([2, 1], inputs[3])).item(), round(model(Tensor([2, 1], inputs[3])).item()) == targets[3])
            # print(model(Tensor([2, 1], inputs[i])))

plt.scatter(epochs, losses)
plt.title("Neuron trained on binary OR function")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("neuron.jpg")