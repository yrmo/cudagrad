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
    
    EPOCHS = 100
    lr = 0.0001
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
        if epoch % (EPOCHS // 10) == 0:
            print(f"{epoch=}", f"{loss.item()}")
            epochs.append(epoch)
            losses.append(loss.item())
            out0 = round(model(Tensor([2, 1], inputs[0])).item())
            out1 = round(model(Tensor([2, 1], inputs[1])).item())
            out2 = round(model(Tensor([2, 1], inputs[2])).item())
            out3 = round(model(Tensor([2, 1], inputs[3])).item())
            print("0 OR 0 = ", out0, "🔥" if out0 == 0 else "🌧️")
            print("0 OR 1 = ", out1, "🔥" if out1 == 1 else "🌧️")
            print("1 OR 0 = ", out2, "🔥" if out2 == 1 else "🌧️")
            print("1 OR 1 = ", out3, "🔥" if out3 == 1 else "🌧️")

plt.scatter(epochs, losses)
plt.title("Neuron trained on binary OR function")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("neuron.jpg")