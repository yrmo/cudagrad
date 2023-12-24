# In [1]: import torch
#    ...: import torch.nn as nn
#    ...: import torch.optim as optim
#    ...: 
#    ...: X = torch.tensor(
#    ...:     [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True
#    ...: )
#    ...: y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)
#    ...: 
#    ...: torch.manual_seed(1337)
#    ...: 
#    ...: 
#    ...: class MLP(nn.Module):
#    ...:     def __init__(self):
#    ...:         super().__init__()
#    ...:         self.layer1 = nn.Linear(2, 2)
#    ...:         self.layer2 = nn.Linear(2, 1)
#    ...: 
#    ...:     def forward(self, x):
#    ...:         x = torch.sigmoid(self.layer1(x))
#    ...:         x = torch.sigmoid(self.layer2(x))
#    ...:         return x
#    ...: 

# In [2]: model = MLP()

# In [3]: model(X)
# Out[3]: 
# tensor([[0.4216],
#         [0.4200],
#         [0.4205],
#         [0.4188]], grad_fn=<SigmoidBackward0>)

# In [4]: model.state_dict()
# Out[4]: 
# OrderedDict([('layer1.weight',
#               tensor([[-0.5963, -0.0062],
#                       [ 0.1741, -0.1097]])),
#              ('layer1.bias', tensor([-0.4237, -0.6666])),
#              ('layer2.weight', tensor([[0.1204, 0.2781]])),
#              ('layer2.bias', tensor([-0.4580]))])

# In [5]: def train(model_class: nn.Module) -> None:
#    ...:     model = model_class()
#    ...:     optimizer = optim.SGD(model.parameters(), lr=0.1)
#    ...:     criterion = nn.MSELoss()
#    ...: 
#    ...:     for epoch in range(25000):
#    ...:         outputs = model(X)
#    ...:         loss = criterion(outputs, y)
#    ...:         optimizer.zero_grad()
#    ...:         loss.backward()
#    ...:         optimizer.step()
#    ...: 
#    ...:         if epoch % 1000 == 0:
#    ...:             print(f"Epoch {epoch}, Loss: {loss.item()}")
#    ...: 
#    ...:     return model
#    ...: 

# In [6]: model = train(MLP)
# Epoch 0, Loss: 0.2516142725944519
# Epoch 1000, Loss: 0.24990874528884888
# Epoch 2000, Loss: 0.24985691905021667
# Epoch 3000, Loss: 0.24977990984916687
# Epoch 4000, Loss: 0.24965229630470276
# Epoch 5000, Loss: 0.24941614270210266
# Epoch 6000, Loss: 0.24892190098762512
# Epoch 7000, Loss: 0.24773162603378296
# Epoch 8000, Loss: 0.24447143077850342
# Epoch 9000, Loss: 0.23555763065814972
# Epoch 10000, Loss: 0.21834306418895721
# Epoch 11000, Loss: 0.1999143660068512
# Epoch 12000, Loss: 0.18157212436199188
# Epoch 13000, Loss: 0.14327096939086914
# Epoch 14000, Loss: 0.08178623020648956
# Epoch 15000, Loss: 0.045810267329216
# Epoch 16000, Loss: 0.029305540025234222
# Epoch 17000, Loss: 0.020791832357645035
# Epoch 18000, Loss: 0.015820659697055817
# Epoch 19000, Loss: 0.01263463869690895
# Epoch 20000, Loss: 0.010447653941810131
# Epoch 21000, Loss: 0.008866837248206139
# Epoch 22000, Loss: 0.007677637040615082
# Epoch 23000, Loss: 0.00675432663410902
# Epoch 24000, Loss: 0.006018902640789747

# In [7]: model(X)
# Out[7]: 
# tensor([[0.0713],
#         [0.9198],
#         [0.9220],
#         [0.0639]], grad_fn=<SigmoidBackward0>)

# In [8]: model.state_dict()
# Out[8]: 
# OrderedDict([('layer1.weight',
#               tensor([[-6.0408,  6.1632],
#                       [ 5.3242, -5.0834]])),
#              ('layer1.bias', tensor([3.1208, 2.5703])),
#              ('layer2.weight', tensor([[-6.0373, -6.1604]])),
#              ('layer2.bias', tensor([8.9376]))])

# In [9]: 