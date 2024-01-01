import torch
import numpy as np

x = np.linspace(0,10,20, dtype="float32") 
y = 2 * x + 1
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
print("x:",x.shape)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()
print(model)

criterion = torch.nn.MSELoss(reduce="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if epoch % 50 == 0:
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w=", model.linear.weight.item())
print("b=", model.linear.bias.item())

x_test = torch.tensor([2.0])
y_test = model(x_test)
print("y_pred=", y_test)
print("y_pred=", y_test.data)