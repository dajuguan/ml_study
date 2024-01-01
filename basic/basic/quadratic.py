import torch
import numpy as np

x = np.linspace(0,10,50, dtype="float32") 
y = x**2 + 1
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
print("x:",x.shape)

class QuadraticModel(torch.nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.l1 = torch.nn.Linear(1,100)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(100,100)
        self.l3 = torch.nn.Linear(100,1)
    def forward(self, x):
        relu = self.relu
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        y_pred = self.l3(x)
        return y_pred

model = QuadraticModel()
print(model)

criterion = torch.nn.MSELoss(reduce="sum")
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9) # SGD is not stable as adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

for epoch in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if epoch % 50 == 0:
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.tensor([2.0])
y_test = model(x_test)
print("y_pred=", y_test)
print("y_pred=", y_test.data)