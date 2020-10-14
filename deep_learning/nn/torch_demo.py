import torch
import torch.nn as nn
import numpy as np


def run():
    x = torch.tensor(np.mat('0 0; 0 1; 1 0; 1 1')).float()
    y = torch.tensor(np.mat('1; 0; 0; 1')).float()

    print(x)
    print(x.shape)

    mynet = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    print(mynet)

    optimzer = torch.optim.SGD(mynet.parameters(), lr=0.05)
    loss_func = nn.MSELoss()

    for epoch in range(5000):
        out = mynet(x)
        loss = loss_func(out, y)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

    print(mynet(x).data)


if __name__ == '__main__':
    run()
