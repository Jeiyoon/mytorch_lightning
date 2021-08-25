"""
reimplementation and comment: Jeiyoon
"""
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

# data
# Lightning operates on pure dataloaders. Here’s the PyTorch code for loading MNIST.
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
"""
[Disclaimer]
Notice this is a lightning module instead of a torch.nn.Module. 
A LightningModule is equivalent to a pure PyTorch Module except it has added functionality. 
However, you can use it EXACTLY the same as you would a PyTorch Module.
"""
class LitMNIST(LightningModule):
  def __init__(self):
    super().__init__()
    # mnist images are (1, 28, 28) (channels, height, width)
    self.layer_1 = nn.Linear(28 * 28, 128)
    self.layer_2 = nn.Linear(128, 256)
    self.layer_3 = nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, height, width = x.size()

    # (b, 1, 28, 28) -> (b, 1 * 28 * 28)
    x = x.view(batch_size, -1)
    x = self.layer_1(x)
    x = F.relu(x)
    x = self.layer_2(x)
    x = F.relu(x)
    x = self.layer_3(x)

    x = F.log_softmax(x, dim = 1)

    return x

  # add training_step
  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y) # negative log likelihood

    return loss

# net = LitMNIST()
# x = torch.randn(1, 1, 28, 28)
# out = net(x)
# print(x.size())
# print(out.size())

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]) # Normalize(mean=(0.1307,), std=(0.3081,))

# data
# os.getcwd(): current directory
mnist_train = MNIST(os.getcwd(), train = True, download = True,
                    transform = transform)
mnist_train = DataLoader(mnist_train, batch_size = 64)

# You can use DataLoaders in 3 ways:

# (1) Pass in the dataloaders to the .fit() function.
"""
model = ListMNIST()
trainer = Trainer()
trainer.fit(model, mnist_train)
"""

# (2) LightningModule DataLoaders
# For fast research prototyping, it might be easier to link the model with the dataloaders.
"""
class LitMNIST(pl.LightningModule):
    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # data
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=64)

    def val_dataloader(self):
        transforms = ...
        mnist_val = ...
        return DataLoader(mnist_val, batch_size=64)

    def test_dataloader(self):
        transforms = ...
        mnist_test = ...
        return DataLoader(mnist_test, batch_size=64)
"""







