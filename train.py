from src.datamodules.lightning_module import LitAutoEncoder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl


# data
dataset = CIFAR10(root="./", train=True, download=True, transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]))
print(len(dataset))
mnist_train, mnist_val = random_split(dataset, [45000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(accelerator="cpu", precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
    
