import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms

class ResNetTransfer(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features=self.model.fc.in_features
    self.model.fc=nn.Linear(num_features,2)
  def forward(self,x):
    return self.model(x)
  def training_step(self, batch, batch_idx):
    image,labels=batch
    outputs=self(image)
    loss=F.cross_entropy(outputs,labels)
    self.log("train_loss", loss, on_step=True, on_epoch=True)
    self.log("train_acc", accuracy(outputs,labels,task="multiclass", num_classes=2), on_step=True, on_epoch=True)
    return loss
  def validation_step(self, batch, batch_idx):
    image,labels=batch
    outputs=self(image)
    loss=F.cross_entropy(outputs,labels)
    self.log("val_loss", loss, on_step=False, on_epoch=True)
    self.log("val_acc", accuracy(outputs,labels,task="multiclass", num_classes=2), on_step=False, on_epoch=True)
    return loss
  def test_step(self, batch, batch_idx):
    image,labels=batch
    outputs=self(image)
    loss=F.cross_entropy(outputs,labels)
    self.log("test_loss", loss, on_step=False, on_epoch=True)
    self.log("test_acc", accuracy(outputs,labels,task="multiclass", num_classes=2), on_step=False, on_epoch=True)
    return loss
  def configure_optimizers(self):
    optimizer=torch.optim.Adam(self.model.fc.parameters(), lr=0.001)
    return optimizer

if __name__=="__main__":
  train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229, 0.224, 0.225))
  ])
  val_test_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229, 0.224, 0.225))
  ])

  full_dataset=ImageFolder(root="data")

  n_train=int(len(full_dataset)*0.8)
  n_val=int(len(full_dataset)*0.1)
  n_test=len(full_dataset)-n_train-n_val
  torch.manual_seed(42)
  train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(full_dataset,[n_train,n_val,n_test])

  train_dataset.dataset.transform=train_transform
  val_dataset.dataset.transform=val_test_transform
  test_dataset.dataset.transform=val_test_transform

  train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
  val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=4)
  test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4)

  model=ResNetTransfer()
  pl.seed_everything(42)
  trainer=pl.Trainer(max_epochs=30, accelerator="cpu", devices=1)
  trainer.fit(model, train_loader, val_loader)
  print(trainer.callback_metrics)
  results=trainer.test(dataloaders=test_loader)
  print(results)

  torch.save(model.state_dict(), "dog_cat_classifier.pt")