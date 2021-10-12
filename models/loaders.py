import os
import pytorch_lightning as lg
import torchvision as tv
import torchvision.transforms as tr
import torch.utils.data as ta

class CIFAR10DataModule(lg.LightningDataModule):
   def __init__(self, batch_size, data_dir):
      super().__init__()
      if os.path.isdir(data_dir):
         self.data_dir = data_dir
      else:
         raise ValueError(f"Path {data_dir} is not a directory")

      self.batch_size = batch_size

      self.transform = tr.Compose([
         tr.ToTensor(),
         tr.Normalize(mean=(125.31, 122.95, 113.87),
                      std=(62.99, 62.09, 66.70))
      ])

      self.dims = (3,32,32)
      self.num_classes = 10

   def prepare_data(self):
      tv.datasets.CIFAR10(self.data_dir,download=True,train=True)
      tv.datasets.CIFAR10(self.data_dir,download=True,train=False)

   def setup(self, stage = None):
      if stage == "fit":
         cifar_full = tv.datasets.CIFAR10(self.data_dir,train = "True", transform = self.transform)
         self.cifar_train, self.cifar_val = ta.random_split(cifar_full, [45000,5000])
      elif stage == "test":
         self.cifar_test = tv.datasets.CIFAR10(self.data_dir,train = "True", transform = self.transform)
      elif stage is None:
         cifar_train = tv.datasets.CIFAR10(self.data_dir,train = "True", transform = self.transform)
         self.cifar_train, self.cifar_val = ta.random_split(cifar_train, [45000,5000])
         self.cifar_test = tv.datasets.CIFAR10(self.data_dir,train = "True", transform = self.transform)

   def train_dataloader(self):
      return ta.DataLoader(self.cifar_train,batch_size = self.batch_size, shuffle = True)

   def val_dataloader(self):
      return ta.DataLoader(self.cifar_val, batch_size = self.batch_size, shuffle = False)

   def test_dataloader(self):
      return DataLoader(self.cifar_test, batch_size = self.batch_size, shuffle = False)
