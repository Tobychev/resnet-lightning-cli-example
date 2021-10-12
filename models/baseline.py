import pytorch_lightning as lg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tm

from models.blocks import ResidualBlock,BottleneckResidualBlock

class DAWNNet(lg.LightningModule):
    def __init__(self, img_channels: int = 3,
                 first_kernel_size: int = 3,
                 batch_size: int = 128,
                 n_channels: int = 64):
        super().__init__()
        self.batch_size = batch_size
        # log hyperparameters
        self.save_hyperparameters()

        can = [n_channels, 2*n_channels, 4*n_channels, 4*n_channels]

        self.conv = nn.Conv2d(img_channels, n_channels,
                              kernel_size = first_kernel_size, stride = 1,
                              padding = first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels)

        blocks = [
            ResidualBlock(can[0],can[0],stride=1),
            ResidualBlock(can[0],can[0],stride=1),
            BottleneckResidualBlock(can[0],can[0],can[1],stride=2),
            ResidualBlock(can[1],can[1],stride=1),
            BottleneckResidualBlock(can[1],can[1],can[2],stride=2),
            ResidualBlock(can[2],can[2],stride=1),
            BottleneckResidualBlock(can[2],can[2],can[3],stride=2),
            ResidualBlock(can[3],can[3],stride=1),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.pool1 = torch.nn.MaxPool2d(4)
        self.pool2 = torch.nn.AvgPool2d(4)
        # Hardcode num outputs to 10 because it's for CIFAR >10<
        self.fc1 = nn.Linear(2*can[3], 10, bias=True)

    def _get_conv_output(self, shape):
        batch_size = 1
        inpt = torch.autograd.Variable(torch.rand(batch_size,*shape))
        output_feat = self._forward_features(inpt)
        return output_feat.data.view(batch_size,-1).size(1)

    def _forward_features(self,x):
        x = self.bn(self.conv(x))
        return self.blocks(x)

    def forward(self, x: torch.Tensor):
        x = self._forward_features(x)
        print(f"\n\nx {x.size()}")
        x1,x2 = self.pool1(x),self.pool2(x)

        x = torch.cat((x1,x2),dim=1)
        x = x.view(x.shape[0],x.shape[1],-1)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)

        loss = F.nll_loss(logits, y, reduction = "none")

        preds = torch.argmax(logits, dim = 1)
        acc = tm.accuracy(preds,y)

        self.log("test_loss",loss,prog_bar = True)
        self.log("test_acc", acc, prog_bar = True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr = 1e-4)

        # Attempt to match the piecewise linear schedule
        # [0,0.1,0.005,0] at epoch [0,15,30,35]
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optim,
            cycle_momentum = False,
            three_phase = True,
            anneal_strategy = "linear",
            max_lr = 0.1,
            pct_start = 0.4286,
            epochs = 35,
            steps_per_epoch = 45000//self.batch_size,
            div_factor = 20,
            final_div_factor = 1e6
        )

        return {"optimizer":optim,
                "scheduler": lr_schedule, "interval": "step"}
