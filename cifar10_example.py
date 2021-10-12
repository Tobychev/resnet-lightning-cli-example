import models.baseline as mb
import models.loaders as ml
import models.validation as mv
import pytorch_lightning as lg
import pytorch_lightning.callbacks as lc
import pytorch_lightning.utilities.cli as lui
import time
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

MODEL_CHKP_PATH = "logs/checkpoints"

#now = time.localtime()
#start_time = f"{now.tm_year}-{now.tm_mon:02d}-{now.tm_mday:02d}at{now.tm_hour:02d}h{now.tm_min:02d}m"

early_stop_cb = EarlyStopping(
    monitor = "val_loss",
    patience = 4,
    verbose = False,
    mode = "min"
)

checkpoint_cb = lc.ModelCheckpoint(
    monitor = "val_loss",
    save_top_k = 3,
    dirpath = MODEL_CHKP_PATH,
    filename = "model-{epoch:02d}-{val_loss:.3f}",
    mode = "min"
    )

cifar = ml.CIFAR10DataModule(batch_size = 128, data_dir="data/")
#cifar.prepare_data()
#cifar.setup()

#val_samples = next(iter(cifar.val_dataloader()))

mod = mb.DAWNNet(img_channels = 3,
                 first_kernel_size = 7,
                 batch_size = 128,
                 n_channels = 64)
"""
trainer = lg.Trainer(max_epochs = 35,
                     progress_bar_refresh_rate = 30,
                     gpus = 0,
                     callbacks = [early_stop_cb,
                                  checkpoint_cb,
                                  mv.ImagePredictionLogger(val_samples)],
                     checkpoint_callback = True)
"""
mb.DAWNNet
cli = lui.LightningCLI(mb.DAWNNet,ml.CIFAR10DataModule)
#trainer.fit(mod)
#trainer.finish()
