import torch
import pytorch_lightning.callbacks as lc

class ImagePredictionLogger(lc.Callback):

   def __init__(self, val_samples, num_samples = 32):
      super().__init__()

      self.num_samples = num_samples
      self.val_imgs, self.val_labels = val_samples

   def on_validation_epoch_end(self, trainer, pl_module):
      val_imgs = self.val_imgs.to(device = pl_module.device)
      val_labels = self.val_labels.to(device = pl_module.device)

      logits = pl_module(val_imgs)
      preds = torch.argmax(logits,-1)
      for x,pred,y in zip(val_imgs[:self.num_samples],
                          preds[:self.num_samples],
                          val_labels[:self.num_samples]):
         trainer.logger.experiment.add_image(f"Pred:{pred}, Label:{y}",x,global_step=self.current_epoch)
