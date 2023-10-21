from os import path
from lightning.pytorch.callbacks import RichProgressBar

def resolvePath(_path):
  _path = path.expandvars(_path)
  _path = path.expanduser(_path)
  _path = path.abspath(_path)
  return _path

class GlobalStepProgressBar(RichProgressBar):
    
    @property
    def total_train_batches(self):
        return self.trainer.estimated_stepping_batches * self.trainer.accumulate_grad_batches

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["global_step"] = trainer.global_step
        return items