from nanoGPT.model import GPTConfig, GPT, LayerNorm
import lightning as L
import torch

class NanoGPT(L.LightningModule):
    def __init__(self, model_args, learning_rate, weight_decay, betas, warmup_iters, lr_decay_iters, min_lr, compile_model=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(GPTConfig(**self.hparams.model_args))
        self.codex = None

    def _compile(self):
      if self.hparams.compile_model and not "OptimizedModule" in str(type(self.model)):
          self.model = torch.compile(self.model)

    def on_load_checkpoint(self, checkpoint):
      self._compile()
      self.codex = checkpoint['codex']
    
    def on_save_checkpoint(self, checkpoint):
      checkpoint['codex'] = self.codex

    def on_train_start(self) -> None:
      self._compile()
      self.codex = self.trainer.datamodule.codex

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        extra_args = dict(fused=True)
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas, **extra_args)

        return optimizer
    
    def lr_lambda(self, step):
        if step < self.hparams.warmup_iters:
            return float(step) / float(max(1, self.hparams.warmup_iters))
        return max(
            float(self.hparams.lr_decay_iters - step) / float(max(1, self.hparams.lr_decay_iters - self.hparams.warmup_iters)),
            float(self.hparams.min_lr) / float(self.hparams.learning_rate),
        )