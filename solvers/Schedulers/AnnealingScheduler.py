from torch.optim.lr_scheduler import _LRScheduler

# Set the learning rate at a given training step with annealing

class AnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6, *args, **kwargs):
        self.optimizer = optimizer
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        super(AnnealingScheduler, self).__init__(self.optimizer, *args, **kwargs)

    def get_lr(self):
        interp_lr = self.mu_f + (self.mu_i - self.mu_f) * (1.0 - self.last_epoch / self.n)
        lrs = [max(interp_lr, self.mu_f) for base_lr in self.base_lrs]
        return lrs


