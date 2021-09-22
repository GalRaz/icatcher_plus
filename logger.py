from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.writer = SummaryWriter(log_dir=opt.root)

    def write_scaler(self, category, name, scalar_value, iterations):
        final_name = category + "/" + name
        self.writer.add_scalar(final_name, scalar_value, iterations)

    def close(self):
        if self.writer is not None:
            self.writer.close()