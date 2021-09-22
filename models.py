import torch
import copy
from pathlib import Path


class MyModel:
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.network = MyNaiveNetwork(opt)
        if opt.continue_train:
            self.load_network("latest")
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, 0.999),
                                          weight_decay=1e-5)
        self.scheduler = self.get_scheduler()
        self.network.to(self.opt.device)

    def get_scheduler(self):
        if self.opt.lr_policy == 'lambda':
            lambda_rule = lambda epoch: self.opt.lr_decay_rate ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_policy == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=5)
        else:
            raise NotImplementedError
        return scheduler

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '{}_net.pth'.format(str(which_epoch))
        load_path = Path.joinpath(self.opt.experiment_path, save_filename)
        net = self.network
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(str(load_path)))
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '{}_net.pth'.format(str(which_epoch))
        save_path = Path.joinpath(self.opt.experiment_path, save_filename)
        torch.save(self.network.cpu().state_dict(), str(save_path))
        self.network.to(self.opt.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


class MyNaiveNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(MyNaiveNetwork, self).__init__()
        self.opt = copy.deepcopy(opt)
        if opt.architecture == "fc":
            fc_network = FullyConnected(self.opt.number_of_classes)
            self.net = fc_network.network
        else:
            raise NotImplementedError
        if self.opt.loss == "cat_cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x


class FullyConnected:
    def __init__(self, args):
        self.network = torch.nn.ModuleList([
            torch.nn.Flatten(),
            torch.nn.Linear((args.image_size**2)*3*args.frames_per_datapoint, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, args.number_of_classes),
        ])