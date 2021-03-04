import models
import torch
import logging
import os

def generate_model(opt):
    model = getattr(models, opt.model)(opt.nclasses)
    best_f1 = 0.
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    if opt.continue_train or opt.mode == 'test':
        model_dict = model.state_dict()
        print('Loading checkpoint......')
        try:
            checkpoint = torch.load('./checkpoints/' + opt.exp_name + '/best.pth', map_location='cpu')
        except FileNotFoundError:
            checkpoint = torch.load('./checkpoints/' + opt.exp_name + '/latest.pth', map_location='cpu')

        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(checkpoint['net'])

        opt.start_epoch = checkpoint['epoch'] + 1

        print('Loaded from epoch %d' % (opt.start_epoch - 1))

        optimizer.load_state_dict(checkpoint['optimizer'])
        best_f1 = checkpoint['best_f1']
    return model, optimizer, best_f1


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        " %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



