import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from opt import opt
from utils import generate_model, get_logger, Metrics, clip_gradient, evaluate
import torch.nn.functional as F


def valid(model, valid_dataloader, total_batch):
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean
                           )

    metrics_result = metrics.mean(total_batch)
    model.train()

    return metrics_result


def train(exp_name):
    model, optimizer, best_f1 = generate_model(opt)

    # load data
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train', size=opt.trainsize)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    if opt.val_period > 0:
        valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='valid', size=opt.testsize)
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        val_total_batch = int(len(valid_data) / 1)

    # load optimizer and scheduler
    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.nEpoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = opt.criterion

    # train
    logger = get_logger('./logs/' + exp_name + '.log')
    logger.info('start training!')

    iter = 0
    for epoch in range(opt.start_epoch, opt.nEpoch + 1):

        total_batch = int(len(train_data) / opt.batch_size)
        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        mean_loss = 0.
        for i, data in bar:
            iter += opt.batch_size
            img = data['image']
            gt = data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()

            if opt.multiscale:
                assert opt.size_rates[-1] in [1, 1.0]
                for rate in opt.size_rates[:-1]:
                    # images = F.interpolate(img, scale_factor=rate, mode='bilinear', align_corners=True)
                    # gts = F.interpolate(gt, scale_factor=rate, mode='bilinear', align_corners=True)
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        images = F.upsample(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = F.upsample(gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                    output = model(images)
                    loss = criterion(output, gts)
                    loss.backward()
                    if opt.clip_gradient:
                        clip_gradient(optimizer, opt.clip)
                    optimizer.step()

            output = model(img)
            loss = criterion(output, gt)
            loss.backward()
            if opt.clip_gradient:
                clip_gradient(optimizer, opt.clip)
            optimizer.step()
            bar.set_postfix_str('loss:%.5s ' % loss.item())
            mean_loss += loss.item()

            if opt.val_period > 0 and iter % opt.val_period == 0:
                metrics_result = valid(model, valid_dataloader, val_total_batch)
                logger.info('-------------------Validation-------------------')
                logger.info('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
                            ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f,'
                            ' F1_best: %.4f'
                            % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                               metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                               metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'],
                               best_f1))
                logger.info('-------------------Validation-------------------')
                if metrics_result['F1'] > best_f1:
                    best_f1 = metrics_result['F1']
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'epoch': epoch, 'best_f1': best_f1}
                    torch.save(state, './checkpoints/' + exp_name + "/best.pth")
                    print('checkpoint has been saved!')

        logger.info('epoch:%d/%d || mean_loss:%.5s' % (epoch, opt.nEpoch, mean_loss / (i + 1)))
        scheduler.step()
    if opt.val_period > 0:
        metrics_result = valid(model, valid_dataloader, val_total_batch)
        logger.info('-------------------Validation-------------------')
        logger.info('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
                    ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
                    ' F1_best: %.4f'
                    % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                       metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                       metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'],
                                   best_f1))
        logger.info('-------------------Validation-------------------')
        if metrics_result['F1'] > best_f1:
            best_f1 = metrics_result['F1']
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch, 'best_f1': best_f1}
            torch.save(state, './checkpoints/' + exp_name + "/best.pth")
            print('checkpoint has been saved!')

    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': epoch, 'best_f1': None}
    torch.save(state, './checkpoints/' + exp_name + "/latest.pth")
    print('checkpoint has been saved!')
    logger.info('finish training!')



if __name__ == '__main__':
    if not os.path.exists('./checkpoints/' + opt.exp_name):
        os.makedirs('./checkpoints/' + opt.exp_name)
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if opt.mode == 'train':
        train(opt.exp_name)
