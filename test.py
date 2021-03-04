import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils import generate_model, get_logger, Metrics


def test(exp_name):
    print('loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test', size=opt.testsize)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)
    model, _, _ = generate_model(opt)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    logger = get_logger('./results/' + exp_name + '.log')
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
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

    print("Test Result:")
    logger.info('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))


if __name__ == '__main__':
    opt.mode = 'test'
    print('--- PolypSeg Test---')
    test(opt.exp_name)

    print('Done')
