import os
import os.path as osp
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class ETIS(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, size=None):
        super(ETIS, self).__init__()
        data_path = osp.join(root, data2_dir)
        self.imglist = glob.glob(data_path + '/image/*.tif')
        self.gtlist = glob.glob(data_path + '/gtpolyp/*.tif')

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((256, 256)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((256, 256)),
                    ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)
        data['path'] = img_path
        return data

    def __len__(self):
        return len(self.imglist)
