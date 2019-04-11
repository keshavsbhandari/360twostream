from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

get_video_name = lambda x: x.split('/')[-2]


# keys = ['images','flowu','flowv','video','class','parentclass','label']

class egocentric_dataset(Dataset):

    def __init__(self,
                 data_list_dict,
                 img_stack=10,
                 transform=None,
                 rows=320,
                 cols=640,
                 resizerow=224,
                 resizecol=224,
                 num_classes=63,
                 istest=False,
                 mode='motion'
                 ):
        self.data = data_list_dict
        self.transform = transform
        self.img_stack = img_stack
        self.img_rows = rows
        self.img_cols = cols
        self.resizerow = resizerow
        self.resizecol = resizecol
        self.num_classes = num_classes
        self.istest = istest
        self.mode = mode

    def stackopf(self, data):
        flow = torch.FloatTensor(2 * self.img_stack, self.resizerow, self.resizecol)

        label = {'class': data['class'],
                 'label': data['label'],
                 'superclass': data['parentclass'],
                 'vidname': data['video']}

        for i, (u, v) in enumerate(zip(data['flowu'], data['flowv'])):
            U = self.transform(Image.fromarray(cv2.imread(u, cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(v, cv2.IMREAD_GRAYSCALE)))
            flow[2 * i, :, :] = U
            flow[2 * i + 1, :, :] = V
        return flow, label

    def __len__(self):
        return len(self.data)

    def getimages(self, data):
        label = {'class': data['class'],
                 'label': data['label'],
                 'superclass': data['parentclass'],
                 'vidname': data['video']}

        imgs = torch.FloatTensor(3 * self.img_stack, self.resizerow, self.resizecol)

        for i, img_path in enumerate(data['images']):
            image = self.transform(Image.fromarray(cv2.imread(img_path)))
            imgs[3 * i, :, :] = image[0]
            imgs[3 * i + 1, :, :] = image[1]
            imgs[3 * i + 1 + 1, :, :] = image[2]

        return imgs, label

    def get_both(self,data):
        flows = self.stackopf(data)[0]
        imgs = self.stackopf(data)
        label = imgs[1]
        imgs = imgs[0]
        return flows, imgs, label

    def __getitem__(self, idx):
        data = self.data[idx]
        return {'motion': self.stackopf, 'spatial': self.getimages, 'both':self.get_both}.get(self.mode)(data)


class EgoCentricDataLoader():
    def __init__(self,
                 BATCH_SIZE,
                 num_workers,
                 train_list_path,
                 test_list_path,
                 val_list_path,
                 img_stack=10,
                 mode='motion'):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.img_stack = img_stack
        self.train_list = self.read_list(train_list_path)
        self.test_list = self.read_list(test_list_path)
        self.val_list = self.read_list(val_list_path)

        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.val_list)

        self.train_list = self.train_list[:100]
        self.val_list = self.val_list[:100]
        self.test_list = self.test_list[:100]

        self.mode = mode

    def __call__(self, *args, **kwargs):
        return self.load(data_list=self.train_list), \
               self.load(data_list=self.val_list, isval=True), \
               self.load(data_list=self.test_list, istest=True)

    def read_list(self, flowlist_path):
        with open(flowlist_path, 'r') as f:
            x = json.loads(f.read())
        return x

    def load(self, data_list, istest=False, isval=False):
        dataset = egocentric_dataset(
            data_list_dict=data_list,
            img_stack=self.img_stack,
            transform=transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]),
            resizecol=224,
            resizerow=224,

            istest=istest,
            mode=self.mode,
        )
        if istest:
            print('==> Testing data :', len(dataset))
        elif isval:
            print('==> Validation data :', len(dataset))
        else:
            print('==> Training data :', len(dataset))

        return DataLoader(
            dataset=dataset,
            batch_size={True: 1, False: self.BATCH_SIZE}.get(istest or isval),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    data_loader = EgoCentricDataLoader(
        BATCH_SIZE=10,
        num_workers=1,
        img_stack=10,
        train_list_path='../Egok_list/merged_train_list.txt',
        test_list_path='../Egok_list/merged_test_list.txt',
        val_list_path='../Egok_list/merged_val_list.txt',
        mode='spatial'
    )
    flow_train_loader, flow_val_loader, flow_test_loader, = data_loader()

    for f, fl in flow_train_loader:
        break

    for ftest, ftestl in flow_test_loader:
        break

    for fval, fvall in flow_val_loader:
        break
