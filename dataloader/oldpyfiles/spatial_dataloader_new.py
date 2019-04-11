from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import json
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

get_video_name = lambda x:x.split('/')[-2]


class spatial_dataset(Dataset):  
    def __init__(self, data_list_dict, rows = 320,
                 cols = 640,resizerow = 224,
                 resizecol = 224,transform=None,num_classes = 63):
        self.data = data_list_dict
        self.transform = transform
        self.img_rows = rows
        self.img_cols = cols
        self.resizerow = resizerow
        self.resizecol = resizecol
        self.num_classes = num_classes


    def __len__(self):
        return len(self.data)

    def __get_randNx(self, label, N):
        filteridx = [i for i,j in enumerate(self.data) if j['class']==label]
        return random.sample(filteridx, N)


    def __getitem__(self,idx):
        label = self.data[idx]['class']
        idxlist = self.__get_randNx(label,3)
        data = {}
        for i,idx_ in enumerate(idxlist):
            impath = self.data[idx_]['x']
            data['img'+str(i)] = self.transform(Image.fromarray(cv2.imread(impath)))
        return data, label



class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, imagelist_path):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_list=self.read_list(imagelist_path)
        self.train_list = self.data_list[:404]
        self.test_list = self.data_list[454:]
        self.val_list = self.data_list[404:454]

    def __call__(self, *args, **kwargs):
        train_loader = self.get_data(self.train_list,"Training")
        val_loader = self.get_data(self.val_list,"Validation")
        test_loader = self.get_data(self.test_list,"Test")
        return train_loader, val_loader, test_loader

    def read_list(self, flowlist_path):
        with open(flowlist_path, 'r') as f:
            x = json.loads(f.read())
        return x

    def get_data(self,data_list,mode = ''):
        data_set = spatial_dataset(data_list_dict=data_list,
                                       transform=transforms.Compose([
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ]))
        print('==> '+mode+' data :', len(data_set), 'frames')
        print(data_set[1][0]['img1'].size())

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)

        return data_loader




if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1,
                                    num_workers=1,
                                    path='.data/images/',
                                    imagelist_path='./Egok_list/imagelist.txt'
                                    )
    train_loader,val_loader,test_video = dataloader()
