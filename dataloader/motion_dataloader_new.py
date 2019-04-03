from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

get_video_name = lambda x:x.split('/')[-2]

class motion_dataset(Dataset):  

    def __init__(self,
                 data_list_dict,
                 img_stack,
                 transform=None,
                 rows = 320,
                 cols = 640,
                 selective = False,
                 resizerow = 224,
                 resizecol = 224,
                 num_classes = 63,
                 istest = False
                 ):

        self.data = data_list_dict
        self.transform = transform
        self.img_stack = img_stack
        self.img_rows = rows
        self.img_cols = cols
        self.resizerow = resizerow
        self.resizecol = resizecol
        self.stack = self.selectivestackopf if selective else self.stackopf
        self.num_classes = num_classes
        self.istest = istest

    def stackopf(self,idx):
        flow = torch.FloatTensor(2 * self.img_stack, self.resizerow, self.resizecol)
        data = self.data[idx]
        label = {'class':data[0]['class'],
                 'label':data[0]['label'],
                 'superclass':data[0]['superclass'],
                 'vidname':data[0]['vidname']}

        for i,d in enumerate(data):
            U = self.transform(Image.fromarray(cv2.imread(d['u'],cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(d['v'],cv2.IMREAD_GRAYSCALE)))
            flow[2*i,:,:] = U
            flow[2*i+1,:,:] = V
        return flow, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.stackopf(idx)
        return self.stack(idx)

class Motion_DataLoader():
    def __init__(self,
                BATCH_SIZE, 
                num_workers, 
                img_stack, 
                train_list_path,
                test_list_path,
                selective = False):
        
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.img_stack = img_stack
        self.train_list = self.read_list(train_list_path)
        self.test_list = self.read_list(test_list_path)
        self.selective = selective
    
    def __call__(self, *args, **kwargs):
        return self.load(self.train_list), self.load(self.test_list,True)

    def read_list(self,flowlist_path):
        with open(flowlist_path,'r') as f:
            x = json.loads(f.read())
        return x

    def load(self,data_list, istest = False):
        dataset = motion_dataset(
                                    data_list_dict=data_list,
                                    img_stack=self.img_stack,
                                    transform = transforms.Compose([
                                                    transforms.Resize([224, 224]),
                                                    transforms.ToTensor(),
                                                    ]),
                                    resizecol=224,
                                    resizerow=224,
                                    selective=self.selective,
                                    istest = istest
                                )
        if istest:
            print('==> Testing data :', len(dataset))
        else:    
            print('==> Training data :',len(dataset),"Selective Mode Training",self.selective)

        return DataLoader(
                            dataset=dataset, 
                            batch_size={True:1,False:self.BATCH_SIZE}.get(istest),
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=True
                        )

if __name__ == '__main__':
    data_loader = Motion_DataLoader(
                                   BATCH_SIZE=10,
                                   num_workers=1,
                                   img_stack=10,
                                   train_list_path = '../Egok_list/flow_trainlist.txt',
                                   test_list_path = '../Egok_list/flow_testlist.txt',
                                   selective = False
                                )
    train_loader,test_loader = data_loader()

    #Test Script
    for data, label in train_loader:
        break