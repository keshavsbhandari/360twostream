from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

get_video_name = lambda x:x.split('/')[-2]

class egocentric_dataset(Dataset):

    def __init__(self,
                 data_list_dict,
                 img_stack = 10,
                 transform=None,
                 rows = 320,
                 cols = 640,
                 resizerow = 224,
                 resizecol = 224,
                 num_classes = 63,
                 istest = False,
                 mode = 'motion'
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

    def stackopf(self,data):
        flow = torch.FloatTensor(2 * self.img_stack, self.resizerow, self.resizecol)

        label = {'class': data[0]['class'],
                 'label': data[0]['label'],
                 'superclass': data[0]['superclass'],
                 'vidname': data[0]['vidname']}

        for i,d in enumerate(data):
            U = self.transform(Image.fromarray(cv2.imread(d['u'],cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(d['v'],cv2.IMREAD_GRAYSCALE)))
            flow[2*i,:,:] = U
            flow[2*i+1,:,:] = V
        return flow, label

    def __len__(self):
        return len(self.data)

    def getimages(self,data):

        label = {'class': data['class'],
                 'label': data['label'],
                 'superclass': data['superclass'],
                 'vidname': data['vidname']}

        image = self.transform(Image.fromarray(cv2.imread(data['images'])))

        return image,label

    def __getitem__(self, idx):
        data = self.data[idx]
        return {'motion':self.stackopf,'spatial':self.getimages}.get(self.mode)(data)

class EgoCentricDataLoader():
    def __init__(self,
                BATCH_SIZE, 
                num_workers,
                train_list_path,
                test_list_path,
                val_list_path,
                img_stack=10,
                mode = 'motion'):
        
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.img_stack = img_stack
        self.train_list = self.read_list(train_list_path)
        self.test_list = self.read_list(test_list_path)
        self.val_list = self.read_list(val_list_path)
        self.mode = mode
    
    def __call__(self, *args, **kwargs):
        return self.load(data_list=self.train_list), \
               self.load(data_list=self.val_list, isval=True),\
               self.load(data_list=self.test_list,istest=True)


    def read_list(self,flowlist_path):
        with open(flowlist_path,'r') as f:
            x = json.loads(f.read())
        return x

    def load(self,data_list, istest = False,isval = False):
        dataset = egocentric_dataset(
                                    data_list_dict=data_list,
                                    img_stack=self.img_stack,
                                    transform = transforms.Compose([
                                                    transforms.Resize([224, 224]),
                                                    transforms.ToTensor(),
                                                    ]),
                                    resizecol=224,
                                    resizerow=224,

                                    istest = istest,
                                    mode = self.mode,
                                )
        if istest:
            print('==> Testing data :', len(dataset))
        elif isval:
            print('==> Validation data :',len(dataset))
        else:
            print('==> Training data :', len(dataset))


        return DataLoader(
                            dataset=dataset, 
                            batch_size={True:1,False:self.BATCH_SIZE}.get(istest or isval),
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=True
                        )

if __name__ == '__main__':
    data_loader = EgoCentricDataLoader(
                                   BATCH_SIZE=10,
                                   num_workers=1,
                                   img_stack=10,
                                   train_list_path = '../Egok_list/flow_trainlist.txt',
                                   test_list_path = '../Egok_list/flow_testlist.txt',
                                   val_list_path = '../Egok_list/flow_vallist.txt',
                                   mode = 'motion'
                                )
    flow_train_loader, flow_val_loader, flow_test_loader, = data_loader()
    for f, fl in flow_train_loader:
        break

    for ftest, ftestl in flow_test_loader:
        break

    for fval, fvall in flow_val_loader:
        break

    # exit(0)

    data_loader = EgoCentricDataLoader(
                                   BATCH_SIZE=10,
                                   num_workers=1,
                                   train_list_path = '../Egok_list/images_trainlist.txt',
                                   test_list_path = '../Egok_list/images_testlist.txt',
                                   val_list_path='../Egok_list/images_vallist.txt',
                                   mode = 'spatial'
                                )
    image_train_loader,image_val_loader,image_test_loader = data_loader()




    #Test Script
    for i, il in image_train_loader:
        break

    for itest, itestl in image_test_loader:
        break

    for ival, ivall in image_test_loader:
        break





