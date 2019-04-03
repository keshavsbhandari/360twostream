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
    def __init__(self, data_list_dict, img_stack, transform=None,
                 rows = 320, cols = 640, selective = False,
                 resizerow = 224, resizecol = 224, num_classes = 63,
                 istest = False):
        #Generate a 16 Frame clip
        self.data = data_list_dict
        self.transform = transform
        self.img_stack = img_stack
        self.img_rows = rows
        self.img_cols = cols
        self.resizerow = resizerow
        self.resizecol = resizecol
        self.stack = self.selectivestacopf if selective else self.stackopf
        self.num_classes = num_classes
        self.istest = istest



    def get_iterable_index(self,idx):
        last = []
        l = None
        for i, id in enumerate(range(idx, idx + self.img_stack)):
            C = self.data[id]['class']
            c = get_video_name(self.data[id]['u'])
            if i == 0:
                l = c
                last.append(id)
            else:
                if l == c:
                    last.append(id)
                else:
                    break
        first = [last[0]]*(self.img_stack-len(last))
        first = [i - (j + 1) for j, i in enumerate(first)][::-1]
        indx_search = first + last

        for ii in indx_search:
            if l != get_video_name(self.data[ii]['u']):
                raise Exception("Not Enough Frames to Stack with stacksize",self.img_stack)

        # assert len(indx_search) != self.img_stack, "Not Enough Videos to stack"

        if len(indx_search) < self.img_stack:
            indx_search = indx_search * self.img_stack
            indx_search = indx_search[:self.img_stack]
        return C, indx_search


    def selectivestacopf(self,idx):
        flow = torch.FloatTensor(6 * self.img_stack, self.resizerow, self.resizecol)
        label, indx_search = self.get_iterable_index(idx)
        for i, id in enumerate(indx_search):
            u = self.data[id]['u']
            v = self.data[id]['v']

            lr = self.img_cols //2
            ml = self.img_cols // 4
            mr = ml + (self.img_cols//2)

            t = lambda x: self.transform(x)

            f = lambda x: t(Image.fromarray(x))

            r = lambda x :f(x[:, :lr])
            m = lambda x: f(x[:, ml:mr])
            l = lambda x: f(x[:, lr:])



            u_all = cv2.imread(u, cv2.IMREAD_GRAYSCALE)
            v_all = cv2.imread(v, cv2.IMREAD_GRAYSCALE)


            flow[6 * i + 0, :, :] = l(u_all)
            flow[6 * i + 1, :, :] = l(v_all)
            flow[6 * i + 2, :, :] = m(u_all)
            flow[6 * i + 3, :, :] = m(v_all)
            flow[6 * i + 4, :, :] = r(u_all)
            flow[6 * i + 5, :, :] = r(v_all)



        # label_tensor = torch.eye(self.num_classes)[label]

        return flow, label

    def stackopf(self,idx):
        flow = torch.FloatTensor(2 * self.img_stack, self.resizerow, self.resizecol)

        label, indx_search = self.get_iterable_index(idx)
        for i,id in enumerate(indx_search):
            v_name = get_video_name(self.data[id]['u'])
            if i == 0:
                vidname = v_name

            assert vidname == v_name, "Not Enough Image to stack, reduce stacksize"

            u =  self.data[id]['u']
            v =  self.data[id]['v']

            U = self.transform(Image.fromarray(cv2.imread(u,cv2.IMREAD_GRAYSCALE)))
            V = self.transform(Image.fromarray(cv2.imread(v,cv2.IMREAD_GRAYSCALE)))

            flow[2*i,:,:] = U
            flow[2*i+1,:,:] = V

        if self.istest:
            sample = (flow, label, vidname)
        else:
            sample = (flow, label)

        return sample

    def __len__(self):
        return len(self.data) - self.img_stack

    def __getitem__(self, idx):
        # return self.stackopf(idx)
        return self.stack(idx)

class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, img_stack, path, flowlist_path, selective = False):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.img_stack = img_stack
        self.data_list =self.read_list(flowlist_path)
        self.selective = selective

        # shuffle(self.data_list)
        # self.train_list = self.data_list[:404]
        # self.test_list = self.data_list[454:]
        # self.val_list = self.data_list[404:454]

        self.train_list = self.test_list = self.val_list = self.data_list

    def __call__(self, *args, **kwargs):
        train_loader = self.train()
        val_loader = self.val()
        test_loader = self.test()
        return train_loader, val_loader, test_loader

    def read_list(self,flowlist_path):
        with open(flowlist_path,'r') as f:
            x = json.loads(f.read())
        return x

    def train(self):
        training_set = motion_dataset(data_list_dict=self.train_list, img_stack=self.img_stack,
                                      transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            ]),
              resizecol=224,
              resizerow=224,
              selective=self.selective)
        print('==> Training data :',len(training_set),"Selective Mode Training",self.selective)

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )
        return train_loader

    def val(self):
        validation_set = motion_dataset(data_list_dict= self.val_list, img_stack=self.img_stack,
                                        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            ]),
            resizecol=224,
            resizerow=224,
            selective=self.selective)
        print('==> Validation data :',len(validation_set))
        # print(validation_set[1])

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

    def test(self):
        test_set = motion_dataset(data_list_dict=self.test_list, img_stack=self.img_stack,
                                  transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            ]),
          resizecol=224,
          resizerow=224,
          selective=self.selective,
        istest = True)

        print('==> Training data :',len(test_set))

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
            )
        return test_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=10,
                                   num_workers=1,
                                   path='./../data/flows/',
                                   flowlist_path='./Egok_list/flowlist.txt',
                                   img_stack=5,
                                        )
    train_loader,val_loader,test_loader = data_loader()



