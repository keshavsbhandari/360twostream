import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

# import torchvision.transforms as transforms
# import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *
from torchvision import transforms
from  torchvision import  models
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='./../data/images/',
                        imagelist_path='./Egok_list/imagelist.txt'
                        )
    
    train_loader, val_loader, test_loader = data_loader()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader
    )
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, val_loader, test_loader):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.val_loader=val_loader

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

        self.validate_1epoch(test=True)

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            # label = label.cuda(async=True)
            label = label.cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']),63).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            prec1, _, _ = accuracy(output.data, label, topk=(1,))
            prec5, _, _ = accuracy(output.data, label, topk=(1,5))
            # losses.update(loss.data[0], data.size(0))
            losses.update(loss.data.item(), data.size(0))
            # top1.update(prec1[0], data.size(0))
            top1.update(prec1, data.size(0))
            # top5.update(prec5[0], data.size(0))
            top5.update(prec5, data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Data Time':[np.round(data_time.avg,3)],
                'Loss':[np.round(losses.avg,5)],
                'Prec@1':[np.round(top1.avg,4)],
                'Prec@5':[np.round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self, test = False):
        if test:
            print('==> Epoch:[{0}/{1}][test stage]'.format(self.epoch, self.nb_epochs))
            reference = self.test_loader
        else:
            print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
            reference = self.val_loader
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(reference)
        total = 0
        correct_preds = 0
        print('Len of val/test loader',len(reference))
        with torch.no_grad():
            for j, (data_dict,label) in enumerate(progress):
                total += len(data_dict)*data_dict['img0'].size(0)
                for i in range(len(data_dict)):
                    data = data_dict['img'+str(i)]
                    label = label.cuda()
                    data_var = Variable(data).cuda()
                    label_var = Variable(label).cuda()
                    # compute output
                    output = self.model(data_var)

                    if i == 0 and j == 0:
                        top_label = label
                        top_output = output

                    # Compute Frame Level Accuracy
                    _, cp, _ = accuracy(output, label_var, None)


                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # total += N
                    correct_preds += cp
        acc = correct_preds / total
        top1, top5, loss = self.frame_level_accuracy(top_label, top_output)
        info = {'Epoch': [self.epoch],
                'Batch Time': [np.round(batch_time.avg, 3)],
                'Loss': [np.round(loss, 5)],
                'Prec@1': [np.round(top1, 3)],
                'Prec@5': [np.round(top5, 3)],
                'Accuracy': [np.round(acc, 3)],

                }
        if test:
            record_info(info, 'record/motion/opf_test.csv','test')
        else:
            record_info(info, 'record/motion/opf_val.csv', 'test')

        return top1, loss

    def frame_level_accuracy(self,label_var, output):
        loss = self.criterion(output,label_var)
        top1, _, _ = accuracy(output, label_var, (1,))
        top5, _, _ = accuracy(output, label_var, (1,5))
        return top1, top5, loss.data.cpu().numpy()







if __name__=='__main__':
    main()
