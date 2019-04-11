import numpy as np
import pickle
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *
#old
# import dataloader.dataloader as dataloader
#New
import dataloader.dataloader as dataloader
import numpy as np
import tabulate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 motion stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# parser.add_argument('--selective',dest='selective', action='store_true')
# parser.add_argument('--no-selective', dest='selective', action='store_false')
parser.add_argument('--img-stack', dest = 'img_stack', default=10, type = int)
# parser.set_defaults(selective = False)



def main():
    global arg
    arg = parser.parse_args()

    #Prepare DataLoader
    data_loader = dataloader.EgoCentricDataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        img_stack = arg.img_stack,
                        # train_list_path = './Egok_list/flow_trainlist.txt',
                        # test_list_path='./Egok_list/flow_testlist.txt',
                        # val_list_path='./Egok_list/flow_vallist.txt',
                        train_list_path='./Egok_list/merged_train_list.txt',
                        test_list_path='./Egok_list/merged_test_list.txt',
                        val_list_path='./Egok_list/merged_val_list.txt',
                        mode = 'motion'
                        )

    train_loader , val_loader, test_loader = data_loader()
    #Model 
    model = Motion_CNN(
                        # Data Loader
                        train_loader = train_loader,
                        val_loader = val_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 20,
                        test_loader=test_loader
                        )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    # model.to(device)

    #Training
    model.run()

class Motion_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, val_loader, channel,test_loader):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.best_prec1=0
        self.channel=channel
        self.test_loader=test_loader
        self.val_loader = val_loader

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=self.channel).cuda()
        #print self.model
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)


    def extractProbability(self, batch, label_dict):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        reference = self.test_loader
        progress = tqdm(reference)
        with torch.no_grad():
            label = label_dict['label'].cuda()
            data_var = Variable(batch).cuda()
            output = self.model(data_var)
        return output


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
            self.epoch=0
            prec1, val_loss = self.validate_1epoch()
            return
    
    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            acc, val_loss = self.validate_1epoch()
            is_best = acc > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = acc
                with open('record/motion/best_motion_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                    # f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/motion/checkpoint.pth.tar','record/motion/model_best.pth.tar')

        self.validate_1epoch(test=True)

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data,label_dict) in enumerate(progress):

            # measure data loading time
            data_time.update(time.time() - end)
            label = label_dict['label'].cuda()
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()


            # compute output
            output = self.model(input_var)

            # print(output.data.shape)


            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3, prec5 = kaccuracy_ontraining(output, label_dict['label'])
            # _, prec1, prec5 = accuracy(output.data, label, topk=(1, 5))

            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1, data.size(0))
            top3.update(prec3, data.size(0))
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
                'Prec@3' :[np.round(top3.avg,4)],
                'Prec@5':[np.round(top5.avg,4)],
                'lr': [self.optimizer.param_groups[0]['lr']]
                }
        print_dict_to_table(info,drop=['Epoch'],transpose=False)
        record_info(info, 'record/motion/new_opf_train.csv','train')

    def validate_1epoch(self, test = False):
        if test:
            f = '_test_'
            print('==> DONE [test stage]'.format(self.epoch, self.nb_epochs))
            reference = self.test_loader
        else:
            f = ''
            print('==> Epoch:[{0}/{1}][val stage]'.format(self.epoch, self.nb_epochs))
            reference = self.val_loader

        batch_time = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        self.dic_super_class_label_preds = {}
        self.dic_sub_class_label_preds = {}
        match_predictions = []
        end = time.time()
        progress = tqdm(reference)

        val_loss = AverageMeter()

        print('Len of val/test loader', len(reference))

        true_super = []
        pred_super = []
        true_sub = []
        pred_sub = []

        with torch.no_grad():
            for i, (batch,label_dict) in enumerate(progress):
                #data = data.sub_(127.353346189).div_(14.971742063)
                label = label_dict['label'].cuda()

                data_var = Variable(batch).cuda()
                label_var = Variable(label).cuda()

                # compute output
                output = self.model(data_var)

                val_loss.update(self.criterion(output, label_var))

                v = label_dict['vidname'][0]
                c = label_dict['class'][0]
                s = label_dict['superclass'][0]


                match1 = topk(output, label_dict['label'], 1)
                match3 = topk(output, label_dict['label'], 3)
                match5 = topk(output, label_dict['label'], 5)

                predicted_label = torch.max(output, 1)[1].item()
                # Predicted superclass s_, and subclass c_
                c_ = reverse_label_list.get(predicted_label)
                s_ = c_.split('/')[0]



                true_super.append(s)
                true_sub.append(c)

                pred_super.append(s_)
                pred_sub.append(c_)

                match = (match1, match3, match5)

                match_predictions.append(match)

                self.dic_video_level_preds = append_update_dic(v, self.dic_video_level_preds, match)
                self.dic_super_class_label_preds = append_update_dic(s, self.dic_super_class_label_preds, match)
                self.dic_sub_class_label_preds = append_update_dic(c, self.dic_sub_class_label_preds, match)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        acc1, acc2, acc3 = get_video_accuracy(match_predictions)
        #Sub Class Level Accuracy
        sub_acc1, sub_acc2, sub_acc3 = get_mean_average_accuracy(self.dic_sub_class_label_preds)
        #Super Class Level Accuracy
        super_acc1, super_acc2, super_acc3 = get_mean_average_accuracy(self.dic_super_class_label_preds)

        #Video Label Accuracy
        video_acc1, video_acc2, video_acc3 = get_mean_average_accuracy(self.dic_video_level_preds)

        info = {'Epoch':[self.epoch] * 3,
                'Batch Time':[np.round(batch_time.avg,3)]*3,
                'Top K':[1, 2, 3],
                'Accuracy':[acc1, acc2, acc3],
                'Aggregated@Accuracy@SubClass': [sub_acc1, sub_acc2, sub_acc3],
                'Prec@Accuracy@SuperClass':[super_acc1,super_acc2,super_acc3],
                'Aggregated@Accuracy@Video': [video_acc1, video_acc2,video_acc3],
                'Average@Loss':[val_loss.avg]*3,
                }
        print("Batch Time : {}".format(np.round(batch_time.avg,3)))


        print_dict_to_table(info, drop = ['Epoch','Batch Time'])

        self.dic_video_level_preds['accuracy'] = [video_acc1, video_acc2, video_acc3]
        self.dic_super_class_label_preds['accuracy'] = [super_acc1, super_acc2, super_acc3]
        self.dic_sub_class_label_preds['accuracy'] = [sub_acc1, sub_acc2, sub_acc3]

        pickleme('record/motion/'+f+'latest_motion_dic_video_level_preds.pickle', self.dic_video_level_preds)
        pickleme('record/motion/'+f+'latest_motion_dic_super_class_label_preds.pickle', self.dic_super_class_label_preds)
        pickleme('record/motion/'+f+'latest_motion_dic_sub_class_label_preds.pickle', self.dic_sub_class_label_preds)

        record_info(info, 'record/motion/'+f+'new_opf_test.csv','test')

        from pandas_ml import ConfusionMatrix

        confusion_matrix_super = ConfusionMatrix(true_super, pred_super)
        df_conf_super = confusion_matrix_super.to_dataframe()
        df_conf_super.to_csv('./record/motion/super_confusion.csv', index=None)

        super_mAP, super_mAR = get_mAP_and_mAR(df_conf_super)

        confusion_matrix_sub = ConfusionMatrix(true_sub, pred_sub)
        df_conf_sub = confusion_matrix_sub.to_dataframe()
        df_conf_sub.to_csv('./record/motion/sub_confusion.csv', index=None)

        sub_mAP, sub_mAR = get_mAP_and_mAR(df_conf_sub)

        info = {'On': ['Sub Class', 'Super Class'],
                'Mean Average Precision': [sub_mAP, super_mAP],
                'Mean Average Recall': [sub_mAR, super_mAR]
                }
        print_dict_to_table(info, drop=[], transpose=False)

        return acc1, val_loss.avg

if __name__=='__main__':
    main()