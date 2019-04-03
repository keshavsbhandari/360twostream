import pickle,os
import pandas as pd
import shutil
import torch
from itertools import chain

def accuracy(output, target, topk=None):
    _, o = torch.max(output, 1)
    t = target
    if topk:
        k = max(topk)
        t, o = t[:k].float(), o[:k].float()
    N = t.numel()
    correct_preds = t.eq(o).sum().item()
    return correct_preds * (100.0 / N),  correct_preds, N

def pickleme(fname,dic):
    with open(fname, 'wb') as f:
        pickle.dump(dic, f)

def append_update_dic(key,dic,val):

    if dic.get(key) is None:
        dic[key] = [val]
    else:
        dic[key].append(val)

    return dic


def get_mean_average_accuracy(dic):
    value = [*chain(dic.values())]
    value = [*map(lambda x:sum(x)/len(x),value)]
    return 100 * (sum(value)/len(value))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):
    if mode =='train':
        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))      
        print(result)
        df = pd.DataFrame(data = info)
    if mode =='test':
        df = pd.DataFrame(data = info)
    if not os.path.isfile(filename):
        df.to_csv(filename,index=None)
    else:
        df.to_csv(filename,mode = 'a', header= False, index=None)


