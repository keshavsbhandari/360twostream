import pickle,os
import pandas as pd
import shutil
import torch
from itertools import chain
import tabulate





def topk(output, target , k = 1):
    _, l = torch.topk(output,k)
    l = [*chain.from_iterable(l.tolist())]
    belongs_to = (target.item() in l) * 1
    return belongs_to



def topK(output, target , k = 1):
    _, l = torch.topk(output,k)
    l = l.tolist()
    belongs_to = (target.item() in l) * 1
    return belongs_to


def kaccuracy_ontraining(output, target):
    percent = lambda x:(sum(x)/len(x)) * 100
    M1 = []
    M3 = []
    M5 = []
    for o,t in zip(output, target):
        # print(o,t)
        M1.append(topK(o, t, 1))
        M3.append(topK(o, t, 3))
        M5.append(topK(o, t, 5))
    # return 1,2,3
    return percent(M1), percent(M3), percent(M5)



def print_dict_to_table(dic, drop = None, transpose = True):
    for d in drop:
        dic.pop(d)
    if transpose:
        dic = get_dic_transpose(dic,on="Top K")
    printinfo = tabulate.tabulate(dic, headers=dic.keys(), tablefmt='github',floatfmt=".4f")
    to = len(printinfo.split('\n')[0])

    print("="*to)
    print(printinfo)
    print("=" * to)


def get_dic_transpose(dic, on):
    new_keys = ['header']
    new_keys += [*map(lambda x:on+'_'+str(x),dic.pop(on))]
    old_keys = dic.keys()
    old_values = dic.values()

    new_dic = {}
    new_dic[new_keys[0]] = list(old_keys)
    new_dic[new_keys[1]] = [*map(lambda x:x[0],old_values)]
    new_dic[new_keys[2]] = [*map(lambda x: x[1], old_values)]
    new_dic[new_keys[3]] = [*map(lambda x: x[2], old_values)]

    return new_dic

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

def get_video_accuracy(value):
    k1, k3, k5 = torch.tensor(value,dtype = torch.float).mean(0).tolist()
    return float(k1 * 100) ,float(k3 * 100) ,float(k5 * 100)

def get_mean_average_accuracy(dic):
    T = lambda x: torch.tensor(x, dtype=torch.float).mean(0)
    value = [*chain(dic.values())]
    value = [*map(T, value)]
    k1, k3, k5 = torch.stack(value).mean(0).tolist()
    return k1 * 100 ,k3 * 100 ,k5 * 100

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


def get_mAP_and_mAR(df):
    values = df.values
    pred_sum = df.sum(axis = 1).values
    real_sum = df.sum(axis = 0).values
    precision = values / (pred_sum.reshape(-1,1) + 1e-9)
    recall = values / (real_sum + 1e-9)

    avg_recall = recall.diagonal().mean() * 100
    avg_precision = precision.diagonal().mean() * 100
    return avg_precision, avg_recall








reverse_label_list = {
                        0: 'Ping-Pong__Ping-Pong',
                        1: 'Ping-Pong__Pickup_ball',
                        2: 'Ping-Pong__Hit',
                        3: 'Ping-Pong__Bounce_ball',
                        4: 'Ping-Pong__Serve',
                        5: 'Sitting__Sitting',
                        6: 'Sitting__Follow_obj',
                        7: 'Sitting__Turn_right',
                        8: 'Sitting__At_computer',
                        9: 'Sitting__Check_phone',
                        10: 'Sitting__Turn_left',
                        11: 'Sitting__Reach',
                        12: 'Driving__Decelerate',
                        13: 'Driving__Stop',
                        14: 'Driving__Driving',
                        15: 'Driving__Still',
                        16: 'Driving__Turn_right',
                        17: 'Driving__Turn_left',
                        18: 'Driving__Accelerate',
                        19: 'Playing_pool__Playing_pool',
                        20: 'Playing_pool__Chalk_up',
                        21: 'Playing_pool__Shooting',
                        22: 'Playing_pool__Turn_right',
                        23: 'Playing_pool__Check_phone',
                        24: 'Playing_pool__Turn_left',
                        25: 'Playing_pool__Reach',
                        26: 'Stairs__Doorway',
                        27: 'Stairs__Turn_right',
                        28: 'Stairs__Up_stairs',
                        29: 'Stairs__Turn_left',
                        30: 'Stairs__Reach',
                        31: 'Stairs__Down_stairs',
                        32: 'Lunch__Eating',
                        33: 'Lunch__Drinking',
                        34: 'Lunch__Turn_right',
                        35: 'Lunch__Turn_left',
                        36: 'Lunch__Ordering',
                        37: 'Playing_cards__Shuffle',
                        38: 'Playing_cards__Take_card',
                        39: 'Playing_cards__Playing_cards',
                        40: 'Playing_cards__Put_card',
                        41: 'Desk_work__Sit_down',
                        42: 'Desk_work__Napping',
                        43: 'Desk_work__Writing',
                        44: 'Desk_work__Turn_right',
                        45: 'Desk_work__Stand_up',
                        46: 'Desk_work__Desk_work',
                        47: 'Desk_work__Turn_left',
                        48: 'Standing__Leaning',
                        49: 'Standing__Standing',
                        50: 'Office_talk__Turn_right',
                        51: 'Office_talk__Office_talk',
                        52: 'Office_talk__Check_phone',
                        53: 'Office_talk__Turn_left',
                        54: 'Office_talk__Reach',
                        55: 'Running__Turn_around',
                        56: 'Running__Looking_at',
                        57: 'Running__Running',
                        58: 'Walking__Breezeway',
                        59: 'Walking__Crossing_street',
                        60: 'Walking__Doorway',
                        61: 'Walking__Hallway',
                        62: 'Walking__Walking'
                    }