from itertools import islice
from pathlib import Path
from termcolor import colored
import os
import json
from glob import glob
from numba import jit

home = True

labellist = {'Desk_work/Desk_work': 0,
             'Desk_work/Napping': 1,
             'Desk_work/Sit_down': 2,
             'Desk_work/Stand_up': 3,
             'Desk_work/Turn_left': 4,
             'Desk_work/Turn_right': 5,
             'Desk_work/Writing': 6,
             'Driving/Accelerate': 7,
             'Driving/Decelerate': 8,
             'Driving/Driving': 9,
             'Driving/Still': 10,
             'Driving/Stop': 11,
             'Driving/Turn_left': 12,
             'Driving/Turn_right': 13,
             'Lunch/Drinking': 14,
             'Lunch/Eating': 15,
             'Lunch/Ordering': 16,
             'Lunch/Turn_left': 17,
             'Lunch/Turn_right': 18,
             'Office_talk/Check_phone': 19,
             'Office_talk/Office_talk': 20,
             'Office_talk/Reach': 21,
             'Office_talk/Turn_left': 22,
             'Office_talk/Turn_right': 23,
             'Ping-Pong/Bounce_ball': 24,
             'Ping-Pong/Hit': 25,
             'Ping-Pong/Pickup_ball': 26,
             'Ping-Pong/Ping-Pong': 27,
             'Ping-Pong/Serve': 28,
             'Playing_cards/Playing_cards': 29,
             'Playing_cards/Put_card': 30,
             'Playing_cards/Shuffle': 31,
             'Playing_cards/Take_card': 32,
             'Playing_pool/Chalk_up': 33,
             'Playing_pool/Check_phone': 34,
             'Playing_pool/Playing_pool': 35,
             'Playing_pool/Reach': 36,
             'Playing_pool/Shooting': 37,
             'Playing_pool/Turn_left': 38,
             'Playing_pool/Turn_right': 39,
             'Running/Looking_at': 40,
             'Running/Running': 41,
             'Running/Turn_around': 42,
             'Sitting/At_computer': 43,
             'Sitting/Check_phone': 44,
             'Sitting/Follow_obj': 45,
             'Sitting/Reach': 46,
             'Sitting/Sitting': 47,
             'Sitting/Turn_left': 48,
             'Sitting/Turn_right': 49,
             'Stairs/Doorway': 50,
             'Stairs/Down_stairs': 51,
             'Stairs/Reach': 52,
             'Stairs/Turn_left': 53,
             'Stairs/Turn_right': 54,
             'Stairs/Up_stairs': 55,
             'Standing/Leaning': 56,
             'Standing/Standing': 57,
             'Walking/Breezeway': 58,
             'Walking/Crossing_street': 59,
             'Walking/Doorway': 60,
             'Walking/Hallway': 61,
             'Walking/Walking': 62}

parent = {'Desk_work': 0,
          'Driving': 1,
          'Lunch': 2,
          'Office_talk': 3,
          'Ping-Pong': 4,
          'Playing_cards': 5,
          'Playing_pool': 6,
          'Running': 7,
          'Sitting': 8,
          'Stairs': 9,
          'Standing': 10,
          'Walking': 11}


def printe(x):
    print(colored(x, 'red'))


FLOW_STACK_SIZE = 10
videos = []


def chunk(it, size=FLOW_STACK_SIZE):
    it = iter(it)
    li = list(iter(lambda: list(islice(iter(it), size)), []))
    if len(li) > 1 and len(li[-1]) < size:
        temp = li[-2] + li[-1]
        li[-1] = temp[-size:]
    return li

if home:
    flowpath = "/home/keshav/DATA/finalEgok360/flows/u/"
    impath = "/home/keshav/DATA/finalEgok360/images/"
    superclass = "/home/keshav/DATA/finalEgok360/images/*/*"
else:
    flowpath = "/data/keshav/360/finalEgok360/flows/u/"
    impath = "/data/keshav/360/finalEgok360/images/"
    superclass = "/data/keshav/360/finalEgok360/images/*/*"

j = glob(superclass)

train = []
test = []
val = []
count = []


def getCount(x):
    if x < 40:
        return 1
    else:
        return int(x * 0.05)


for ji in j:
    allvid = glob(ji + '/*')
    n = getCount(len(allvid))

    data = sorted(allvid, key=lambda x: int(x.split('/')[-1]))

    train.extend(data[2 * n:])
    test.extend(data[:n])
    val.extend(data[n:2 * n])

print(len(train))
print(len(test))
print(len(val))

print("here")

# @jit
def getData(X):
    T = []
    for i in X:
        u = i.replace('/images/', '/flows/u/')
        v = i.replace('/images/', '/flows/v/')

        allimg = sorted(glob(i + '/*.jpg'), key=lambda x: int(Path(x).stem))[:-1]
        allu = sorted(glob(u + '/*.jpg'), key=lambda x: int(Path(x).stem))
        allv = sorted(glob(v + '/*.jpg'), key=lambda x: int(Path(x).stem))

        for imi,imu,imv in zip(chunk(allimg), chunk(allu), chunk(allv)):
            unitdata = {}



            unitdata['images'] = imi
            unitdata['flowu'] = imu
            unitdata['flowv'] = imv


            pth = Path(imu[0])

            p0 = pth.parent.name#video
            p1 = pth.parent.parent.name#child
            p2 = pth.parent.parent.parent.name#parent

            vidname = p0
            cls = p2 + '/' + p1
            scls = p2
            print(vidname, cls, scls)
            label = labellist[cls]



            assert label is not None

            unitdata['video'] = vidname
            unitdata['class'] = cls
            unitdata['parentclass'] = scls
            unitdata['label'] = label

            T.append(unitdata)

    return T
# keys = ['images','flowu','flowv','video','class','parentclass','label']

    # chunk1 = chunk([*map(lambda x:int(Path(x).stem),allimg)])
print("TEST DATA STARTED")
TestData = getData(test)
print("Train DATA STARTED")
TrainData = getData(train)
print("vAL DATA STARTED")
ValData = getData(val)




def save(x, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(x))

save(TrainData,'../Egok_list/merged_train_list.txt')
save(TestData,'../Egok_list/merged_test_list.txt')
save(ValData,'../Egok_list/merged_val_list.txt')

