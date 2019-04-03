from itertools import islice
from pathlib import Path
from termcolor import colored
import os
import json


FLOW_STACK_SIZE = 10


def printe(x):
    print(colored(x,'red'))

def chunk(it, size = FLOW_STACK_SIZE):
    it = iter(it)
    li = list(iter(lambda: list(islice(iter(it), size)), []))
    if len(li) > 1 and len(li[-1]) < size:
        temp = li[-2] + li[-1]
        li[-1] = temp[-size:]
    return li

flowpath = "/home/keshav/DATA/finalEgok360/flows/u/"
impath = "/home/keshav/DATA/finalEgok360/images/"




p = Path(flowpath)
p = [*p.glob("**/*.jpg")]
dic = {}

labellist = {"Ping-Pong__Ping-Pong": 0, "Ping-Pong__Pickup_ball": 1, "Ping-Pong__Hit": 2, "Ping-Pong__Bounce_ball": 3, "Ping-Pong__Serve": 4,
             "Sitting__Sitting": 5, "Sitting__Follow_obj": 6, "Sitting__Turn_right": 7, "Sitting__At_computer": 8, "Sitting__Check_phone": 9,
             "Sitting__Turn_left": 10, "Sitting__Reach": 11, "Driving__Decelerate": 12, "Driving__Stop": 13, "Driving__Driving": 14, "Driving__Still": 15,
             "Driving__Turn_right": 16, "Driving__Turn_left": 17, "Driving__Accelerate": 18, "Playing_pool__Playing_pool": 19, "Playing_pool__Chalk_up": 20,
             "Playing_pool__Shooting": 21, "Playing_pool__Turn_right": 22, "Playing_pool__Check_phone": 23, "Playing_pool__Turn_left": 24,
             "Playing_pool__Reach": 25, "Stairs__Doorway": 26, "Stairs__Turn_right": 27, "Stairs__Up_stairs": 28, "Stairs__Turn_left": 29,
             "Stairs__Reach": 30, "Stairs__Down_stairs": 31, "Lunch__Eating": 32, "Lunch__Drinking": 33, "Lunch__Turn_right": 34,
             "Lunch__Turn_left": 35, "Lunch__Ordering": 36, "Playing_cards__Shuffle": 37, "Playing_cards__Take_card": 38,
             "Playing_cards__Playing_cards": 39, "Playing_cards__Put_card": 40, "Desk_work__Sit_down": 41, "Desk_work__Napping": 42,
             "Desk_work__Writing": 43, "Desk_work__Turn_right": 44, "Desk_work__Stand_up": 45, "Desk_work__Desk_work": 46,
             "Desk_work__Turn_left": 47, "Standing__Leaning": 48, "Standing__Standing": 49, "Office_talk__Turn_right": 50,
             "Office_talk__Office_talk": 51, "Office_talk__Check_phone": 52, "Office_talk__Turn_left": 53, "Office_talk__Reach": 54,
             "Running__Turn_around": 55, "Running__Looking_at": 56, "Running__Running": 57, "Walking__Breezeway": 58, "Walking__Crossing_street": 59,
             "Walking__Doorway": 60, "Walking__Hallway": 61, "Walking__Walking": 62}

for i in p:
    key = i.parent.name
    subclass = i.parent.parent.name
    superclass = i.parent.parent.parent.name
    c = superclass + '__'+subclass

    ui = str(i)
    vi = ui.replace('/flows/u/','/flows/v/')
    vi = vi.replace('u.jpg', 'v.jpg')
    assert os.path.exists(ui)
    try:
        assert os.path.exists(vi)
    except:
        print(ui)
        print(vi)
        # exit(0)

    val = {'u':ui,'v':vi,'class':c,'label':labellist.get(c),'vidname':key,'superclass':superclass}

    if key not in dic.keys():
        dic.update({key:[val]})
    else:
        item = dic.get(key)
        item.append(val)
        dic[key] = item

S = lambda x:int(x.split('_')[-2])

Train = []
Test = []

printe(len(dic))

import random

trainlist = list(dic.keys())
testlist = random.sample(trainlist,int(len(dic) * 0.05))

for i in testlist:
    trainlist.remove(i)


printe(len(trainlist))
printe(len(testlist))


import random


for k in trainlist:
    dic[k] = sorted(dic[k],key=lambda x:S(x['u']))
    temp = []
    for c in chunk(dic[k]):
        Train.append(c)

for k in testlist:
    dic[k] = sorted(dic[k],key=lambda x:S(x['u']))
    temp = []
    for c in chunk(dic[k]):
        Test.append(c)

T = []


for k in dic.keys():
    dic[k] = sorted(dic[k],key=lambda x:S(x['u']))
    temp = []
    for c in chunk(dic[k]):
        T.append(c)


printe(len(Test))
printe(len(Train))
printe(len(T))
assert len(T) == len(Test) + len(Train)



def save(x, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(x))

save(Train,'../Egok_list/flow_trainlist.txt')
save(Test,'../Egok_list/flow_testlist.txt')


