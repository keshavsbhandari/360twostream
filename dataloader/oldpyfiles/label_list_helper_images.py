from itertools import islice
from pathlib import Path
from termcolor import colored
import os
import json

IMAGE_STACK_SIZE = 3

def printe(x):
    print(colored(x,'red'))

def chunk(it, size = IMAGE_STACK_SIZE):
    it = iter(it)
    li = list(iter(lambda: list(islice(iter(it), size)), []))
    if len(li) > 1 and len(li[-1]) < size:
        temp = li[-2] + li[-1]
        li[-1] = temp[-size:]
    return li

impath = "/home/keshav/DATA/finalEgok360/images/"




p = Path(impath)
p = [*p.glob("**/*.jpg")]

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


parent = {
    "Ping-Pong":0,
    "Sitting":1,
    "Driving":2,
    "Playing_pool":3,
    "Stairs":4,
    "Lunch":5,
    "Playing_cards":6,
    "Desk_work":7,
    "Standing":8,
    "Office_talk":9,
    "Running":10,
    "Walking":11,
    }

data = []



for i in p:
    key = i.parent.name
    subclass = i.parent.parent.name
    superclass = i.parent.parent.parent.name
    c = superclass + '__'+subclass

    imi = str(i)
    assert os.path.exists(imi)

    assert labellist.get(c) is not None
    assert parent.get(superclass) is not None

    val = {'images':imi,'class':c,'label':labellist.get(c),'vidname':key,'superclass':superclass,'slabel':parent.get(superclass)}

    data.append(val)

import random

random.shuffle(data)


label_keys = list(labellist.keys())
label_value_init = [0] * len(label_keys)
count_dict_val = dict(zip(label_keys, label_value_init.copy()))
count_dict_test = dict(zip(label_keys, label_value_init.copy()))
dict_count_class_wise = dict(zip(label_keys, label_value_init.copy()))
max_limit = dict(zip(label_keys, label_value_init.copy()))
count_dict_train = dict(zip(label_keys, label_value_init.copy()))

for d in data:
    key = d['class']
    dict_count_class_wise[key]+= 1


def map_max_count(val):
    return int(val*0.05)

for m in max_limit:
    max_limit[m] = map_max_count(dict_count_class_wise[m])

print(max_limit.values())
print(min(list(max_limit.values())))
print(sum(list(max_limit.values())))


train_data = []
test_data = []
val_data = []


for d in data:
    k = d['class']
    limit = max_limit[k]
    if count_dict_val[k] < limit:
        val_data.append(d)
        count_dict_val[k] += 1
    elif count_dict_test[k] < limit:
        test_data.append(d)
        count_dict_test[k] += 1
    else:
        train_data.append(d)
        count_dict_train[k] += 1


print("VERIFYING")
print(len(train_data))
print(len(test_data))
print(len(val_data))
print(len(train_data) + len(test_data) + len(val_data))
print(len(data))

print(min(list(count_dict_train.values())))
print(min(list(count_dict_test.values())))
print(min(list(count_dict_val.values())))

def save(x, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(x))

save(train_data,'../Egok_list/images_trainlist.txt')
save(test_data,'../Egok_list/images_testlist.txt')
save(val_data,'../Egok_list/images_vallist.txt')