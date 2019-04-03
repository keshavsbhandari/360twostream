from pathlib import Path
import os
from os.path import join
from glob import glob
from itertools import chain
import json

images = '../data/images/'
flowu = '../data/flows/u/'
flowv = '../data/flows/v/'

current_path = os.path.dirname(os.path.realpath(__file__))

f = lambda x:[*chain(*x)]

def s(x):
    getkey = lambda a:int(a.split('_')[-1].split('.jpg')[0].split('X')[-1])
    for i in range(len(x)):
        x[i] = sorted(x[i],key=lambda t:getkey(t))
    return f(x)

getdir = lambda x,root:os.listdir(
    join(x,root,''))

mapdir = lambda X,root:s([*map(
    lambda x:glob(join(X,root,x,'*.jpg')),getdir(X,root))])

classes = []
for topclass in os.listdir(images):
    for subclass in os.listdir(join(images,topclass)):
        subclass = topclass+'__'+subclass
        classes.append(subclass)
classes = dict(zip(classes,range(len(classes))))


def save(x, fname):
    with open(fname, 'w') as f:
        import json
        f.write(json.dumps(x))


def load(fname):
    with open(fname, 'r') as f:
        import json
        x = json.loads(f.read())
    return x


imageList = []
flowList = []

for item in classes:
    c = classes.get(item)

    ipath = item.replace('__', '/')

    imgs = [{'x': i, 'class': c} for i in mapdir(images, ipath)]

    us = mapdir(flowu, ipath)

    vs = mapdir(flowv, ipath)

    flow = [{'u': i, 'v': j, 'class': c} for i, j in zip(us, vs)]

    imageList.extend(imgs)
    flowList.extend(flow)

save(flowList, 'flowlist.txt')
save(imageList, 'imagelist.txt')
