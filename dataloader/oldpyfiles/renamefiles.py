# exit(0)
# from os.path import join
# from pathlib import Path
# from glob import glob
#
# ipath = "/home/keshav/DATA/finalEgok360/images/"
# vpath = "/home/keshav/DATA/finalEgok360/flows/v/"
# upath = "/home/keshav/DATA/finalEgok360/flows/u/"
#
# # impath = "/home/keshav/DATA/finalEgok360/images/"
#
# images = glob(ipath+'*/*/*')
# uflows = glob(upath+'*/*/*')
# vflows = glob(vpath+'*/*/*')
#
#
# fK = lambda x:int(x.as_posix().split('_')[-2])
# iK = lambda x:int(x.as_posix().split('_')[-1].split('.jpg')[0])
#
# flowsort = lambda x : sorted(x,key = fK)
# imgsort = lambda x :  sorted(x,key=iK)
#
# getlist = lambda x:[*Path(x).glob("*.jpg")]
#
# def rename(x):
#     for i,xi in enumerate(x):
#         xi = Path(xi)
#         newxi = xi.parent/'{}.jpg'.format(i)
#         xi.rename(newxi)
#
# # for i,u,v in zip(images, uflows, vflows):
# #
# #     ip = imgsort(getlist(i))
# #     iu = flowsort(getlist(u))
# #     iv = flowsort(getlist(v))
# #     rename(ip)
# #     rename(iu)
# #     rename(iv)
#
#
# sortfolder = lambda x:sorted(x, key = lambda X:int(X.split('_')[-1]))
#
# def renamefolder(x,y,z):
#     x = sortfolder(x)
#     y = sortfolder(y)
#     z = sortfolder(z)
#
#
#     for i,(ix,iy,iz) in enumerate(zip(x,y,z)):
#         ix,iy,iz = Path(ix),Path(iy),Path(iz)
#         assert ix.name == iy.name
#         assert ix.name == iz.name
#         assert iy.name == iz.name
#         new = lambda x:Path(x).parent/'{}'.format(str(i).zfill(4))
#         ixnew = new(ix)
#         iynew = new(iy)
#         iznew = new(iz)
#
#         assert ixnew.name == iynew.name
#         assert ixnew.name == iznew.name
#         assert iynew.name == iznew.name
#
#         ix.rename(ixnew)
#         iy.rename(iynew)
#         iz.rename(iznew)
#
#         # xi = Path(xi)
#         # newxi = xi.parent/'{}'.format(str(i).zfill(4))
#         # print(newxi)
#         # print(xi)
#         # if i==5:break
#         # # xi.rename(newxi)
#     print(len(x))
#     print(len(y))
#     print(len(z))
#
#     print(iynew.name)
#
#
# renamefolder(images,uflows,vflows)