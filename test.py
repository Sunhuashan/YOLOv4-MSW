# coding=utf-8
import os
import shutil

#目标文件夹，此处为相对路径，也可以改为绝对路径
determination = 'E:/big_paper/object_detection/dataset/changmiao/photo'
# if not os.path.exists(determination):
#     os.makedirs(determination)
#
# #源文件夹路径
# path = 'E:/big_paper/object_detection/dataset/畅淼-shipdata/BoatData/海事以图搜图数据库/船舶库'
# folders = os.listdir(path)
# count=1
# for folder in folders:
#     dir = path + '/' + str(folder)
#     files = os.listdir(dir)
#     for file in files:
#         source = dir + '/' + str(file)
#         if source.endswith(".jpg"):
#             deter = determination + '/' + str(count)+".jpg"
#             count=count+1
#             print(deter)
#             shutil.copyfile(source, deter)
#
folders=os.listdir(determination)
count=1
for file in  folders:
    os.rename(determination+"/"+file,determination+'/'+str(count)+'.jpg')
    count+=1