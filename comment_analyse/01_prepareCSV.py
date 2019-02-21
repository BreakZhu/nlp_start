# -*- encoding: utf-8 -*-
import os
import numpy as np


# 生成一个map，<文件名,类型>
def makeFileMap(input_path):
    fileMap = {}
    cata1_set = set()
    # 获取文件列表
    fileList = os.listdir(input_path)
    # 生成一个map，<文件名,类型>
    for i in range(0, len(fileList)):
        f = fileList[i]
        path = os.path.join(input_path, f)
        # 只对文件夹下的文件操作
        if os.path.isfile(path):
            # 车险商品的名称
            catalog1 = f.split(".")[0].split("_")[0]
            num = f.split(".")[0].split("_")[1]
            if not catalog1 in cata1_set:
                cata1_set.add(catalog1)
            # 文件名称当作KEY
            key = catalog1
            if not key in fileMap:
                fileMap[key] = 0
            # key对应的value设定为文章类型
            fileMap[key] += 1
    print "input catalog number:", len(fileMap)
    for f in fileMap:
        print "{0}:{1}".format(f, fileMap[f])
    return fileMap


# 把文章移动到对应的目录下面
def moveFile(fmap, mainPath):
    for catlg in fmap:
        # 创建子目录
        subPath = mainPath + catlg + "/"
        if not os.path.exists(subPath):
            os.makedirs(subPath)
        # 拼接文件名
        for i in range(fmap[catlg]):
            fname = catlg + '_' + str(i) + '.csv'
            # 获得移动源目录
            oldPath = os.path.join(mainPath, fname)
            # 获得移动目标目录
            newPath = os.path.join(subPath, fname)
            # 移动文件
            os.rename(oldPath, newPath)
            print "move %s to % s" % (oldPath, newPath)


# 根据文件名所带的文章分类，把采集到的文章移动到对应的目录里面
def main():
    # 修改抓取的文本文件路径
    input_path = "./data/comment/"
    fileList = makeFileMap(input_path)
    moveFile(fileList, input_path)

if __name__ == '__main__':
    main()