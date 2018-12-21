# -*-coding:utf-8 -*-
import os

folder = 'E:\\sougo\\SogouCS.reduced\\'  # 存储文本的目录
rootdir = 'E:\\sougo\\souhunews\\'
listDir = [dirs[0] for dirs in os.walk(folder)]  # 获取所有的子目录
for dataDir in listDir:
    files = [os.path.join(dataDir, i) for i in os.listdir(dataDir)]  # 获取绝对路径
    for words in files:
        pos, filename = os.path.split(words)
        newFile = file(os.path.join(rootdir, filename[:-4] + '_.txt'), 'w')  # 建立新文件
        try:  # 转码
            newFile.write(file(words, 'r').read().decode('gb2312').encode('utf-8'))
        except:
            newFile.write(file(words, 'r').read().decode('gbk', 'ignore').encode('utf-8'))
        newFile.close()
        # os.remove(words)  # 删除旧文件
