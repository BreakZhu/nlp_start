# -*— encoding: utf-8 -*-
"""
先逐个读入这些txt文件内容，然后正则表达匹配出URL（新闻类别）和content（新闻内容）
，然后根据URL将content存入不同文件夹/文件中
<url>(.*?)</url> # 匹配URL
<content>(.*?)</content>  # 匹配content
"""
import os
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')


"""生成原始语料文件夹下文件列表"""

data_dir = 'E:\\sougo\\souhunews\\'
data_simple = 'E:\\sougo\\souhusimple\\'


def listdir(data_path, list_names):
    for fl in os.listdir(data_path):
        file_path = os.path.join(data_path, fl)
        if os.path.isdir(file_path):
            listdir(file_path, list_names)
        else:
            list_names.append(file_path)


"""字符数小于这个数目的content将不被保存"""
content_words = 30
"""获取所有语料"""
list_name = []
listdir(data_dir, list_name)

"""对每个语料"""
for path in list_name:
    print(path)
    source_file = open(path, 'rb').read().decode("utf8")

    """
    正则匹配出url和content
    """
    patternURL = re.compile(r'<url>(.*?)</url>', re.S)
    patternContent = re.compile(r'<content>(.*?)</content>', re.S)

    classes = patternURL.findall(source_file)
    contents = patternContent.findall(source_file)

    """
    # 把所有内容小于30字符的文本全部过滤掉
    """
    for i in range(len(contents))[::-1]:
        if len(contents[i]) < content_words:
            contents.pop(i)
            classes.pop(i)

    """
    把URL进一步提取出来，只提取出一级url作为类别
    """
    for i in range(len(classes)):
        patternClass = re.compile(r'http://(.*?)/', re.S)
        classi = patternClass.findall(classes[i])
        classes[i] = classi[0]

    """
    按照RUL作为类别保存到samples文件夹中
    """
    for i in range(len(classes)):
        file_name = data_simple + classes[i].split('.')[0] + '.txt'
        f = open(file_name, 'a+')
        f.writelines(contents[i]+"\n")
