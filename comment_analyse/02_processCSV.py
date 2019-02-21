# -*- encoding: utf-8 -*-
import sys
import codecs
import csv
import MeCab
import os
import re
from collections import Counter, OrderedDict
import json

reload(sys)
sys.setdefaultencoding('utf-8')


# 读取停用词一览
def readStopWord(ssfile):
    ss = []
    fr = open(ssfile, "r")
    for line in fr.readlines():
        line = line.strip()
        if line != '':
            ss.append(line)
    fr.close()
    return ss


# 读取CSV文件
def readDataFile(fileName):
    with open(fileName, "r+") as csvfile:
        reader = csv.reader(csvfile)
        print "Read CSV:", fileName
        # 读取内容
        for line in reader:
            return line


# 获取词汇本体
def get_surfaces(node):
    words = []
    nouns = []
    while node:
        word = node.surface
        words.append(word)
        noun = node.feature.split(",")[0]
        nouns.append(noun)
        node = node.next
    return words, nouns


# 利用词性筛选词汇
def select_feature(data, attr):
    result = []
    pattern1 = re.compile(r'[^0-9]+')
    pattern2 = re.compile(ur'[^０１２３４５６７８９]+')
    pattern3 = re.compile(ur'[^\[\]!$%&\'\"\(\):\-\.,/;=<>]+')
    pattern4 = re.compile(ur'[^！　％、。）※「」（＞～』＜？－．♪【⇒∞★〇・⇔]+')
    # 需要根据词性做筛选
    if attr is not None:
        for (w, a) in data:
            if a in attr:
                tmp = w.decode('utf-8')
                matcher1 = re.match(pattern1, w)
                matcher2 = re.match(pattern2, tmp)
                matcher3 = re.match(pattern3, tmp)
                matcher4 = re.match(pattern4, tmp)
                if (matcher1 is not None) and (matcher2 is not None) and (matcher3 is not None) and (
                            matcher4 is not None):
                    result.append((w, a))
    else:
        # 记号以外的词全部输出
        for (w, a) in data:
            if a != u'記号':
                tmp = w.decode('utf-8')
                matcher1 = re.match(pattern1, w)
                matcher2 = re.match(pattern2, tmp)
                matcher3 = re.match(pattern3, tmp)
                matcher4 = re.match(pattern4, tmp)
                if (matcher1 is not None) and (matcher2 is not None) and (matcher3 is not None) and (
                            matcher4 is not None):
                    result.append((w, a))
    return result


# 把一个目录内的文件汇总以后写入一个文件
def writeOutput(fileName, data, mode):
    fw = open(fileName, mode)
    line = ""
    for w in data:
        line = line + w + "\t"
    fw.writelines(line + "\n")
    fw.close()


# 把筛选以后的结果写入单个文件
def writeSingleFile(fileName, data, mode):
    fw = open(fileName, mode)
    line = ""
    for item in data:
        w, a = item
        fw.writelines(w + "," + a + "\n")
    fw.close()


# from collections import Counter,OrderedDict
# Count结果，然后写入JSON文件
# 抽取计数前 top_n
def writeCountJSON(fileName, data, top_n=10, sort=False, itemFilter=False, filterSet=None):
    count = Counter(data)
    tmp = count.most_common(top_n)
    # 排序
    if sort:
        print " * dictionary sorted * "
        out = OrderedDict(sorted(tmp, key=lambda item: int(item[1]), reverse=False))
    else:
        out = dict(tmp)
    # 过滤元素
    if itemFilter:
        if filterSet is not None:
            for f in out.keys():
                if f in filterSet:
                    out.pop(f)
    with open(fileName, 'w') as json_file:
        json_file.write(json.dumps(out, sort_keys=False, indent=4, ensure_ascii=False))


# 写入JSON文件
def writeJSON(fileName,data):
    with open(fileName, 'w') as json_file:
        json_file.write(json.dumps(data,indent=4, ensure_ascii=False))


# 词性筛选 mode
# 筛选 形容詞 ：mode 指定 "A"
# 筛选 动词 ：mode 指定 "V"
# 例如：筛选 名词，形容词的场合 :mode = "NA"
def processFile(inPath, outPath, stopwords, mode):
    # 保存一个商品所有评论的分词
    wordFreq = []
    region = []
    age = []
    gender = []
    car = []
    fee = []
    ### 情感倾向
    emotion = []
    # 根据输入的目录获得商品名
    syouhin = inPath.split('/')[-2]
    # 词频统计结果
    outFileFQ = outPath + "freq_" + syouhin + ".json"
    # 地域统计结果
    outFileRegion = outPath + "Region_" + syouhin + ".json"
    # 年龄统计结果
    outFileAge = outPath + "Age_" + syouhin + ".json"
    # 性别统计结果
    outFileGender = outPath + "Gender_" + syouhin + ".json"
    # 车型统计结果
    outFileCar = outPath + "Car_" + syouhin + ".json"
    # 保险费统计结果
    outFileFee = outPath + "Fee_" + syouhin + ".json"
    ### 情感统计结果
    outFileEmotion = outPath + "Emotion_" + syouhin + ".json"
    # 商品分词结果
    outFile = outPath + "Summary_" + syouhin + ".txt"
    # 全部商品分词结果合并
    outFileAll = outPath + "all_summary.txt"
    # 全部评论合并
    outFileMerge = outPath + "all_merge.txt"
    # 商品评论分词结果（mapreduce用）
    outSinglePath = outPath + syouhin + "_mapred/"
    if not os.path.exists(outSinglePath):
        os.makedirs(outSinglePath)
        # 商品评论分词结果（mapreduce用）
    outSingleJSON = outPath + syouhin + "_json/"
    if not os.path.exists(outSingleJSON):
        os.makedirs(outSingleJSON)
        # 获取文件列表
    fileList = os.listdir(inPath)
    for i in range(0, len(fileList)):
        f = fileList[i]
        jsonData = {}
        path = os.path.join(inPath, f)
        # 只对文件夹下的文件操作
        if os.path.isfile(path):
            print "-----Processing File:", f
            # 编辑文件路径
            fileName = inPath + f
            # 获取类别(验证用)
            catlog = f.split("_")[0]
            # 获取文件内容
            content = readDataFile(fileName)
            # 获取评论属性
            info = catlog + "#" + content[1]
            commentInfo = content[1].replace("（", "").replace("）", "").split("/")

            # 编辑JSON文件内容
            jsonData['product'] = catlog
            jsonData['region'] = commentInfo[0]
            jsonData['age'] = commentInfo[1]
            jsonData['gender'] = commentInfo[2]
            jsonData['car'] = commentInfo[3]
            jsonData['fee'] = commentInfo[4]
            jsonData['emotion'] = content[2]
            jsonData['comment'] = content[0]

            # 加入list 用于统计
            region.append(commentInfo[0])  # 地域
            age.append(commentInfo[1])  # 年龄
            gender.append(commentInfo[2])  # 性别
            car.append(commentInfo[3])  # 车型
            fee.append(commentInfo[4].split(" ")[1])  # 保险费
            emotion.append(content[2])  # 情感

            # 定义分词器
            mt = MeCab.Tagger('mecabrc')
            node = mt.parseToNode(content[0])
            words, nouns = get_surfaces(node)
            wordList = zip(words, nouns)
            attr = []
            if "N" in mode:
                # 筛选名词结果写入CSV
                attr.append(u'名詞')
            if "A" in mode:
                # 筛选形容詞结果写入CSV
                attr.append(u'形容詞')
            if "V" in mode:
                # 筛选動詞结果写入CSV
                attr.append(u'動詞')
            if mode == 'all':
                attr = None
            # 筛选名词结果写入CSV
            selectList = select_feature(wordList, attr)
            # 分词后结果输出集合
            outWord = []
            # 评论者信息
            outWord.append(info)
            # 评论情感标志
            outWord.append(content[2])
            # 分词前文本合并集合
            outMerge = []
            # 评论者情报
            outMerge.append(info)
            # 评论情感标志
            outMerge.append(content[2])
            # 评论本体
            outMerge.append(content[0])
            mapred = []
            # 去掉停用词
            for (w, a) in selectList:
                #                 print "w=",w
                if w not in stopwords:
                    outWord.append(w)
                    mapred.append((w, a))
                    wordFreq.append(w)
            # 为每一个目录(商品)，汇总生成一个文件
            writeOutput(outFile, outWord, "a")
            # 每一个CSV，生成一个分词文件，当作mapreduce的输入
            singleFile = outSinglePath + f.split('.')[0] + "_mapred.txt"
            writeSingleFile(singleFile, mapred, "wb")

            # 每一个CSV，生成一个json文件
            jsonFile = outSingleJSON + f.split('.')[0] + "_info.json"
            writeJSON(jsonFile, jsonData)

            # 把所有的CSV文件的评论内容，合并到一个文件里
            writeOutput(outFileMerge, outMerge, "a")

            # 把所有的CSV文件的分词筛选结果，写入一个文件里
            writeOutput(outFileAll, outWord, "a")

            # 计算一个商品所有评论的词频，写入文件
    filterList = [u"AIU", u"事故", u"保険料", u"対応", u"補償内容", u"車", u"保険", u"代理店", u"損保", u"補償",
                  u"アクサ", u"契約", u"連絡", u"アクサダイレクト", u"イーデザイン", u"電話", u"保険会社",
                  u"ニッセイ", u"ダイレクト", u"三井", u"サービス", u"三井住友海上", u"ジャパン", u"日新火災",
                  u"加入", u"セゾン", u"チューリッヒ", u"担当者", u"担当", u"必要", u"会社", u"内容", u"車両保険",
                  u"自動車保険", u"インターネット", u"ネット"]
    writeCountJSON(outFileFQ, wordFreq, top_n=40, sort=True, itemFilter=True, filterSet=filterList)
    writeCountJSON(outFileRegion, region, top_n=50)
    writeCountJSON(outFileAge, age, top_n=10)
    writeCountJSON(outFileGender, gender, top_n=10)
    writeCountJSON(outFileCar, car, top_n=10)
    writeCountJSON(outFileFee, fee, top_n=50)
    writeCountJSON(outFileEmotion, emotion)


def processAll():
    # 读入停用词一览
    ssfile = "./conf/Japanese_stopwords.txt"
    ssList = readStopWord(ssfile)
    input_common = ["./data/comment/aiu-sonpo/",
                    "./data/comment/axa-direct/",
                    "./data/comment/e-design/",
                    "./data/comment/ioi-sonpo/",
                    "./data/comment/mitsui-direct/",
                    "./data/comment/ms-ins/",
                    "./data/comment/nipponkoa/",
                    "./data/comment/nisshin-kasai/",
                    "./data/comment/saison/",
                    "./data/comment/sonpo24/",
                    "./data/comment/sony-sonpo/",
                    "./data/comment/zurich/"]

    input_lstm = ["./data/comment_lstm/aiu-sonpo/",
                  "./data/comment_lstm/axa-direct/",
                  "./data/comment_lstm/e-design/",
                  "./data/comment_lstm/ioi-sonpo/",
                  "./data/comment_lstm/mitsui-direct/",
                  "./data/comment_lstm/ms-ins/",
                  "./data/comment_lstm/nipponkoa/",
                  "./data/comment_lstm/nisshin-kasai/",
                  "./data/comment_lstm/saison/",
                  "./data/comment_lstm/sonpo24/",
                  "./data/comment_lstm/sony-sonpo/",
                  "./data/comment_lstm/zurich/"]
    # lstm 用的输入文件
    input_path = input_lstm
    output_common = "./data/processed/"
    output_lstm = "./data/processed_lstm/"
    # lstm
    output_path = output_lstm
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        # 筛选名词
    #     processFile(input_path,output_path,"N")
    # 筛选名词和形容次
    #     processFile(input_path,output_path,"NA")
    # 筛选名词和形容次，动词
    #     processFile(input_path,output_path,"NAV")
    for p in input_path:
        print "Process Folder:", p
        #         processFile(p,output_path,ssList,"NA")
        processFile(p, output_path, ssList, "all")


def processTest():
    # 读入停用词一览
    ssfile = "./conf/Japanese_stopwords.txt"
    ssList = readStopWord(ssfile)

    input_path = ["./data/comment/aiu-sonpo/"]

    output_path = "./data/test/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

        # 筛选名词
    #     processFile(input_path,output_path,"N")
    # 筛选名词和形容次
    #     processFile(input_path,output_path,"NA")
    # 筛选名词和形容次，动词
    #     processFile(input_path,output_path,"NAV")
    for p in input_path:
        print "Process Folder:", p
        processFile(p, output_path, ssList, "NA")


def main():
    # 处理全体数据
    processAll()
    # 处理测试数据
    #     processTest()

if __name__ == '__main__':
    main()