# -*- coding: utf-8 -*-
import csv
import json

import jieba
import re
#中文繁体转简体
import opencc
import nltk
from nltk.corpus import stopwords
import gensim

import checklist_data_gen

en_stopwords = []
with open("/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/data/stopwords-en.txt",
          "r", encoding="utf8") as f:
    for w in f:
        en_stopwords.append(w.strip())

converter1 = opencc.OpenCC('t2s.json')
from str_level_filter import URL_REGEX, PHONE_REGEX, COLON_REGEX, is_chinese_char, remove_emoji3, REPLY_MENTION_REGEX, \
    TM_REGEX, ALPHA_NUM_REGEX

stopwords = []
with open("/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/data/stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())
      
stopwords_dict = { word:1 for word in stopwords }
      
def clean_str(string, TREC=False, del_stop=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    
    ret = []
    # print(stopwords_dict)
    # print(string.split(' '))
    for wd in string.split(' '):
        if wd.lower() in en_stopwords and del_stop:
            # print('del word:', wd)
            continue
        else:
            ret.append(wd)
    
    string = ' '.join(ret)
    return string.strip() if TREC else string.strip().lower()

def load_corpus(path, is_adv=False, adv_dict=False, is_bert='0', label_type='int',noise_type=None):
    """
    加载语料库
    weibo2018
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            if noise_type=='add':
                if seniment=='1':
                    content = content + '我心情很好。'
                if seniment=='0':
                    content = content + '你情绪低落。'
            if is_bert=='0':
                content = processing(content, is_adv=is_adv, adv_dict=adv_dict)
            else:
                content = processing_bert(content)
            if label_type=='int':
                data.append((content, int(seniment)))
            else:
                data.append((content, seniment))
    return data

def load_tnews_corpus(label_path, data_path, is_adv=False, adv_dict=False, is_bert='0', label_type='int',
                      noise_type=None):
    """
    加载tnews
    """
    label2idx = {}
    with open(label_path,"r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            # print(item, item.get('label'))
            label2idx[item['label']] = len(label2idx)
    
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            content_pre = item['sentence']
            if noise_type == 'add':
                keyword = item['keywords']
                if keyword == '':
                    content_pre = item['sentence']
                else:
                    content_pre = content_pre + ' ' + keyword.split(',')[0]
            if is_bert=='0':
                content = processing(content_pre, is_adv=is_adv, adv_dict=adv_dict)
            else:
                content = processing_bert(content_pre)
            label = label2idx[item['label']]
            if label_type == 'int':
                data.append((content, label))
            else:
                data.append((content, str(label)))
    return data

def load_MR_corpus(path, is_adv=False, adv_dict=None, clean=True, lower=True, noise_type=None):
    data = []
    with open(path, encoding="utf8") as fin:
        for line in fin:
            content, label = line.split('\t')
            label = int(label)
            if noise_type == 'add':
                if label == 0:
                    content = content + ' ' + 'This work is spectacular'
                else:
                    content = content + ' ' + 'This work is bad'
            if clean:
                content = clean_str(content.strip()) if clean else content.strip()
            if is_adv:
                ret = []
                for wd in content.split(' '):
                    if wd in adv_dict:
                        continue
                    else:
                        ret.append(wd)
                content = " ".join(ret)
                
            if lower:
                content = content.lower()
            # print(content)
            data.append((content, label))
    return data

def load_AG_corpus(path, is_adv=False, adv_dict=None, clean=True, lower=True,
                   noise_type=None):
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if noise_type == 'add':
                content = line[2]
            else:
                content = line[1] + " " + line[2]
              
            label = int(line[0])
            if clean:
                content = clean_str(content.strip()) if clean else content.strip()
            if is_adv:
                ret = []
                for wd in content.split(' '):
                    if wd in adv_dict:
                        continue
                    else:
                        ret.append(wd)
                content = " ".join(ret)
            if lower:
                content = content.lower()
            data.append((content, label))
    return data
        
def load_corpus_bert(path):
    """
    加载语料库; weibo2018
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def pre_processing(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = URL_REGEX.sub(' ', text)  #去除URL
    text = PHONE_REGEX.sub(' ', text)  #去除电话
    text = REPLY_MENTION_REGEX.sub(' ', text) #去除回复
    text = ALPHA_NUM_REGEX.sub(' ',text)
    text = remove_emoji3(text)  #删除emoji
    text = TM_REGEX.sub(' ',text) #tm
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = converter1.convert(text)  # 汉子
    text = text.replace('//', ' ')
    
    return text

def processing(text, is_adv=False, adv_dict=None):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)  # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)  # 去除 @xxx (用户名)
    # text = converter1.convert(text)  # 汉子
    # text = text.replace('//', ' ')
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    # print(text)
    # 分词
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # 对否定词`不`做特殊处理: 与其后面的词进行拼接
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # 列表切片赋值的酷炫写法
    
    
    if is_adv:
        #del
        '''
        ret_words = []
        for wd in words:
            if wd in adv_dict:
                continue
            else:
                ret_words.append(wd)
        words = ret_words
        '''
        #tongyi
        words = checklist_data_gen.ch_tongyi_word(words)
    # 用空格拼接成字符串
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    return text


def load_adv_corpus(fname):
    org = 0
    new = 1
    idx = 1
    rets = []
    with open(fname) as lines:
        for line in lines:
            if idx % 3 == 1:
                org = line.strip()
            if idx % 3 == 2:
                new = line.strip()
            
            if idx % 3 == 0:
                org_items = org.split('\t')
                new_items = new.split('\t')
                label = org_items[1]
                context = new_items[2]
                # print(label, context)
                rets.append((context, int(label)))
            idx += 1
    return rets


def load_org_adv_corpus(fname, test_data):
    
    org = 0
    new = 1
    idx = 1
    rets = []
    sample_idx = 0
    with open(fname) as lines:
        for line in lines:
            if idx % 3 == 1:
                org = line.strip()
            if idx % 3 == 2:
                new = line.strip()
            
            if idx % 3 == 0:
                org_items = org.split('\t')
                new_items = new.split('\t')
                label = test_data[sample_idx][1]
                context = test_data[sample_idx][0]
                # print(label, context)
                rets.append((context, label))
                sample_idx+=1
            idx += 1
    print('all test datas:', len(rets))
    return rets

if __name__ == '__main__':
    dir_path = '/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/adv_results'
    load_adv_corpus(dir_path+'/ag-cnn-adv-10')
    