import bz2
import csv
import gzip
import random

import numpy as np
import spacy
from checklist.perturb import Perturb
from checklist.expect import Expect
from checklist.test_types import MFT, INV, DIR
from pypinyin import pinyin, lazy_pinyin, Style
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
nlp_chinese = spacy.load('zh_core_web_sm')


def load_ch_name(fname='/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/data/Chinese_Names_Corpus_Gender（120W）.txt'):
  name_dict ={}
  gender_dict={'男': set(), '女': set(), '未知': set()}
  with open(fname) as lines:
    for line in tqdm(lines):
      items = line.strip().split(',')
      # print(items)
      if len(items) == 2:
        name_dict[items[0]] = items[1]
        gender_dict[items[1]].add(items[0])
      
  
  print(len(name_dict))
  if '赵丽颖' in name_dict:
    print(name_dict['赵丽颖'])
  
  print(len(gender_dict['男']), len(gender_dict['女']))
  
  return name_dict, gender_dict

def load_ch_location(fname='/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/data/world_cities.txt'):
  country2city_dict = {}
  city2coutry_dict = {}
  
  with open(fname, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
      city, country = line[2], line[5]
      if country not in country2city_dict:
        country2city_dict[country]=set()
      
      country2city_dict[country].add(city)
      city2coutry_dict[city]=country
  return country2city_dict, city2coutry_dict


def load_embedding_npz(path):
  data = np.load(path)
  return [w.decode('utf8') for w in data['words']], data['vals']


def load_embedding_txt(path):
  print('load word embeddings ...')
  if path.endswith(".gz"):
    file_open = gzip.open
    open_mode = "rb"
  elif path.endswith(".bz2"):
    file_open = bz2.open
    open_mode = "rb"
  else:
    file_open = open
    open_mode = "r"
  words = []
  vals = []
  real_words = {}
  with file_open(path, open_mode) as fin:
    fin.readline()
    for line in tqdm(fin):
      line = line.rstrip()
      if line:
        if open_mode == 'r':
          parts = line.split(' ')
        else:
          parts = line.decode('utf8').strip().split(' ')
        word = parts[0]
        val_list = parts[1:]
        
        if word not in real_words:
          real_words[word] = 1
        else:
          print('duplicate word:', word)
          continue
        words.append(word)
        vals += [float(x) for x in val_list]
  return words, np.asarray(vals).reshape(len(words), -1)


def load_embedding(path):
  if path.endswith(".npz"):
    return load_embedding_npz(path)
  else:
    return load_embedding_txt(path)
  
def get_all_char_pinyin():
  wd2pinyin_dict = {}
  pinyin2wd_dict = {}
  words, _ = load_embedding("/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/data/sgns.weibo.word.bz2")
  # words = ['我', '是', '吴', '俊', '爽','黄小磨']
  for ch in tqdm(words):
    ch_pinyin = pinyin(ch, style=Style.TONE3, heteronym=False, strict=False)
    # print(ch_pinyin)
    ret = ''
    for p_li in ch_pinyin:
      ret = ret + p_li[0] + ' '
    
    p = ret.strip()
    
    if ch not in wd2pinyin_dict:
      wd2pinyin_dict[ch] = [p]
    else:
      wd2pinyin_dict[ch].append(p)
    
    if p not in pinyin2wd_dict:
      pinyin2wd_dict[p] = set()
    pinyin2wd_dict[p].add(ch)
    
    # print(ch, wd2pinyin_dict[ch])
  # print(wd2pinyin_dict)
  # print(pinyin2wd_dict)
  #
  # print(pinyin_dict)
  return pinyin2wd_dict, wd2pinyin_dict


def ch_tongyi_word(test_x):
  pinyin2wd_dict, wd2pinyin_dict = get_all_char_pinyin()
  
  # test_x, test_y = dataloader.read_weibo_corpus('data/weibo2018/test.txt', shuffle=False)
  ret_x = []
  revise_sent = 0
  revise_percent = []
  for data in test_x:
    ret_data = []
    flag = False
    revise_time = 0
    for ch in data:
      if ch in wd2pinyin_dict:
        p = wd2pinyin_dict[ch][0]
        ty_list = pinyin2wd_dict[p]
        ty_list.discard(ch)
        if len(ty_list) != 0:
          ret_data.append(random.choice(list(ty_list)))
          flag = True
          revise_time += 1
        else:
          ret_data.append(ch)
      else:
        ret_data.append(ch)
    
    if flag:
      revise_sent += 1
    # print(data)
    # print(ret_data)
    ret_x.append(ret_data)
    revise_percent.append(float(revise_time) / float(len(data)))
    assert (len(data) == len(ret_data))
    # print('------------')
  print(revise_sent)
  print(np.average(revise_percent))
  assert (len(ret_x) == len(test_x))
  return ret_x


def ch_tongyi_word_bert(test_x):
  pinyin2wd_dict, wd2pinyin_dict = get_all_char_pinyin()
  
  # test_x, test_y = dataloader.read_weibo_corpus('data/weibo2018/test.txt', shuffle=False)
  ret_x = []
  revise_sent = 0
  revise_percent = []
  for item in test_x:
    data = item[0].split(" ")
    ret_data = []
    flag = False
    revise_time = 0
    for ch in data:
      if ch in wd2pinyin_dict:
        p = wd2pinyin_dict[ch][0]
        ty_list = pinyin2wd_dict[p]
        ty_list.discard(ch)
        if len(ty_list) != 0:
          ret_data.append(random.choice(list(ty_list)))
          flag = True
          revise_time += 1
        else:
          ret_data.append(ch)
      else:
        ret_data.append(ch)
    
    if flag:
      revise_sent += 1
    # print(data)
    # print(ret_data)
    ret_x.append((" ".join(ret_data), item[1]))
    revise_percent.append(float(revise_time) / float(len(data)))
    assert (len(data) == len(ret_data))
    # print('------------')
  print(revise_sent)
  print(np.average(revise_percent))
  assert (len(ret_x) == len(test_x))
  return ret_x


# 17063
# 0.7407047660117432
def ner_replace(dataset=['John is a man']):
  pdataset = list(nlp.pipe(dataset))
  ret = Perturb.perturb(pdataset, Perturb.change_locations, n=2)
  print(ret.data)

def add_negation(dataset=['John is a man']):
  pdataset = list(nlp.pipe(dataset))
  ret = Perturb.perturb(pdataset, Perturb.add_negation)
  print(ret.data)

def add_typos(test_x):
  # dataset = ['This was a very nice movie directed by John Smith.']
  ret_x = []
  for data in test_x:
    
    if len(data) ==0:
      ret_x.append(data)
      continue
    dataset = [" ".join(data)]
    ret = Perturb.perturb(dataset, Perturb.add_typos)
    rev_data = ret.data[0][1].split(" ")
    ret_x.append(rev_data)
    print(data)
    print(rev_data)
    print(" ".join(rev_data))
    print("------------")
  return ret_x


def add_typos_bert(test_x):
  # dataset = ['This was a very nice movie directed by John Smith.']
  ret_x = []
  for item in test_x:
    data = item[0].split(" ")
    
    if len(data) == 0:
      ret_x.append((item[0], item[1]))
      continue
    dataset = [" ".join(data)]
    ret = Perturb.perturb(dataset, Perturb.add_typos)
    rev_data = ret.data[0][1].split(" ")
    ret_x.append((" ".join(rev_data), item[1]))
    print(data)
    print(rev_data)
    print(" ".join(rev_data))
    print("------------")
  return ret_x
  
def add_INV(test_x):
  # test_x = ['This was a very nice movie directed by John Smith.',
  #            'This was a very nice movie happended in New York.',
  #           'Junshuang Wu born in 1992.']
  # t = Perturb.perturb(dataset, Perturb.add_typos)
  
  ret_x = []
  idx = 0
  flag = None
  for data in test_x:
    if len(data) == 0:
      ret_x.append(data)
      continue
    
    dataset = [" ".join(data)]
    pdataset = list(nlp.pipe(dataset))
    ret = Perturb.perturb(pdataset, Perturb.change_names, n=2)
    rev_data = ret.data
    flag = 'name'
    
    if len(rev_data)==0:
      ret = Perturb.perturb(pdataset, Perturb.change_location, n=2)
      rev_data = ret.data
      flag='location'
      
    if len(rev_data)==0:
      ret = Perturb.perturb(pdataset, Perturb.change_number, n=2)
      rev_data = ret.data
      flag = 'number'
    
    if len(rev_data)==0:
      ret_x.append(data)
    else:
      ret_x.append(rev_data[0][1].split(' '))
      print('['+flag+'] has changed...')
      print(idx)
      print(data)
      print(ret_x[-1])
      print('-------------------------')
    
    idx += 1
    
    # print(ret_x[-1])
  
  return ret_x


def add_INV_bert(test_x):
  # test_x = ['This was a very nice movie directed by John Smith.',
  #            'This was a very nice movie happended in New York.',
  #           'Junshuang Wu born in 1992.']
  # t = Perturb.perturb(dataset, Perturb.add_typos)
  
  ret_x = []
  for item in test_x:
    data = item[0].split(" ")
  
    if len(data) == 0:
      ret_x.append((item[0], item[1]))
      continue
      
    dataset = [" ".join(data)]
    
    pdataset = list(nlp.pipe(dataset))
    ret = Perturb.perturb(pdataset, Perturb.change_names, n=2)
    rev_data = ret.data
    
    if len(rev_data) == 0:
      ret = Perturb.perturb(pdataset, Perturb.change_location, n=2)
      rev_data = ret.data
    
    if len(rev_data) == 0:
      ret = Perturb.perturb(pdataset, Perturb.change_number, n=2)
      rev_data = ret.data
    
    if len(rev_data) == 0:
      ret_x.append((item[0], item[1]))
    else:
      ret_x.append((rev_data[0][1], item[1]))
      print(data)
      print(ret_x[-1])
      print('change.....')
      
  return ret_x

def add_ch_INV(test_x=None):
  # test_x = ['赵丽颖出生于河北省']
  # t = Perturb.perturb(dataset, Perturb.add_typos)
  
  name_dict, gender_dict = load_ch_name()
  per_sets = set(name_dict.keys())
  country2city_dict, city2coutry_dict = load_ch_location()
  
  change_idx = 0
  ret_x = []
  line_idx = 0
  for data in tqdm(test_x):
    line_idx += 1
    noise_type = []
    dataset = "".join(data)
    doc = nlp_chinese(dataset)
    
    flag=False
    ret_data = []
    for token in doc:
      token_text = token.text
      token_type = token.ent_type_
      
      if token_type=='PERSON':
        new_token = random.choice(list(per_sets))
        ret_data.append(new_token)
        # print('PERSON:', token_text, new_token)
        flag=True
        noise_type.append('PERSON')
      else:
        ret_data.append(token_text)
    # print(ret_data, len(ret_data))
    # print(doc, len(doc))
    # assert(len(ret_data) == len(doc))
    
    if not flag:
      new_data = "".join(ret_data)
      doc = nlp_chinese(new_data)
      
      ret_data = []
      for token in doc:
        token_text = token.text
        token_type = token.ent_type_
        
        if token_text in city2coutry_dict and token_type=='GPE':
          country_name = city2coutry_dict[token_text]
          city_sets = country2city_dict[country_name]
          new_token = random.choice(list(city_sets))
          ret_data.append(new_token)
          flag = True
          # print('GPE:', token_text, new_token)
          noise_type.append('GPE')
        else:
          ret_data.append(token_text)
      # assert(len(ret_data) == len(doc))
      
    if not flag:
      new_data = "".join(ret_data)
      doc = nlp_chinese(new_data)

      ret_data = []
      for token in doc:
        token_text = token.text
        token_type = token.ent_type_
        if token_text in country2city_dict and token_type=='GPE':
          country_sets = set(list(country2city_dict.keys()))
          country_sets.remove(token_text)
          #wujs
          new_token = random.choice(list(country_sets))
          ret_data.append(new_token)
          flag = True
          noise_type.append('GPE')
        else:
          ret_data.append(token_text)
          
    if flag:
      change_idx += 1
      ret_x.append(ret_data)
      print('['+" ".join(noise_type)+']'+" change ...............")
      print(line_idx)
      print(data)
      print(ret_data)
    else:
      ret_x.append(data)
  
  print(change_idx, len(ret_x))
  # assert(len(ret_x)==len(test_x))
  
  # for d1, d2 in zip(ret_x, test_x):
  #   if " ".join(d1) != " ".join(d2):
  #     print(d1)
  #     print(d2)
  #     print('---------------')
  
  return ret_x


def add_ch_INV_bert(test_x):
  # test_x = ['This was a very nice movie directed by John Smith.',
  #            'This was a very nice movie happended in New York.',
  #           'Junshuang Wu born in 1992.']
  # t = Perturb.perturb(dataset, Perturb.add_typos)
  
  name_dict, gender_dict = load_ch_name()
  name_sets = list(name_dict.keys())
  country2city_dict, city2coutry_dict = load_ch_location()
  country_sets = list(country2city_dict.keys())
  city_sets = list(city2coutry_dict.keys())
  
  ret_x = []
  change_idx = 0
  
  for item in test_x:
    data = item[0]

    doc = nlp_chinese(data)
    
    flag = False
    ret_data = []
    for token in doc:
      token_text = token.text
      token_type = token.ent_type_
      # print(token.text, token.pos_, token.dep_, token.ent_type_)
      # and not flag
      if token_text in name_dict and token_type == 'PERSON':
        gender = name_dict[token_text]
        per_sets = gender_dict[gender]
        new_token = random.choice(list(per_sets))
        ret_data.append(new_token)
        print('PERSON:', token_text, new_token)
        flag = True
      else:
        ret_data.append(token_text)
    
    # assert (len(ret_data) == len(doc))

    # if flag == False:
    if True:
      new_data = "".join(ret_data)
      doc = nlp_chinese(new_data)
      
      ret_data = []
      for token in doc:
        token_text = token.text
        token_type = token.ent_type_
        # and not flag
        if token_text in city2coutry_dict and token_type == 'GPE':
          country_name = city2coutry_dict[token_text]
          city_sets = country2city_dict[country_name]
          new_token = random.choice(list(city_sets))
      
          # ret_data.append(new_token)
          flag = True
          # print('GPE:', token_text, new_token)
        else:
          ret_data.append(token_text)
      # assert (len(ret_data) == len(doc))
    
    if flag:
      change_idx += 1
      ret_x.append((" ".join(ret_data), item[1]))
      # print(data)
      # print(ret_data)
      # print('change-----------------')
    else:
      ret_x.append(item)
  
  print(change_idx, len(ret_x))
  # assert (len(ret_x) == len(test_x))
  
  # for d1, d2 in zip(ret_x, test_x):
  #   if d1 != d2:
  #     print(d1)
  #     print(d2)
  #     print('---------------')
      
  return ret_x


def DIV(dataset = ['This was a very nice movie directed by John Smith.']):
  def add_negative(x):
      phrases = ['Anyway, I thought it was bad.', 'Having said this, I hated it', 'The director should be fired.']
      return ['%s %s' % (x, p) for p in phrases]
  
  t = Perturb.perturb(dataset, add_negative)
  monotonic_decreasing = Expect.monotonic(label=1, increasing=False, tolerance=0.1)
  test3 = DIR(**t, expect=monotonic_decreasing)
  print(test3.data)

if __name__ =="__main__":
  print('test...')
  # load_ch_name()
  # load_ch_location()
  add_ch_INV()
  # ch_tongyi_word()
  # add_typos()
