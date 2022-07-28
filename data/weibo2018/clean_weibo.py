from partd import pandas

from utils import pre_processing
import pandas as pd

def gen_new_test_weibo():
  pd_all = pd.read_csv('weibo_senti_100k.csv')
  
  # print('评论数目（总体）：%d' % pd_all.shape[0])
  # print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
  # print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])
  
  # print(pd_all[pd_all.label==1].sample(frac=0.1).shape)
  # print(pd_all[pd_all.label==0].sample(frac=0.1).shape)
  
  tidx = 0
  
  idx_list = []
  label_list = []
  review_list = []
  for idx, item in pd_all[pd_all.label == 1].iterrows():
    # print(str(idx)+','+str(item['label'])+','+str(processing(item['review'])))
    idx_list.append(str(idx))
    label_list.append(item['label'])
    review_list.append(pre_processing(item['review']))
    if tidx == 250:
      break
    tidx += 1
  
  tidx = 0
  for idx, item in pd_all[pd_all.label == 0].iterrows():
    # print(str(idx)+','+str(item['label'])+','+str(processing(item['review'])))
    idx_list.append(str(idx))
    label_list.append(item['label'])
    review_list.append(pre_processing(item['review']))
    if tidx == 250:
      break
    tidx += 1
  
  out_df = pd.DataFrame({'label': label_list,
                         'review': review_list})
  
  out_df.to_csv('new_test.txt', header=False)
if __name__ == '__main__':
  gen_new_test_weibo()