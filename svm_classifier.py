import pandas as pd
from utils import load_corpus, load_tnews_corpus, load_MR_corpus, load_AG_corpus, stopwords, pre_processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC,SVC
from sklearn.decomposition import TruncatedSVD


class SVM_Cls(object):
  def __init__(self, df_train):
    self.vectorizer = TfidfVectorizer(
                                      token_pattern='\[?\w+\]?',
                                      stop_words=stopwords)
    X_train = self.vectorizer.fit_transform(df_train["words"])
    y_train = df_train["label"]
    self.clf = LinearSVC(dual=False)
    self.clf.fit(X_train, y_train)
  
  def pred_test(self, df_test):
    X_test = self.vectorizer.transform(df_test["words"])
    y_test = df_test["label"]
    y_pred = self.clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred))
    print("准确率:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
  data_tag = 'mr'
  
  if data_tag == 'weibo2018':
    TRAIN_PATH = "data/weibo2018/train.txt"
    TEST_PATH = "data/weibo2018/new_test.txt"
    
    train_data = load_corpus(TRAIN_PATH)
    test_data = load_corpus(TEST_PATH)
  elif data_tag == 'tnews':
    TRAIN_PATH = "data/tnews/train.json"
    TEST_PATH = "data/tnews/dev.json"
    LABEL_PATH = "data/tnews/labels.json"
    
    train_data = load_tnews_corpus(data_path=TRAIN_PATH, label_path=LABEL_PATH)
    test_data = load_tnews_corpus(data_path=TEST_PATH, label_path=LABEL_PATH)
  elif data_tag == 'mr':
    TRAIN_PATH = "data/mr_all/train.csv"
    TEST_PATH = "data/mr_all/dev.csv"
    train_data = load_MR_corpus(TRAIN_PATH)
    test_data = load_MR_corpus(TEST_PATH)
  elif data_tag == 'ag_news':
    TRAIN_PATH = "data/ag_news_csv/train.csv"
    TEST_PATH = "data/ag_news_csv/test.csv"
    
    train_data = load_AG_corpus(TRAIN_PATH)
    test_data = load_AG_corpus(TEST_PATH)
  
  df_train = pd.DataFrame(train_data, columns=["words", "label"])  # label 0:表示负面，label 1: 表示正面
  df_test = pd.DataFrame(test_data, columns=["words", "label"])
  df_train.head()
  
  NB_Class = SVM_Cls(df_train)
  NB_Class.pred_test(df_test)