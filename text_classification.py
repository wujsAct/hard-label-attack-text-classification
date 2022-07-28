import numpy as np
from pyhanlp import SafeJClass

import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH
from sklearn.naive_bayes import MultinomialNB
'''
好神奇~~
The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.
'''


def test_data_path():
    """
  获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
  :return:
  """
    data_path = os.path.join(HANLP_DATA_PATH, 'test')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path


## 验证是否存在 MSR语料库，如果没有自动下载
def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        return dest_path

    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path


sogou_corpus_path = ensure_data(
    '搜狗文本分类语料库迷你版',
    'http://file.hankcs.com/corpus/sogou-text-classification-corpus-mini.zip')

## ===============================================
## 以下开始朴素贝叶斯分类

NaiveBayesClassifier = SafeJClass(
    'com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')


def train_or_load_classifier():
    model_path = sogou_corpus_path + '.ser'
    print('model path:', model_path)

    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()  # 朴素贝叶斯分类器
    classifier.train(sogou_corpus_path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return NaiveBayesClassifier(model)


def predict(classifier, text):
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # 如需获取离散型随机变量的分布，请使用predict接口
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def naba():
    '''
  朴素贝叶斯新闻分类
  '''
    news = fetch_20newsgroups(subset='all')

    x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                        news.target,
                                                        test_size=0.25)
    print(np.shape(x_test), np.shape(y_test))
    print(x_test[0], y_test[0])
    print(x_test[10], y_test[10])
    # 对文本进行特征提取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print('预测的文章类别为：', y_predict)

    print('准确率为：', mlt.score(x_test, y_test))

    # print('每个类别的精确率和召回率：',
    #       classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    # classifier = train_or_load_classifier()
    # # predict(classifier, "C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练")
    # # predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
    # # predict(classifier, "研究生考录模式亟待进一步专业化")
    # # predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
    # # predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")
    # # predict(classifier, "SpaceX在24小时内连续发射两批“星链”卫星，发射数累计达2600颗")
    # # predict(classifier,"还有王法吗？以后各国想发射卫星，发射时机还得由马斯克来决定！")
    # # predict(classifier,"国产C919大飞机批量交付在即！即便拿不到欧美适航证，也不愁销路")
    # # predict(classifier,"操盘必读｜央行下调房贷利率，首架预交付C919首飞成功产业链迎机遇！欧美股市全线暴涨")
    # # predict(classifier,"中国东航，C919即将正式商用")
    # # predict(classifier,"FAA和EASA适航证卡脖子？C919飞机启动国内市场，仍让欧美如坐针毡")
    # # predict(classifier, "FAA和EASA适航证卡脖子？C919飞机启动国内市场")
    # # predict(classifier, "C919飞机被适航证FAA卡脖子")
    # predict(classifier, "飞机的卡脖子技术包括适航证")
    # predict(classifier, "卫星")
    naba()