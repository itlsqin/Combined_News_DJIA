# -*- coding: utf-8 -*-

"""
    项目名称：根据日常新闻预测股市动向
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV


def proc_text(raw_line):
    """
        处理每行的文本数据
        返回分词结果
    """
    raw_line = str(raw_line)
    # 全部转为小写
    raw_line = raw_line.lower()

    # 去除 b'...' 或 b"..."
    if raw_line[:2] == 'b\'' or raw_line[:2] == 'b"':
        raw_line = raw_line[2:-1]

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw_line)
    meaninful_words = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(meaninful_words)


def clean_text(raw_text_df):
    """
        清洗原始文本数据
    """
    cln_text_df = pd.DataFrame()
    cln_text_df['date'] = raw_text_df['Date'].values
    cln_text_df['label'] = raw_text_df['Label'].values
    cln_text_df['text'] = ''

    # 处理25列文本数据，['Top1', ..., 'Top25']
    col_list = ['Top' + str(i) for i in range(1, 26)]

    for i, col in enumerate(col_list):
        raw_text_df[col] = raw_text_df[col].apply(proc_text)
        # 合并列
        cln_text_df['text'] = cln_text_df['text'].str.cat(raw_text_df[col], sep=' ')
        print('已处理{}列.'.format(i + 1))

    return cln_text_df


def split_train_test(data_df):
    """
        分割训练集和测试集
    """
    # 训练集时间范围 2008-08-08 ~ 2014-12-31
    train_text_df = data_df.loc['20080808':'20141231', :]
    # 将时间索引替换为整型索引
    train_text_df.reset_index(drop=True, inplace=True)

    # 测试集时间范围 2015-01-02 ~ 2016-07-01
    test_text_df = data_df.loc['20150102':'20160701', :]
    # 将时间索引替换为整型索引
    test_text_df.reset_index(drop=True, inplace=True)

    return train_text_df, test_text_df


def get_word_list_from_data(text_df):
    """
        将数据集中的单词放入到一个列表中
    """
    word_list = []
    for _, r_data in text_df.iterrows():
        word_list += r_data['text'].split(' ')
    return word_list


def extract_feat_from_data(text_df, text_collection, common_words_freqs):
    """
        特征提取
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    n_sample = text_df.shape[0]
    n_feat = len(common_words_freqs)
    common_words = [word for word, _ in common_words_freqs]

    # 初始化
    X = np.zeros([n_sample, n_feat])
    y = np.zeros(n_sample)

    print('提取特征...')
    for i, r_data in text_df.iterrows():
        if (i + 1) % 100 == 0:
            print('已完成{}个样本的特征提取'.format(i + 1))

        text = r_data['text']

        feat_vec = []
        for word in common_words:
            if word in text:
                # 如果在高频词中，计算TF-IDF值
                tf_idf_val = text_collection.tf_idf(word, text)
            else:
                tf_idf_val = 0

            feat_vec.append(tf_idf_val)

        # 赋值
        X[i, :] = np.array(feat_vec)
        y[i] = int(r_data['label'])

    return X, y


def get_best_model(model, X_train, y_train, params, cv=5):
    """
        交叉验证获取最优模型
        默认5折交叉验证
    """
    clf = GridSearchCV(model, params, cv=cv, verbose=3)
    clf.fit(X_train, y_train)
    return clf.best_estimator_
