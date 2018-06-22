# -*- coding: utf-8 -*-

"""
    项目名称：根据日常新闻预测股市动向
"""
import os
import constant
import pandas as pd
from tools import clean_text, split_train_test, get_word_list_from_data, \
    extract_feat_from_data, get_best_model
import nltk
from nltk.text import TextCollection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def main():
    """
        主函数
    """
    # Step 1: 处理数据集
    print('===Step1: 处理数据集===')

    if not os.path.exists(constant.cln_text_csv_file):
        print('清洗数据...')
        # 读取原始csv文件
        raw_text_df = pd.read_csv(constant.raw_text_csv_file)

        # 清洗原始数据
        cln_text_df = clean_text(raw_text_df)

        # 保存处理好的文本数据
        cln_text_df.to_csv(constant.cln_text_csv_file, index=None)
        print('完成，并保存结果至', constant.cln_text_csv_file)

    print('================\n')

    # Step 2. 查看整理好的数据集，并选取部分数据作为模型的训练
    print('===Step2. 查看数据集===')
    text_data = pd.read_csv(constant.cln_text_csv_file)
    text_data['date'] = pd.to_datetime(text_data['date'])
    text_data.set_index('date', inplace=True)
    print('各类样本数量：')
    print(text_data.groupby('label').size())

    # Step 3. 分割训练集和测试集
    print('===Step3. 分割训练集合测试集===')
    train_text_df, test_text_df = split_train_test(text_data)
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：')
    print(train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：')
    print(test_text_df.groupby('label').size())
    print('================\n')

    # Step 4. 特征提取
    print('===Step4. 文本特征提取===')
    # 计算词频
    n_common_words = 200

    # 将训练集中的单词拿出来统计词频
    print('统计词频...')
    all_words_in_train = get_word_list_from_data(train_text_df)
    fdisk = nltk.FreqDist(all_words_in_train)
    common_words_freqs = fdisk.most_common(n_common_words)
    print('出现最多的{}个词是：'.format(n_common_words))
    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 在训练集上提取特征
    text_collection = TextCollection(train_text_df['text'].values.tolist())
    print('训练样本提取特征...')
    train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
    print('完成')
    print()

    print('测试样本提取特征...')
    test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
    print('完成')
    print('================\n')

    # 特征处理
    # 特征范围归一化
    scaler = StandardScaler()
    tr_feat_scaled = scaler.fit_transform(train_X)
    te_feat_scaled = scaler.transform(test_X)

    # 3.6 特征选择
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    tr_feat_scaled_sel = sel.fit_transform(tr_feat_scaled)
    te_feat_scaled_sel = sel.transform(te_feat_scaled)

    # 3.7 PCA降维操作
    pca = PCA(n_components=0.95)  # 保留95%贡献率的特征向量
    tr_feat_scaled_sel_pca = pca.fit_transform(tr_feat_scaled_sel)
    te_feat_scaled_sel_pca = pca.transform(te_feat_scaled_sel)
    print('特征处理结束')
    print('处理后每个样本特征维度：', tr_feat_scaled_sel_pca.shape[1])

    # Step 5. 训练模型
    models = []
    print('===Step5. 训练模型===')
    print('1. 朴素贝叶斯模型：')
    gnb_model = GaussianNB()
    gnb_model.fit(tr_feat_scaled_sel_pca, train_y)
    models.append(['朴素贝叶斯', gnb_model])
    print('完成')
    print()

    print('2. 逻辑回归：')
    lr_param_grid = [
        {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    ]
    lr_model = LogisticRegression()
    best_lr_model = get_best_model(lr_model,
                                   tr_feat_scaled_sel_pca, train_y,
                                   lr_param_grid, cv=3)
    models.append(['逻辑回归', best_lr_model])
    print('完成')
    print()

    print('3. 支持向量机：')
    svm_param_grid = [
        {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svm_model = svm.SVC(probability=True)
    best_svm_model = get_best_model(svm_model,
                                    tr_feat_scaled_sel_pca, train_y,
                                    svm_param_grid, cv=3)
    models.append(['支持向量机', best_svm_model])
    print('完成')
    print()

    print('4. 随机森林：')
    rf_param_grid = [
        {'n_estimators': [10, 50, 100, 150, 200]}
    ]

    rf_model = RandomForestClassifier()
    best_rf_model = get_best_model(rf_model,
                                   tr_feat_scaled_sel_pca, train_y,
                                   rf_param_grid, cv=3)
    rf_model.fit(tr_feat_scaled_sel_pca, train_y)
    models.append(['随机森林', best_rf_model])
    print('完成')
    print()

    # Step 6. 测试模型
    print('===Step6. 测试模型===')
    for i, model in enumerate(models):
        print('{}-{}'.format(i + 1, model[0]))
        # 输出准确率
        print('准确率：', accuracy_score(test_y, model[1].predict(te_feat_scaled_sel_pca)))
        print('AUC：', roc_auc_score(test_y, model[1].predict_proba(te_feat_scaled_sel_pca)[:, 0]))
        # 输出混淆矩阵
        print('混淆矩阵')
        print(confusion_matrix(test_y, model[1].predict(te_feat_scaled_sel_pca)))
        print()


if __name__ == '__main__':
    main()
