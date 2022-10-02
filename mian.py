import pandas as pd
from joblib.numpy_pickle_utils import xrange
from sklearn.ensemble import RandomForestClassifier

import wash
from sklearn.feature_extraction.text import CountVectorizer

# 使用pandas包的read_csv函数读取训练数据集
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# 得到训练数据的size
num_reviews = train["review"].size
# 初始化一个空的列表来装清洗后的数据
clean_train_reviews = []
# 进行训练集数据的清洗
for i in xrange(0, num_reviews):
    clean_train_reviews.append(wash.review_to_words(train["review"][i]))
# 初始化一个对象用于统计评论中的单词频度，为了限制后面的每个特征向量的大小，我们只统计前5000个出现最多的词的频度
vector = CountVectorizer(analyzer="word",
                         tokenizer=None,
                         preprocessor=None,
                         stop_words=None,
                         max_features=5000)
# 将干净训练数据特征向量化
train_data_features = vector.fit_transform(clean_train_reviews)
# 将特征向量转为数组
train_data_features = train_data_features.toarray()
# 开始训练
print("start train......")
# 使用随机森林算法，设置树为100
forest = RandomForestClassifier(n_estimators=100)
# 使用数字训练特征和每个特征向量的原始情绪标签
forest = forest.fit(train_data_features, train["sentiment"])
# 读取测试数据
test = pd.read_csv("testData.tsv", header=0, delimiter="\t",
                   quoting=3)
# 得到测试数据评论数目
test_num_reviews = len(test["review"])
# 创建一个空的列表装清洗后的测试数据
clean_test_reviews = []
# 清理测试数据
for i in xrange(0, test_num_reviews):
    clean_test_reviews.append(wash.review_to_words(test["review"][i]))
# 测试数据特征化
test_data_features = vector.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# 使用训练结果进行预测
result = forest.predict(test_data_features)
# 格式化输出结果
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
# 创建csv文件
output.to_csv("movie_word_model.csv", index=False, quoting=3)
