import pandas as pd
from joblib.numpy_pickle_utils import xrange
import wash
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 使用pandas包的read_csv函数读取训练数据集,选12条数据进行测试
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, nrows=12)
# 查看原始训练数据
print(train)
# 得到训练数据的size，其实是3
num_reviews = train["review"].size
print(num_reviews)
# 初始化一个空的列表来装清洗后的数据
clean_train_reviews = []
# 进行训练集数据的清洗并查看结果
for i in xrange(0, num_reviews):
    clean_train_reviews.append(wash.review_to_words(train["review"][i]))
print(clean_train_reviews)
# 初始化一个对象用于统计评论中的单词频度，为了限制后面的每个特征向量的大小，最多统计前100个出现最多的词的频度
vector = CountVectorizer(analyzer="word",
                         tokenizer=None,
                         preprocessor=None,
                         stop_words=None,
                         max_features=100)
# 将干净数据特征向量化
train_data_features = vector.fit_transform(clean_train_reviews)
# 将特征向量转为数组,查看数组规模
train_data_features = train_data_features.toarray()
print(train_data_features.shape)
# 查看词汇表
vocab = vector.get_feature_names()
print(vocab)
# 查看词汇表中每个单词的计数
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)
# 开始训练
print("start train......")
# 使用随机森林算法，设置树为10
forest = RandomForestClassifier(n_estimators=10)
# 使用数字训练特征和每个特征向量的原始情绪标签
forest = forest.fit(train_data_features, train["sentiment"])
# 读取4条测试数据
test = pd.read_csv("testData.tsv", header=0, delimiter="\t",
                   quoting=3, nrows=4)
print(test.shape)
# 查看测试数据的评论条数，实际为4
test_num_reviews = len(test["review"])
print(test_num_reviews)
# 创建一个空的列表装清洗后的测试数据
clean_test_reviews = []
# 清理测试数据，并查看
for i in xrange(0, test_num_reviews):
    clean_test_reviews.append(wash.review_to_words(test["review"][i]))
print(clean_test_reviews)
# 测试数据特征化
test_data_features = vector.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print(test_data_features)
# 使用训练结果进行预测
result = forest.predict(test_data_features)
print(result)
# 格式化输出结果
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
# 创建csv文件
output.to_csv("movie_word_model_beta.csv", index=False, quoting=3)
