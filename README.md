# movie_words
- 使用IMDB电影评论数据集，进行评论分析
- 数据集来自https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data
- 算法参考同上
- 项目代码结构说明：
1. wash.py：实现原始评论的数据清理，去除了html标签、无意义词
2. main.py：分别使用wash.py清洗原始数据和测试数据，并分别为其构造“单词包”特征向量。使用随机森林算法训练原始特征向量集合得到训练模型。使用训练模型对测试特征向量集合进行预测。
- beta代码结构说明：
1. main_beta.py：由于个人pc无法承载50,000数据工作，笔者学习时，采用了12条训练数据，4条测试数据，遵循3:1。
2. movie_word_model_beta.csv：按照task要求生成标准csv文件
