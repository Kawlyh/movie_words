from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


# 本函数将原始评论清洗为不含标签、索引词的列表
def review_to_words(raw_review):
    # 使用bs4包的get_text清理评论中的标签超链接等
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    # 使用正则表达式将评论中的非字母符号替换为空格
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 将大写字母替换为小写字母，并拆开为多个单词，得到words列表
    words = letters_only.lower().split()
    # 清理掉非索引字，比如说a is the这些词，因为他们无实际意义
    meaningful_words = [w for w in words if not w in stopwords.words("english")]
    # 使用join方法将有意义的序列中的单词用空格连接生成一个新的字符串，并返回
    return " ".join(meaningful_words)
