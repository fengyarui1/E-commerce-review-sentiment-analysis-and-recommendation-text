import os
import re
import jieba
import joblib
from typing import List, Dict, Tuple

def load_stopwords(file_path: str) -> set:
    """加载停用词表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        return stopwords
    except FileNotFoundError:
        print(f"警告: 停用词文件 '{file_path}' 未找到，将使用空停用词表")
        return set()

def clean_text(text: str, stopwords: set) -> List[str]:
    """清洗文本并分词"""
    # 去除HTML标签、URL和特殊字符
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 使用jieba分词
    words = jieba.cut(text)
    
    # 去除停用词和单字符
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return words

def load_and_clean_data(data_dir: str, stopwords: set) -> Tuple[List[List[str]], List[int]]:
    """加载并清洗数据集"""
    texts = []
    labels = []
    
    # 加载负面评论 (标签为0)
    neg_dir = os.path.join(data_dir, 'neg')
    if os.path.exists(neg_dir):
        for file_name in os.listdir(neg_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(neg_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='gb18030') as f:
                        content = f.read()
                        cleaned_words = clean_text(content, stopwords)
                        if cleaned_words:  # 确保清洗后的文本不为空
                            texts.append(cleaned_words)
                            labels.append(0)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    
    # 加载正面评论 (标签为1)
    pos_dir = os.path.join(data_dir, 'pos')
    if os.path.exists(pos_dir):
        for file_name in os.listdir(pos_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(pos_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='gb18030') as f:
                        content = f.read()
                        cleaned_words = clean_text(content, stopwords)
                        if cleaned_words:  # 确保清洗后的文本不为空
                            texts.append(cleaned_words)
                            labels.append(1)
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    
    return texts, labels

if __name__ == "__main__":
    # 测试数据清洗模块
    stopwords = load_stopwords('stopwords.txt')
    texts, labels = load_and_clean_data('data', stopwords)
    
    print(f"加载了 {len(texts)} 条评论数据")
    if texts:
        print(f"示例清洗后的文本: {texts[0]}")
        print(f"对应的标签: {labels[0]}")    