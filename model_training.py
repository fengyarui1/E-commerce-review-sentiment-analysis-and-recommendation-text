import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from data_cleaning import load_stopwords, load_and_clean_data

def train_model(data_dir: str, model_path: str = 'sentiment_model.pkl') -> None:
    """训练情感分析模型并保存"""
    # 加载停用词
    stopwords = load_stopwords('stopwords.txt')

    # 加载和清洗数据
    print("加载和清洗数据中...")
    texts, labels = load_and_clean_data(data_dir, stopwords)

    if not texts:
        print("错误: 没有找到有效的训练数据")
        return

    # 将分词后的文本转换为字符串
    texts = [' '.join(words) for words in texts]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 创建模型管道
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    # 训练模型
    print("训练模型中...")
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"模型准确率: {accuracy:.4f}")
    print(f"模型F1分数: {f1:.4f}")

    # 保存模型
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")

    return accuracy, f1

if __name__ == "__main__":
    # 训练模型
    train_model('data', 'models/sentiment_model.pkl')