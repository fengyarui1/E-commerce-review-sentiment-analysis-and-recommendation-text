import joblib
import jieba
from typing import List, Dict, Tuple
from data_cleaning import load_stopwords, clean_text


class SentimentPredictor:
    def __init__(self, model_path: str = 'models/sentiment_model.pkl',
                 stopwords_path: str = 'stopwords.txt',
                 neutral_threshold: float = 0.2):
        """初始化情感分析预测器"""
        # 加载模型
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件 '{model_path}' 未找到，请先训练模型")

        # 加载停用词
        self.stopwords = load_stopwords(stopwords_path)

        # 中性情感阈值
        self.neutral_threshold = neutral_threshold

        # 情感标签映射
        self.sentiment_mapping = {
            0: "负面",
            1: "正面",
            2: "中性"
        }

    def predict(self, text: str) -> Dict:
        """预测文本的情感极性"""
        # 清洗文本
        cleaned_words = clean_text(text, self.stopwords)

        if not cleaned_words:
            return {
                'sentiment': '无法判断',
                'confidence': 0.0,
                'words': []
            }

        # 将分词后的文本转换为字符串
        cleaned_text = ' '.join(cleaned_words)

        # 预测情感概率
        sentiment_probs = self.model.predict_proba([cleaned_text])[0]

        # 获取正面和负面情感的概率
        pos_prob = sentiment_probs[1]
        neg_prob = sentiment_probs[0]

        # 判断情感类别
        if abs(pos_prob - neg_prob) < self.neutral_threshold:
            # 中性情感：正反概率接近
            sentiment_class = 2
            confidence = (pos_prob + neg_prob) / 2  # 中性情感的置信度取平均
        elif pos_prob > neg_prob:
            # 正面情感
            sentiment_class = 1
            confidence = pos_prob
        else:
            # 负面情感
            sentiment_class = 0
            confidence = neg_prob

        # 获取情感标签
        sentiment_label = self.sentiment_mapping.get(sentiment_class, "未知")

        return {
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'words': cleaned_words,
            'probabilities': {
                'negative': float(neg_prob),
                'positive': float(pos_prob)
            }
        }

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """批量预测文本情感极性"""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # 测试情感预测器
    try:
        # 设置中性阈值为0.2（可根据需要调整）
        predictor = SentimentPredictor(neutral_threshold=0.2)

        # 测试文本
        test_texts = [
            "这个产品真的很棒，我非常喜欢！",
            "这款产品还可以。",
            "太糟糕了，体验非常差，差评！",
            "很帮的一次体验，电影非常有趣，而且还送了饮料，下次还会再来"
        ]

        # 预测结果
        results = predictor.batch_predict(test_texts)

        # 打印结果
        for text, result in zip(test_texts, results):
            print(f"文本: {text}")
            print(f"情感: {result['sentiment']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(
                f"概率分布: 负面 {result['probabilities']['negative']:.4f}, 正面 {result['probabilities']['positive']:.4f}")
            print("-" * 50)

    except Exception as e:
        print(f"预测过程中发生错误: {e}")    