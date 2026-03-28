import tkinter as tk
from tkinter import messagebox
from sentiment_predictor import SentimentPredictor
from comment_generation import extract_keywords, generate_request, main

# 创建情感预测器实例
predictor = SentimentPredictor(neutral_threshold=0.2)

def predict_sentiment():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("警告", "请输入要分析的文本！")
        return
    result = predictor.predict(text)
    sentiment = result['sentiment']
    confidence = result['confidence']
    neg_prob = result['probabilities']['negative']
    pos_prob = result['probabilities']['positive']

    result_text = f"情感: {sentiment}\n置信度: {confidence:.4f}\n概率分布: 负面 {neg_prob:.4f}, 正面 {pos_prob:.4f}"
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, result_text)

def generate_comment():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("警告", "请输入要生成好评的文本！")
        return
    # 使用TextRank算法提取关键词
    keywords = extract_keywords(text, topK=5, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l'))
    # 生成请求内容
    question = generate_request(keywords)
    # 发送请求给星火大模型
    result = main(question)
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, result)

# 创建主窗口
root = tk.Tk()
root.title("电商评论情感分析与推荐文本生成系统")

# 获取屏幕宽度和高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 设置窗口宽度和高度
window_width = 455
window_height = 400

# 计算窗口在屏幕中央的坐标
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

# 设置窗口位置
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 创建一个框架来包含输入框和输出框
input_output_frame = tk.Frame(root)
input_output_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# 创建文本输入框
text_input = tk.Text(input_output_frame, height=20, width=30)
text_input.pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)

# 创建结果输出框
result_output = tk.Text(input_output_frame, height=20, width=30)
result_output.pack(side=tk.LEFT, padx=(10, 10), fill=tk.Y)

# 创建按钮框架
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)

# 创建预测按钮
predict_button = tk.Button(button_frame, text="预测情感", command=predict_sentiment)
predict_button.pack(side=tk.LEFT, padx=5)

# 创建生成按钮
generate_button = tk.Button(button_frame, text="生成好评", command=generate_comment)
generate_button.pack(side=tk.LEFT, padx=5)

# 运行主循环
root.mainloop()