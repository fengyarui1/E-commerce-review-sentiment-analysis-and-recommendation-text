import jieba
import jieba.analyse
import json
import websocket
import _thread as thread
import base64
import datetime
import hashlib
import hmac
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import ssl
import sys

# 配置信息（从原程序中获取）
appid = "88cbd0fa"
api_secret = "YWE5YWNjNWIxNjFmMTYzMGM4OTAwNWI3"
api_key = "b182ab71b9424eef3b1a78d574aec901"
domain = "x1"
Spark_url = "wss://spark-api.xf-yun.com/v1/x1"

# 初始化jieba分词并加载自定义词典（如果有）
jieba.initialize()

answer = ""


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    def create_url(self):
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }

        url = self.Spark_url + '?' + urlencode(v)
        return url


def on_error(ws, error):
    print("")


def on_close(ws, one, two):
    print(" ")


def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, domain=ws.domain, question=ws.question))
    ws.send(data)


def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        text = choices['text'][0]
        if 'content' in text and text['content']:
            content = text["content"]
            print(content, end="")
            global answer
            answer += content
        if status == 2:
            ws.close()


def gen_params(appid, domain, question):
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234",
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 1.2,
                "max_tokens": 32768
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data


def main(question):
    global answer
    answer = ""  # 每次调用前清空结果
    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return answer


def extract_keywords(text, topK=5, withWeight=False, allowPOS=()):
    """使用TextRank算法提取关键词"""
    keywords = jieba.analyse.textrank(text, topK=topK, withWeight=withWeight, allowPOS=allowPOS)
    return keywords


def generate_request(keywords):
    """生成发送给星火大模型的请求内容"""
    keyword_str = "、".join(keywords)
    return [{"role": "user", "content": f"帮我写一段简短的好评，关键字为：{keyword_str}，60-80字左右"}]


if __name__ == '__main__':
    while(1):
        # 获取用户输入的文本
        print("请输入文本：")
        input_text = input()
        # 使用TextRank算法提取关键词
        keywords = extract_keywords(input_text, topK=5, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l'))
        print(f"提取的关键词：\n{keywords}")
        # 生成请求内容
        question = generate_request(keywords)
        # 发送请求给星火大模型
        print("生成的好评：")
        result = main(question)