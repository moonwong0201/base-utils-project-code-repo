# python自带库
import time
import traceback
from typing import Union, Dict
from datetime import datetime

# 第三方库
import openai
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyResponse   # 自定义的响应数据结构
from data_schema import TextClassifyRequest    # 自定义的请求数据结构
from model.prompt import model_for_gpt         # GPT模型的分类函数
from model.bert import model_for_bert          # BERT模型的分类函数
from model.regex_rule import model_for_regex   # 正则表达式分类函数
from model.tfidf_ml import model_for_tfidf     # TF-IDF+机器学习的分类函数
from logger import logger   # 自定义的日志工具，用于记录程序运行信息

# 创建了一个 FastAPI 应用实例（app），后续所有的 API 接口都将基于这个实例定义
app = FastAPI(title="意图识别助手", version="1.0")


@app.get("/health", tags=["健康检查"])
def health_check() -> Dict:
    """
    基础健康检查接口
    """
    return {
        "status": "healthy",
        "service": "intent-classify",
        "version": "1.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "服务正常运行"
    }


# 定义一个POST类型的API接口，路径为/v1/text-cls/regex
@app.post("/v1/text-cls/regex")
def regex_classify(req: TextClassifyRequest) -> TextClassifyResponse:  # 接口函数，接收请求数据，返回响应数据
    """
    利用正则表达式进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    # 初始化响应对象（按预设格式准备返回数据）
    response = TextClassifyResponse(
        request_id=req.request_id,    # 保留请求ID（方便追踪请求）
        request_text=req.request_text,   # 保留原始请求文本
        classify_result="",  # 分类结果
        classify_time=0,  # 分类耗时
        error_msg=""  # 错误信息
    )

    logger.info(f"{req.request_id} {req.request_text}")  # 打印请求ID和请求文本

    # 核心：调用正则表达式模型进行分类
    try:
        # 调用正则分类函数，传入请求文本，得到分类结果
        response.classify_result = model_for_regex(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/tfidf")
def tfidf_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用TFIDF进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    logger.info(f"Get requst: {req.json()}")

    try:
        response.classify_result = model_for_tfidf(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用BERT进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    # info 日志
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/gpt")
def gpt_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用大语言模型进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    try:
        response.classify_result = model_for_gpt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


if __name__ == "__main__":
    import uvicorn
    # 启动服务：适配生产环境，支持多线程+端口配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 服务端口
        reload=False     # 生产环境关闭热重载
    )