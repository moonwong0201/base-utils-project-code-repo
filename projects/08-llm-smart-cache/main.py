import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import asyncio

from llm_cache import EmbeddingsCache, SemanticCache, SemanticMessageHistory, SemanticRouter
import config

import logging
import sys

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 确保输出到stdout
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI 向量检索与智能缓存服务")

# 启动时加载，只加载一次
print("加载Embedding模型...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)


def get_embedding(texts: Union[str, List[str]]) -> np.ndarray:
    """同步版本，供非异步代码调用"""
    # 强制转列表
    if isinstance(texts, str):
        texts = [texts]

    # 批量查询缓存
    cached_embeddings, hit_status = embed_cache.call(texts)

    # 处理缓存结果
    if cached_embeddings is None:
        cached_embeddings = [None] * len(texts)
    if isinstance(hit_status, bool):
        hit_status = [hit_status] * len(texts)

    # 检查是否全部命中
    if all(hit_status):
        result = np.array([e for e in cached_embeddings if e is not None])
        print(f"[Embedding] 全部命中缓存，shape: {result.shape}")
        return result  # 直接返回，保持2D

    # 计算未命中的
    miss_texts = [text for text, hit in zip(texts, hit_status) if not hit]
    print(f"[Embedding] 未命中: {miss_texts}")

    miss_embeddings = embedding_model.encode(
        miss_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # 存储新计算的
    embed_cache.store(miss_texts, miss_embeddings)

    # 合并结果（保持顺序）
    final_embeddings = []
    miss_idx = 0
    for i, hit in enumerate(hit_status):
        if hit:
            final_embeddings.append(cached_embeddings[i])
        else:
            final_embeddings.append(miss_embeddings[miss_idx])
            miss_idx += 1

    result = np.array(final_embeddings)

    # 确保2D
    if len(result.shape) == 1:
        result = result.reshape(1, -1)

    print(f"[Embedding] 返回shape: {result.shape}, dtype: {result.dtype}")
    return result


# 初始化4个核心模块
embed_cache = EmbeddingsCache(
    name="embedding_cache",
    redis_url=config.REDIS_URL,
    redis_port=config.REDIS_PORT,
    redis_password=config.REDIS_PASSWORD
)

semantic_cache = SemanticCache(
    name="semantic_cache",
    embedding_method=get_embedding,  # 适配同步接口
    redis_url=config.REDIS_URL,
    redis_port=config.REDIS_PORT,
    redis_password=config.REDIS_PASSWORD
)

router = SemanticRouter(
    name="router",
    embedding_method=get_embedding,
    redis_url=config.REDIS_URL,
    redis_port=config.REDIS_PORT,
    redis_password=config.REDIS_PASSWORD
)

# OpenAI客户端
openai_client = openai.AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)


# ========== 请求/响应模型 ==========
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    route: Optional[str] = None  # 命中了哪个路由
    from_cache: bool = False  # 是否来自缓存
    embedding_hit: bool = False  # 标记embedding是否来自缓存


class AddRouteRequest(BaseModel):
    target: str
    questions: List[str]


# ========== 接口 ==========

@app.post("/router/add")
def add_route(req: AddRouteRequest):
    """添加路由规则"""
    router.add_route(req.questions, req.target)
    return {"target": req.target, "questions": len(req.questions)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主流程：路由 → 查缓存 → 调LLM → 存历史
    """

    # 1. 意图路由
    route = router.route(req.message)

    # 2. 查语义缓存（相似问题是否问过）
    cached_answer = semantic_cache.call(req.message)

    if cached_answer and len(cached_answer) > 0 and cached_answer[0]:
        return ChatResponse(
            response=cached_answer[0],
            route=route,
            from_cache=True,
            embedding_hit=True
        )


    # 3. 没缓存，调用OpenAI
    # 先取历史记录
    history = SemanticMessageHistory(
        name=req.session_id,
        redis_url=config.REDIS_URL,
        redis_port=config.REDIS_PORT
    )
    past_msgs = history.get_recent(top_k=5)

    # 构造消息
    messages = [{"role": "system", "content": "请尽量简单回答。"}]
    for msg in past_msgs:
        role = "assistant" if msg.get("role") == "llm" else msg.get("role", "user")
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    # 调模型
    response = await openai_client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages
    )
    answer = response.choices[0].message.content

    semantic_cache.store(req.message, answer)

    history.add_message([
        {"role": "user", "content": req.message},
        {"role": "llm", "content": answer}
    ])

    history.delete_history(top_k=20)
    return ChatResponse(
        response=answer,
        route=route,
        from_cache=False,
        embedding_hit=False  # 重新调用了LLM，说明未命中缓存
    )


@app.get("/history/sessions")
def list_sessions():
    """列出所有有历史记录的 session_id"""
    import redis
    r = redis.Redis(
        host=config.REDIS_URL,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )
    # 查找所有 semantic_history:* 的 key
    keys = r.keys("semantic_history:*")
    # 提取 session_id
    sessions = [k.replace("semantic_history:", "") for k in keys]
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """查某个会话的历史"""
    history = SemanticMessageHistory(
        name=session_id,
        redis_url=config.REDIS_URL,
        redis_port=config.REDIS_PORT
    )
    return {"session_id": session_id, "messages": history.get_history()}


@app.post("/history/search/keyword")
def search_history_keyword(session_id: str, keyword: str):
    """关键词搜索历史"""
    history = SemanticMessageHistory(
        name=session_id,
        redis_url=config.REDIS_URL,
        redis_port=config.REDIS_PORT
    )
    all_msgs = history.get_history()
    # 简单关键词匹配
    results = [m for m in all_msgs if keyword in m.get("content", "")]
    return {"keyword": keyword, "results": results}


@app.post("/history/clear")
def clear_history(session_ids: List[str]):
    """清空指定会话的所有历史记录"""
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    deleted_count = 0
    for id in session_ids:
        history = SemanticMessageHistory(
            name=id,
            redis_url=config.REDIS_URL,
            redis_port=config.REDIS_PORT
        )
        deleted = history.clear_history()
        deleted_count += 1

    return {
        "session_id": session_ids,
        "deleted_count": deleted_count
    }


@app.post("/cache/clear")
def clear_cache(clear_router: bool = False):
    """清空所有缓存（包括所有“提问-问答”、Faiss索引数据、embedding 缓存）"""
    semantic_cache.clear_cache()
    embed_cache.clear_all()
    if clear_router:
        router.clear_cache()  # 可选：清路由
        return {"cleared": ["semantic", "embedding", "router"]}

    return {"cleared": ["semantic", "embedding"]}  # 默认保留路由


@app.get("/")
def root():
    """根路径自动跳转到 API 文档"""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    # 启动FastAPI服务（默认端口8000）
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,
        reload=True
    )
