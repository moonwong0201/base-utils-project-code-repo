import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

from llm_cache import EmbeddingsCache, SemanticCache, SemanticMessageHistory, SemanticRouter
import config

app = FastAPI(title="AI 向量检索与智能缓存服务")

# ========== 全局单例（启动时加载，只加载一次） ==========
print("加载Embedding模型...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)


def get_embedding(texts):
    """统一的embedding函数"""
    if isinstance(texts, str):
        texts = [texts]

    # 1. 查缓存
    cached = embed_cache.call(texts)
    if cached and all(c is not None for c in cached):
        return np.array(cached)

    # 2. 没缓存，调模型
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # 3. 存缓存
    embed_cache.store(texts, embeddings)

    return embeddings


# 初始化4个核心模块
embed_cache = EmbeddingsCache(
    name="embedding_cache",
    redis_url=config.REDIS_URL,
    redis_port=config.REDIS_PORT,
    redis_password=config.REDIS_PASSWORD
)

semantic_cache = SemanticCache(
    name="semantic_cache",
    embedding_method=get_embedding,
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
    # 1. 意图路由（判断是问候、退款还是其他）
    route = router.route(req.message)

    # 2. 查语义缓存（相似问题是否问过）
    cached_answer = semantic_cache.call(req.message)
    if cached_answer and cached_answer[0]:
        return ChatResponse(response=cached_answer[0], route=route, from_cache=True)

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
        model="qwen-max",
        messages=messages
    )
    answer = response.choices[0].message.content

    # 4. 存结果
    semantic_cache.store(req.message, answer)  # 存语义缓存
    # 存对话历史
    history.add_message([
        {"role": "user", "content": req.message},
        {"role": "llm", "content": answer}
    ])

    history.delete_history(top_k=20)
    return ChatResponse(response=answer, route=route, from_cache=False)


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
def clear_history(session_id: str):
    """清空指定会话的所有历史记录"""
    history = SemanticMessageHistory(
        name=session_id,
        redis_url=config.REDIS_URL,
        redis_port=config.REDIS_PORT
    )
    deleted = history.clear_history()
    return {
        "session_id": session_id,
        "deleted_count": deleted
    }


@app.post("/history/clear-all")
def clear_all_history():
    """清空所有会话的历史记录（慎用！）"""
    import redis
    r = redis.Redis(
        host=config.REDIS_URL,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )
    keys = r.keys("semantic_history:*")
    if keys:
        deleted = r.delete(*keys)
    else:
        deleted = 0
    return {"deleted_sessions": deleted, "keys": keys[:5]}  # 只显示前5个


class DeleteCacheRequest(BaseModel):
    texts: Union[List[str], str]


@app.post("/cache/delete")
def delete_cache(req: DeleteCacheRequest):
    """删除指定文本的 embedding 缓存"""
    deleted = embed_cache.delete(req.texts)
    return {
        "deleted_count": deleted
    }


@app.post("/cache/clear")
def clear_cache():
    """清空所有缓存（包括所有“提问-问答”、Faiss索引数据、embedding 缓存）"""
    semantic_cache.clear_cache()
    embed_cache.clear_all()
    return {"cleared": True}


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
        reload=True  # 开发模式：代码修改自动重启（生产环境关闭）
    )