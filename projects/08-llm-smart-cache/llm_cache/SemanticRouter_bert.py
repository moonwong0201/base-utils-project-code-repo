"""
基于语义向量的意图路由模块（基于微调Bert模型）
是大模型应用的 "意图识别 / 业务分流入口"
"""

import torch
from typing import Optional, List, Union, Any, Dict, Callable
import redis
import os
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertForSequenceClassification
import numpy as np
from config import BERT_MODEL_PKL_PATH, BERT_MODEL_PERTRAINED_PATH, LABEL_MAP_PATH, DEVICE
import logging

logger = logging.getLogger("SemanticRouter")


class SemanticRouter_bert:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.3
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.routes = {}
        self.idx_to_target = {}

        self.label2id, self.id2label = self._load_label_map()
        self.tokenizer, self.model = self._load_bert_model()

        self.index_file = f"{self.name}_routes.index"
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)

                routes_data = self.redis.get(f"{self.name}_routes_config")
                if routes_data:
                    self.routes = json.loads(routes_data)
                    for target in self.routes:
                        if self.routes[target]['embeddings'] is not None:
                            self.routes[target]['embeddings'] = np.array(self.routes[target]['embeddings'])
                    self._rebuild_idx_mapping()
            except Exception as e:
                self.index = None
        else:
            self.index = None

    def _rebuild_idx_mapping(self):
        """重建索引映射"""
        current_idx = 0
        for target, route_data in self.routes.items():
            count = len(route_data['questions'])
            for i in range(count):
                self.idx_to_target[current_idx + i] = target
            current_idx += count

    def add_route(self, questions: List[str], target: str):
        """添加路由规则"""
        start_idx = self.index.ntotal if self.index else 0

        if target not in self.routes:
            self.routes[target] = {'questions': [], 'embeddings': None}

        # 去重添加
        for q in questions:
            if q not in self.routes[target]['questions']:
                self.routes[target]['questions'].append(q)

        # 获取embedding并确保2D
        embeddings = self.embedding_method(self.routes[target]['questions'])
        if embeddings is None or len(embeddings) == 0:
            logger.error(f"生成embedding失败，目标意图：{target}，问题列表：{questions}")
            return

        # 强制2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.routes[target]['embeddings'] = embeddings

        # 初始化或添加索引
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings.astype(np.float32))

        # 保存配置
        self.redis.setex(
            f"{self.name}_routes_config",
            self.ttl,
            json.dumps(self.routes, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        )
        faiss.write_index(self.index, self.index_file)

        # 建立映射
        for i in range(len(questions)):
            self.idx_to_target[start_idx + i] = target

        # 缓存问题
        for q in questions:
            self.redis.setex(f"{self.name}_route_cache:{q}", self.ttl, target)

    def _load_label_map(self):
        if not os.path.exists(LABEL_MAP_PATH):
            raise Exception(f"标签映射表不存在，请先训练BERT模型！")

        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        label2id = {k: int(v) for k, v in label_map["label2id"].items()}
        id2label = {int(k): v for k, v in label_map["id2label"].items()}
        return label2id, id2label

    def _load_bert_model(self):
        """加载BERT模型（带异常处理 + 半精度优化）"""
        try:
            # 加载Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)

            # 加载分类模型（动态设置类别数，避免硬编码）
            num_labels = len(self.label2id)
            model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL_PERTRAINED_PATH,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,  # 兼容微调时类别数变化的情况
                label2id=self.label2id,  # 传入训练时的标签映射
                id2label=self.id2label
            )

            # 加载微调权重（支持CPU/GPU）
            model.load_state_dict(
                torch.load(BERT_MODEL_PKL_PATH, map_location=DEVICE),
                strict=False  # 兼容权重少量不匹配的情况
            )

            # 模型移到设备 + 半精度优化（减少显存占用，提升速度）
            model = model.to(DEVICE)
            model.eval()
            return tokenizer, model

        except FileNotFoundError as e:
            raise Exception(f"模型文件不存在：{e}")
        except Exception as e:
            raise Exception(f"模型加载失败：{str(e)}")

    def model_for_bert(self, question):
        try:
            encoding = self.tokenizer(
                question,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)

            logits = outputs.logits
            pred_label_id = torch.argmax(logits, dim=1).item()  # 取分数最高的类别ID
            classify_result = self.id2label[pred_label_id]
            return classify_result
        except Exception as e:
            logger.error(f"BERT意图预测失败：{str(e)}，问题：{question}")
            return None  # 预测失败返回None，触发全局回退

    def route(self, question: str):
        """匹配路由"""
        # 查精确缓存
        cached_result = self.redis.get(f"{self.name}_route_cache:{question}")
        if cached_result:
            return cached_result.decode()

        predicted_intent = self.model_for_bert(question)


        self.redis.setex(f"{self.name}_route_cache:{question}", self.ttl, predicted_intent)
        return predicted_intent


    def clear_cache(self):
        """清空缓存"""

        self.redis.delete(f"{self.name}_routes_config")
        keys = self.redis.keys(f"{self.name}_route_cache:*")
        if keys:
            self.redis.delete(*keys)

        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        self.index = None
        self.routes = {}
        self.idx_to_target = {}


if __name__ == "__main__":
    model = SentenceTransformer(
        "/bge-small-zh-v1.5")
    def get_embedding(text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        embeddings = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    router = SemanticRouter_bert(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=0.5
    )
    router.clear_cache()

    router.add_route(
        questions=["还有双鸭山到淮阴的汽车票吗", "查询北京飞桂林的飞机", "自驾游去深圳都经过那些地方啊"],
        target="Travel-Query"
    )

    router.add_route(
        questions=["随便播放一首专辑阁楼里的佛里的歌", "播放钢琴曲命运交响曲", "我一定要单曲循环赵雷的我们的时光这首流行", "来放一个德语歌曲给我吧"],
        target="Music-Play"
    )

    print("还有双鸭山到淮阴的汽车票吗: ", router.route("还有双鸭山到淮阴的汽车票吗"))
    print("播放钢琴曲命运交响曲: ", router.route("播放钢琴曲命运交响曲"))
    print("请播放带戏曲唱腔的专辑赤伶:", router.route("请播放带戏曲唱腔的专辑赤伶"))
    print("南昌明天会降雨吗:", router.route("南昌明天会降雨吗"))
