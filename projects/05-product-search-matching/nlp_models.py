from config import DEVICE, BGE_MODEL, CLIP_MODEL
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.preprocessing import normalize

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bge_model = SentenceTransformer(
    BGE_MODEL, device=DEVICE
)

clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL)
clip_model.to(DEVICE)
clip_processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL)


def get_text_bge_features(texts):
    """
    接收文本列表，生成对应的 BGE 文本向量（384 维）
    :param texts: 文本列表
    :return: 二维数组（形状为[文本数量, 384]），每个元素对应一条文本的 384 维向量
    """
    logger.info(f"BGE 推理: {texts[0][:20]}...")
    try:
        if isinstance(texts, str):
            texts = [texts]
        # 2. 前置校验：确保是列表且全为字符串
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError(f"输入必须是字符串或字符串列表，当前类型：{type(texts)}")

        embeddings = bge_model.encode(texts, normalize_embeddings=True)
        return embeddings
    except Exception as e:
        logger.error(f"BGE 推理失败: {str(e)}")
        return None


def get_clip_image_features(images):
    """
    接收图片对象列表，生成对应的 CLIP 图片向量（512 维），用于图片相关的检索（A1 图搜图、A3 图搜文本）
    :param images: PIL.Image 对象列表
    :return: 二维数组（形状为[图片数量, 512]），每个元素对应一张图片的 512 维向量
    """
    try:
        if isinstance(images, Image.Image):  # 单张图片→转列表
            images = [images]

        # 校验：确保列表内都是合法的PIL图片（提前拦截错误）
        if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
            raise ValueError(f"输入必须是PIL.Image或PIL.Image列表，当前类型：{type(images)}")

        inputs = clip_processor(images=images, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        image_features = clip_model.get_image_features(**inputs)
        return normalize(image_features.detach().cpu().numpy(), axis=1)
    except Exception as e:
        logger.error(f"CLIP 图片推理失败: {str(e)}")
        return None


def get_clip_text_features(texts):
    """
    接收文本列表，生成对应的 CLIP 文本向量（512 维），用于跨模态检索（A4 文本搜图片、A3 图搜文本的比对基准）
    :param texts: 文本列表
    :return:
    """
    try:
        if isinstance(texts, str):
            texts = [texts]
        # 2. 前置校验：确保是列表且全为字符串（提前拦截错误）
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError(f"输入必须是字符串或字符串列表，当前类型：{type(texts)}")

        inputs = clip_processor(text=texts, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        text_features = clip_model.get_text_features(**inputs)
        return normalize(text_features.detach().cpu().numpy(), axis=1)
    except Exception as e:
        logger.error(f"CLIP 文本推理失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 计算时间
    import time

    start_time = time.time()
    get_text_bge_features(["你好"])
    end_time = time.time()
    print(f"bge模型计算时间: {end_time - start_time} 秒")

    start_time = time.time()
    get_clip_image_features([Image.open("pokemon.jpeg").resize((224, 224))])
    end_time = time.time()
    print(f"clip模型计算时间: {end_time - start_time} 秒")

    start_time = time.time()
    get_clip_text_features(["你好"])
    end_time = time.time()
    print(f"clip模型计算时间: {end_time - start_time} 秒")
    
