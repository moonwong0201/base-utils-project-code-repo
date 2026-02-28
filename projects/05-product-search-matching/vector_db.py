import traceback

from pymilvus import MilvusClient
from PIL import Image
from nlp_models import get_clip_image_features, get_clip_text_features, get_text_bge_features, bge_model

from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MilvusClient(
    uri="https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
)


def insert_product_(image_path: str, title: str):
    # 函数功能：将商品的图片和标题转换为向量，插入到Milvus数据库
    # 参数：
    #   image_path: 商品图片的本地路径（如./product_images/xxx.jpg）
    #   title: 商品标题文本
    # 返回值：(是否成功, Milvus中的主键ID)

    # 提取特征
    image_clip_features = None
    title_bge_embedding = None
    title_clip_embedding = None

    try:
        # 步骤1：加载图片并生成CLIP模型的图片向量
        image = Image.open(image_path)
        # 调用nlp_models的get_clip_image_features函数，传入图片列表，获取第1个图片的向量
        features = get_clip_image_features([image])
        if features is not None:
            image_clip_features = list(features[0])
    except Exception as e:
        # 若图片向量生成失败（如图片损坏、模型调用出错），打印错误堆栈
        traceback.print_exc()
        # 生成一个全0的512维向量作为默认值（避免插入失败，后续可通过业务逻辑处理无效数据）
        image_clip_features = [0] * 512

    # logger.info(f"图片向量:{image_clip_features}")
    # 提取特征
    try:
        # 步骤2：生成标题的BGE模型文本向量（384维）
        # 调用nlp_models的get_text_bge_features函数，传入标题列表，获取第1个标题的向量
        features = get_text_bge_features([title])
        if features is not None:
            title_bge_embedding = list(features[0])
    except Exception as e:
        # 若BGE向量生成失败，打印错误并生成全0的384维向量
        traceback.print_exc()
        title_bge_embedding = [0] * 384
    # logger.info(f"bge文本向量:{title_bge_embedding}")
    # 提取特征
    try:
        # 步骤3：生成标题的CLIP模型文本向量（512维）
        # 调用nlp_models的get_clip_text_features函数，传入标题列表，获取第1个标题的向量
        features = get_clip_text_features([title])
        if features is not None:
            title_clip_embedding = list(features[0])
    except Exception as e:
        # 若CLIP文本向量生成失败，打印错误并生成全0的512维向量
        traceback.print_exc()
        title_clip_embedding = [0] * 512
    # logger.info(f"clip文本向量:{title_clip_embedding}")
    # 如果全部失败，返回失败
    if all(x is None for x in [image_clip_features, title_bge_embedding, title_clip_embedding]):
        return False, None

    try:
        # 步骤4：组装要插入Milvus的数据
        data = [
            {
                "image_clip_vector": image_clip_features,  # CLIP图片向量（512维）
                "text_bge_vector": title_bge_embedding,  # BGE文本向量（384维）
                "text_clip_vector": title_clip_embedding,  # CLIP文本向量（512维）
                "image_path": image_path,  # 图片路径
                "title": title  # 标题文本
            }
        ]
        # 调用Milvus客户端的insert方法，插入数据到名为"product_new"的集合（表）
        insert_result = client.insert(
            collection_name="product_new",  # 目标集合名称
            data=data  # 要插入的数据列表（一次可插入多条，此处仅一条）
        )
        # 提取插入数据的主键ID（Milvus自动生成，唯一标识该条向量记录）
        milvus_primary_key = insert_result["ids"][0]  # "ids" 字段对应「所有插入数据的主键列表」
        return True, milvus_primary_key  # 返回成功标识和主键ID

    except Exception as e:
        # 若插入过程出错（如网络问题、集合不存在），打印错误并返回失败`
        traceback.print_exc()
        return False, None


def delete_product_(ids: List[int]):
    try:
        client.delete(
            collection_name="product_new",
            ids=ids
        )
        return True
    except Exception as e:
        traceback.print_exc()
        return False


def search_product_(
        image: Optional[Image.Image] = None,
        title: Optional[str] = None,
        task: str = "text2text",
        top_k: int = 10
):
    try:
        if image is None and title is None:
            return False, None

        if image is not None:
            image_clip_features = list(get_clip_image_features([image])[0])
        else:
            image_clip_features = None

        if title is not None:
            title_bge_embedding = list(get_clip_text_features([title])[0])
        else:
            title_bge_embedding = None

        if image_clip_features:
            data = [image_clip_features]
        else:
            data = [title_bge_embedding]

        if task in ["text2text"]:
            anns_field = "text_clip_vector"
        elif task in ["text2image"]:
            anns_field = "image_clip_vector"
        elif task in ["image2text"]:
            anns_field = "text_clip_vector"
        elif task in ["image2image"]:
            anns_field = "image_clip_vector"
        else:
            return False, None

        results = client.search(
            collection_name="product_new",
            anns_field=anns_field,
            data=data,
            limit=top_k,
            search_params={
                "metric_type": "COSINE"
            }
        )
        return True, results

    except Exception as e:
        traceback.print_exc()
        return False, None

