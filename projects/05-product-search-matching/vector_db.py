import traceback

from pymilvus import MilvusClient

from PIL import Image
from nlp_models import get_clip_image_features, get_clip_text_features, get_text_bge_features, bge_model

from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MilvusClient(
    uri="https://in03-3834be1737c78de.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="2cb0ff05416f60fdde42d2a3d37cc82c3a6c62df228e34b46950e034b6879d637d762ffc3cf2274c9a344a1aa444929b404e530d"
)

COLLECTION_NAME = "product_new"


def delete_collection(collection_name: str):
    """
    删除 Milvus 中的整个集合，包括所有向量和索引
    """
    try:
        client.drop_collection(collection_name)
        logger.info(f"Milvus 集合 {collection_name} 已删除")
        return True
    except Exception as e:
        logger.error(f"删除 Milvus 集合失败: {str(e)}")
        traceback.print_exc()
        return False


from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema


def create_collection():
    """
    重建Milvus的product_new集合（含向量维度、索引配置）
    适配CLIP(512维) + BGE(512维)双向量
    """

    try:
        # 1. 删除旧集合
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            logger.info(f"删除旧集合: {COLLECTION_NAME}")

        # 2. 定义字段
        fields = [
            # 主键
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # 三个向量字段
            FieldSchema(name="image_clip_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="text_bge_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="text_clip_vector", dtype=DataType.FLOAT_VECTOR, dim=512),

            # 标量字段
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        ]

        # 3. 创建schema
        schema = CollectionSchema(fields, description="多模态商品向量库")

        # 4. 创建集合
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
        )
        logger.info(f"创建集合成功: {COLLECTION_NAME}")

        for field_name in ["image_clip_vector", "text_bge_vector", "text_clip_vector"]:
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=field_name,
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 2048}
            )

            client.create_index(
                collection_name=COLLECTION_NAME,
                index_params=index_params
            )
            logger.info(f"{field_name} 索引创建成功")

        # 6. 加载集合
        client.load_collection(COLLECTION_NAME)
        logger.info(f"集合加载完成，可使用")

        return True

    except Exception as e:
        logger.error(f"重建失败: {e}")
        traceback.print_exc()
        return False


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
        title_bge_embedding = [0] * 512
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
                "image_clip_vector": image_clip_features,  # CLIP图片向量
                "text_bge_vector": title_bge_embedding,  # BGE文本向量
                "text_clip_vector": title_clip_embedding,  # CLIP文本向量
                "image_path": image_path,  # 图片路径
                "title": title  # 标题文本
            }
        ]
        # 调用Milvus客户端的insert方法，插入数据到名为"product_new"的集合
        insert_result = client.insert(
            collection_name=COLLECTION_NAME,  # 目标集合名称
            data=data  # 要插入的数据列表
        )
        # 提取插入数据的主键ID（Milvus自动生成，唯一标识该条向量记录）
        milvus_primary_key = insert_result["ids"][0]  
        logger.info(f"插入成功，主键ID: {milvus_primary_key}")
        logger.info(f"当前索引列表: {client.list_indexes(COLLECTION_NAME)}")
        return True, milvus_primary_key  # 返回成功标识和主键ID

    except Exception as e:
        # 若插入过程出错（如网络问题、集合不存在），打印错误并返回失败`
        traceback.print_exc()
        return False, None


def insert_products_batch(image_paths: List[str], titles: List[str], batch_size: int = 10):
    """
    批量插入商品，减少 Milvus 网络往返
    """
    try:
        all_data = []

        # 1. 批量推理（并行）
        for image_path, title in zip(image_paths, titles):
            # 特征提取（保持原有逻辑）
            image_clip_features = None
            title_bge_embedding = None
            title_clip_embedding = None

            try:
                image = Image.open(image_path)
                features = get_clip_image_features([image])
                if features is not None:
                    image_clip_features = list(features[0])
            except:
                image_clip_features = [0] * 512

            try:
                features = get_text_bge_features([title])
                if features is not None:
                    title_bge_embedding = list(features[0])
            except:
                title_bge_embedding = [0] * 512

            try:
                features = get_clip_text_features([title])
                if features is not None:
                    title_clip_embedding = list(features[0])
            except:
                title_clip_embedding = [0] * 512

            all_data.append({
                "image_clip_vector": image_clip_features,
                "text_bge_vector": title_bge_embedding,
                "text_clip_vector": title_clip_embedding,
                "image_path": image_path,
                "title": title
            })

        # 2. 批量插入 Milvus
        insert_result = client.insert(
            collection_name=COLLECTION_NAME,
            data=all_data  # 批量数据
        )

        # 返回所有 ID
        ids = insert_result["ids"]
        return True, ids

    except Exception as e:
        logger.error(f"批量插入失败: {e}")
        return False, []


def delete_product_(ids: List[int]):
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            ids=ids
        )
        return True
    except Exception as e:
        logger.info(f"商品 {ids} 删除失败：{e}")
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
            title_bge_embedding = list(get_text_bge_features([title])[0])
        else:
            title_bge_embedding = None

        if task in ["text2text"]:
            data = [title_bge_embedding]
            anns_field = "text_bge_vector"
        elif task in ["text2image"]:
            data = [title_bge_embedding]
            anns_field = "image_clip_vector"
        elif task in ["image2text"]:
            data = [image_clip_features]
            anns_field = "text_clip_vector"
        elif task in ["image2image"]:
            data = [image_clip_features]
            anns_field = "image_clip_vector"
        else:
            return False, None

        results = client.search(
            collection_name=COLLECTION_NAME,
            anns_field=anns_field,
            data=data,
            limit=top_k,
            search_params={
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
        )
        return True, results

    except Exception as e:
        traceback.print_exc()
        return False, None

