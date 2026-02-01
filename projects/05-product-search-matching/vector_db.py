import traceback

from pymilvus import MilvusClient
from PIL import Image
from nlp_models import get_clip_image_features, get_clip_text_features, get_text_bge_features, bge_model

from typing import List, Optional

client = MilvusClient(
    uri="https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
)


def insert_product_(image_path: str, title: str):
    try:
        image = Image.open(image_path)
        image_clip_features = list(get_clip_image_features([image])[0])
    except Exception as e:
        traceback.print_exc()
        image_clip_features = [0] * 512

    try:
        title_bge_embedding = list(get_text_bge_features([title])[0])
    except Exception as e:
        traceback.print_exc()
        title_bge_embedding = [0] * 384

    try:
        title_clip_embedding = list(get_clip_image_features([title])[0])
    except Exception as e:
        traceback.print_exc()
        title_clip_embedding = [0] * 512

    try:
        data = [
            {
                "image_clip_vector": image_clip_features,
                "text_bge_vector": title_bge_embedding,
                "text_clip_vector": title_clip_embedding,
                "image_path": image_path,
                "title": title
            }
        ]
        insert_result = client.insert(
            collection_name="product_new",
            data=data
        )

        milvus_primary_key = insert_result["ids"][0]
        return True, milvus_primary_key

    except Exception as e:
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

