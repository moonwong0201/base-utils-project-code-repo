"""
搭建了一个 FastAPI Web 服务，核心实现了两大模块功能，完全覆盖项目的 A1-A4 需求（A5 文生图仅预留接口）：
· 商品基础管理接口：实现商品的创建、查询列表、查询详情、更新标题 / 图片、删除，保证商品数据在 SQL 数据库、Milvus、本地图片目录的同步。
· 跨模态检索接口：实现文本搜文本、文本搜图片、图片搜文本、图片搜图四种检索模式，返回按相似度排序的结果。
· 附加功能：服务健康检查接口，用于监控服务运行状态。
"""
import base64
import os.path
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi import Depends, Query
import time
import uuid
import traceback
from PIL import Image
from io import BytesIO

from concurrent.futures import ThreadPoolExecutor
import asyncio

from sqlalchemy import create_engine
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker

from orm_models import create_tables, Product

from vector_db import insert_product_, delete_product_, search_product_
import vector_db
from data_models import BasicResponse, BasicRequest, SearchRequest

from config import DATABASE_URL, IMAGE
from functools import partial

# 全局线程池
executor = ThreadPoolExecutor(max_workers=4)

start_time = time.time()

SQLALCHEMY_DATABASE_URL = DATABASE_URL
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,  # 关闭自动提交
    bind=engine  # 所有会话的数据库操作，都通过这个 engine 执行
)

create_tables(engine)

app = FastAPI(
    title="多模态商品管理系统"
)

IMAGE_DIR = IMAGE
os.makedirs(IMAGE_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health_check():
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "service": "Product Search and Matching Service",
        "uptime": f"{uptime: .2f} second",
        # 调用 Milvus 客户端的list_collections()方法，返回所有集合列表，若能正常返回则说明 Milvus 连接正常
        "milvus": vector_db.client.list_collections()
    }


@app.post("/product", response_model=BasicResponse)
async def create_product(
        title: str = Form(...),
        image: UploadFile = File(...),
        db=Depends(get_db)
):
    """
    接收商品标题和图片，完成 “保存图片→插入 Milvus→插入 SQL 数据库” 的全流程，
    返回创建结果（使用form表单提交）
    Args:
        title: 商品标题（form字段）
        image: 商品图片文件（文件上传字段，支持jpg、png等格式）
        db: 数据库会话

    Returns:
        创建的商品信息
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="请上传有效的图片文件"
            )

        file_extension = image.filename.split(".")[-1] if "." in image.filename else "jpg"

        unique_filename = str(uuid.uuid4()) + f".{file_extension}"
        image_path = os.path.join(IMAGE_DIR, unique_filename)

        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)

        insert_result, milvus_primary_key = insert_product_(image_path, title)

        if insert_result:
            # 关系型数据库：创建商品实例，使用相对路径存储
            db_product = Product(
                title=title,
                image_path=image_path,
                milvus_primary_key=milvus_primary_key,
                is_synced=True  # 标记已同步
            )
            message = "商品创建成功"
        else:
            db_product = Product(
                title=title,
                image_path=image_path,
                milvus_primary_key=-1,
                is_synced=False  # 标记未同步
            )
            message = "商品创建成功，但向量同步失败，将异步重试"

        # 保存到数据库
        db.add(db_product)  # 将实例添加到数据库会话
        db.commit()

        # 从数据库中重新加载最新的商品数据，更新当前内存中的 db_product 对象
        db.refresh(db_product)

        return BasicResponse(
            status=200,
            message=message,
            data={
                "id": db_product.id,
                "create_at": db_product.created_at,
                "is_synced": db_product.is_synced
            }
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"图片处理失败: {str(e)}"
        )


@app.get("/product/list", response_model=BasicResponse)
async def list_products(
        page_index: int = Query(1, description="页码（从 1 开始）", ge=1),
        page_size: int = Query(10, description="每页商品条数", ge=1, le=100),
        db=Depends(get_db)
):
    """
    获取所有商品列表（支持分页+排序）

    Args:
        page_index: 查询页面的大小（默认1，最小1）
        page_size: 具体查询的页面（默认10，最小1，最大100）
        order: 排序逻辑（默认created_at_desc：按插入时间倒序）
        db: 数据库会话

    Returns:
        包含分页商品信息、总条数、总页数的结果
    """
    try:
        sort_func = desc(Product.created_at)

        offset_num = (page_index - 1) * page_size

        # query = db.query(
        #     Product.id, Product.title, Product.image_path,
        #     Product.created_at, Product.updated_at,
        #     Product.milvus_primary_key
        # )
        query = db.query(Product).filter(Product.is_synced == True)

        query = query.order_by(sort_func)
        query = query.offset(offset_num)
        query = query.limit(page_size)

        products_data = query.all()

        products = []
        if products_data:
            for product in products_data:
                product_dict = {
                    "id": product.id,
                    "title": product.title,
                    "image_path": product.image_path,
                    "created_at": product.created_at,
                    "updated_at": product.updated_at,
                    "milvus_primary_key": product.milvus_primary_key
                }
                products.append(product_dict)

        total = db.query(Product).count()
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return BasicResponse(
            status=200,
            message="查询结果成功",
            data={
                "products": products,
                "pagination": {
                    "page_index": page_index,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages
                }
            }
        )
    except Exception as e:
        traceback.print_exc()
        return BasicResponse(
            status=500,
            message=f"商品列表查询失败：{str(e)}",
            data=None
        )


@app.get("/product/{product_id}", response_model=BasicResponse)
async def get_product(product_id: int, db=Depends(get_db)):  # product_id 是接收路由中 {product_id} 的值
    product = db.query(Product).filter(Product.id == product_id).first()

    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        product_dict = {
            "id": product.id,
            "title": product.title,
            "image_path": product.image_path,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
            "milvus_primary_key": product.milvus_primary_key
        }
        return BasicResponse(
            status=200,
            message="查询成功",
            data=product_dict
        )


@app.patch("/product/{product_id}/title", response_model=BasicResponse)
async def update_product_title(
        product_id: int,
        title: str = Form(None),
        db=Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()

    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        update_data = {"title": title}
        for field, value in update_data.items():
            setattr(product, field, value)

        delete_product_([product.milvus_primary_key])

        insert_result, milvus_primary_key = insert_product_(product.image_path, product.title)

        product.milvus_primary_key = milvus_primary_key

        db.commit()
        db.refresh(product)

        product_dict = {
            "id": product.id,
            "title": product.title,
            "image_path": product.image_path,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
            "milvus_primary_key": product.milvus_primary_key
        }

        return BasicResponse(
            status=200,
            message="商品更新成功",
            data=product_dict
        )


@app.patch("/product/{product_id}/image", response_model=BasicResponse)
async def update_product_image(
        product_id: int,
        image: UploadFile = File(None),
        db=Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()

    if not product:
        return BasicResponse(
            status=404,
            message="商品不存在",
            data={}
        )
    else:
        new_image_path = image.filename

        with open(product.image_path, "wb") as f:
            content = await image.read()
            f.write(content)

        delete_product_([product.milvus_primary_key])

        insert_result, milvus_primary_key = insert_product_(new_image_path, product.title)

        product.image_path = new_image_path
        product.milvus_primary_key = milvus_primary_key

        db.commit()
        db.refresh(product)

        product_dict = {
            "id": product.id,
            "title": product.title,
            "image_path": product.image_path,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
            "milvus_primary_key": product.milvus_primary_key
        }

        return BasicResponse(
            status=200,
            message="商品更新成功",
            data=product_dict
        )

@app.delete("/product/{product_id}", response_model=BasicResponse)
def remove_product(product_id: int, db=Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()

    try:
        if not product:
            return BasicResponse(
                status=404,
                message='商品不存在',
                data={}
            )

        try:
            if product.milvus_primary_key:
                delete_product_([product.milvus_primary_key])
            print("Milvus 数据删除成功")
        except Exception as e:
            traceback.print_exc()
            print(f"Milvus 数据删除失败：{str(e)}")

        try:
            if os.path.exists(product.image_path):
                os.remove(product.image_path)
        except Exception as e:
            traceback.print_exc()
            print(f"图片删除失败：{str(e)}")

        db.delete(product)
        db.commit()

        return BasicResponse(
            status=200,
            message='商品删除成功',
            data={"product_id": product_id}
        )
    except Exception as e:
        traceback.print_exc()
        return BasicResponse(
            status=500,
            message=f"商品删除失败：{str(e)}",
            data={"product_id": product_id}
        )


@app.post("/product/search", response_model=BasicResponse)
async def search_product(search_request: SearchRequest, db=Depends(get_db)):
    """
    支持四种跨模态检索模式，返回按相似度排序的商品结果，是项目核心业务接口
    支持四种搜索模式：
    1. text2text: 文本搜索文本
    2. text2image: 文本搜索图片
    3. image2text: 图片搜索文本
    4. image2image: 图片搜索图片

    Args:
        search_request: 搜索请求参数，包含搜索类型、查询内容和返回数量
        db: 数据库会话

    Returns:
        搜索结果列表，按相似度排序

    Raises:
        HTTPException: 当请求参数无效时返回400错误
    """
    # 在线程池中执行同步的模型推理
    loop = asyncio.get_event_loop()

    valid_search_types = ["text2text", "text2image", "image2text", "image2image"]

    if search_request.search_type not in valid_search_types:
        return BasicResponse(
            status=400,
            message=f"无效的搜索类型，请使用以下类型之一: {', '.join(valid_search_types)}",
            data=None
        )

    if search_request.search_type == "text2text":
        if not search_request.query_text:
            return BasicResponse(
                status=400,
                message="文本搜索模式需要提供query_text参数",
                data=None
            )
    elif search_request.search_type == "text2image":
        if not search_request.query_text:
            return BasicResponse(
                status=400,
                message="文本搜索模式需要提供query_text参数",
                data=None
            )
    elif search_request.search_type == "image2text":
        if not search_request.query_image:
            return BasicResponse(
                status=400,
                message="图片搜索模式需要提供query_image参数",
                data={}
            )
    elif search_request.search_type == "image2image":
        if not search_request.query_image:
            return BasicResponse(
                status=400,
                message="图片搜索模式需要提供query_image参数",
                data=None
            )

    try:
        if search_request.search_type in ["text2text", "text2image"]:
            # 【文本搜索】把 vector_db.search_product 放到线程池
            search_func = partial(
                vector_db.search_product_,
                title=search_request.query_text,
                image=None,
                task=search_request.search_type,
                top_k=search_request.top_k
            )
            success, results = await loop.run_in_executor(executor, search_func)
        elif search_request.search_type in ["image2text", "image2image"]:
            # 【图片搜索】分两步：Base64解码（同步，但很快）+ 推理（线程池）

            # 1. Base64解码（同步，数据小，不阻塞太久）
            image_bytes = base64.b64decode(search_request.query_image)
            image = Image.open(BytesIO(image_bytes))

            # 2. 模型推理（放到线程池，避免阻塞）
            search_func = partial(
                vector_db.search_product_,
                image=image,
                title=None,
                task=search_request.search_type,
                top_k=search_request.top_k
            )
            success, results = await loop.run_in_executor(executor, search_func)

        if not success:
            return BasicResponse(
                status=400,
                message="搜索失败",
                data=None
            )

        top_product_ids = [item["primary_key"] for item in results[0]]
        top_product_distance = [item["distance"] for item in results[0]]

        top_products = db.query(Product).filter(Product.milvus_primary_key.in_(top_product_ids)).all()

        search_results = []
        for product in top_products:
            distance = top_product_distance[top_product_ids.index(product.milvus_primary_key)]
            search_results.append(
                {
                    "id": product.id,
                    "title": product.title,
                    "image_path": product.image_path,
                    "created_at": product.created_at,
                    "updated_at": product.updated_at,
                    "milvus_primary_key": product.milvus_primary_key,
                    "distance": distance  # 相似度
                }
            )

        search_results.sort(key=lambda x: x["distance"], reverse=True)

        return BasicResponse(
            status=200,
            message="搜索成功",
            data=search_results
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return BasicResponse(
            status=500,
            message=f"搜索过程中发生错误: {str(e)}",
            data=None
        )

if __name__ == "__main__":
    # 用uvicorn运行FastAPI服务
    uvicorn.run(
        app,  # 要运行的API实例
        host="127.0.0.1",  # 允许所有设备访问（比如同一局域网的电脑能访问）
        port=8000,  # 端口号从配置文件拿（比如8000）
    )
    
