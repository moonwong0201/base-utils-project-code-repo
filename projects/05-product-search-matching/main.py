import base64
import os.path

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi import Depends, Query
import time
import uuid
import traceback
from PIL import Image
from io import BytesIO

from sqlalchemy import create_engine
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker

from orm_models import create_tables, Product

from vector_db import insert_product_, delete_product_, search_product_

from data_models import BasicResponse, BasicRequest, SearchRequest

from config import DATABASE_URL, IMAGE

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
    }


@app.post("/product", response_model=BasicResponse)
async def create_product(
        title: str = Form(...),
        image: UploadFile = File(...),
        db=Depends(get_db)
):
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

        db_product = Product(
            title=title,
            image_path=image_path,
            milvus_primary_key=milvus_primary_key
        )

        db.add(db_product)
        db.commit()
        db.refresh(db_product)

        return BasicResponse(
            status=200,
            message="商品创建成功",
            data={
                "id": db_product.id,
                "create_at": db_product.created_at
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
        page_index:int = Query(1, description="页码（从 1 开始）", ge=1),
        page_size: int = Query(10, description="每页商品条数", ge=1, le=100),
        db=Depends(get_db)
):
    try:
        sort_func = desc(Product.created_at)

        offset_num = (page_index - 1) * page_size

        query = db.query(
            Product.id, Product.title, Product.image_path,
            Product.created_at, Product.updated_at,
            Product.milvus_primary_key
        )

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
        success = False
        results = []
        if search_request.search_type in ["text2text", "text2image"]:
            success, results = search_product_(
                title=search_request.query_text,
                top_k=search_request.top_k,
                task=search_request.search_type
            )
        elif search_request.search_type in ["image2text", "image2image"]:
            image = Image.open(BytesIO(base64.b64decode(search_request.query_image)))
            success, results = search_product_(
                image=image,
                top_k=search_request.top_k,
                task=search_request.search_type
            )

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
    test_title = "测试皮卡丘"
    test_image_path = "/Users/wangyingyue/materials/大模型学习资料——八斗/第十一周：多模态检索与问答/Week11/05-product-search-matching/pokemon.jpeg"

    # 2. 手动创建数据库会话
    db = SessionLocal()

    try:
        # 模拟保存图片（和接口逻辑一致）
        file_extension = test_image_path.split(".")[-1]
        unique_filename = str(uuid.uuid4()) + f".{file_extension}"
        image_path = os.path.join(IMAGE_DIR, unique_filename)
        with open(test_image_path, "rb") as f_in, open(image_path, "wb") as f_out:
            f_out.write(f_in.read())

        # 3. 直接调用insert_product（重点：这里加断点）
        insert_result, milvus_primary_key = insert_product_(image_path, test_title)
        print(f"Milvus主键: {milvus_primary_key}")  # 打印主键，看是否为空

        # 4. 写入数据库
        db_product = Product(
            title=test_title,
            image_path=image_path,
            milvus_primary_key=milvus_primary_key
        )
        db.add(db_product)
        db.commit()
        db.refresh(db_product)

        print(f"商品创建成功，ID: {db_product.id}, Milvus主键: {db_product.milvus_primary_key}")
    except Exception as e:
        db.rollback()
        print(f"测试失败: {str(e)}")
    finally:
        db.close()
