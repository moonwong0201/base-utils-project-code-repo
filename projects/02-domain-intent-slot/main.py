from fastapi import FastAPI
import openai


from config import Args
from data_schema import TextRequest, ExtractResponse
from bert import BertExtract
from tools import tools_extraction
from prompt import prompt_Extraction


app = FastAPI(title="意图与槽位抽取", version="1.0")

args = Args()
client = openai.OpenAI(
    api_key=args.API_KEY,
    base_url=args.BASE_URL
)

try:
    bert_extractor = BertExtract()
    print("BERT模型全局初始化成功")
except Exception as e:
    bert_extractor = None
    print(f"BERT模型全局初始化失败: {str(e)}")


def get_error_response(request: TextRequest, error_msg: str) -> ExtractResponse:
    """统一失败响应格式，避免Pydantic校验错误"""
    return ExtractResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        msg=f"抽取失败: {error_msg}",
        data=[]  # 失败时data也返回空列表，而非字符串
    )


@app.post("/extract/bert", response_model=ExtractResponse)
async def extract_bert(request: TextRequest):
    try:
        result = bert_extractor.extract(request.request_text)
        return ExtractResponse(
            request_id=request.request_id,
            request_text=request.request_text,
            msg="success",
            data=result
        )
    except Exception as e:
        return get_error_response(request, str(e))


@app.post("/extract/prompt", response_model=ExtractResponse)
async def extract_prompt(request: TextRequest):
    try:
        result = prompt_Extraction(request.request_text)
        return ExtractResponse(
            request_id=request.request_id,
            request_text=request.request_text,
            msg="success",
            data=result
        )
    except Exception as e:
        return get_error_response(request, str(e))


@app.post("/extract/tools", response_model=ExtractResponse)
async def extract_tools(request: TextRequest):
    try:
        result = tools_extraction(request.request_text)
        return ExtractResponse(
            request_id=request.request_id,
            request_text=request.request_text,
            msg="success",
            data=result
        )
    except Exception as e:
        return get_error_response(request, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=8000, reload=True)


