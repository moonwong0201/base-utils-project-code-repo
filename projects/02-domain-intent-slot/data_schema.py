from pydantic import BaseModel, Field  # BaseModel是Pydantic的基础类；Field用于字段描述和约束
from typing import Dict, List, Any, Union, Optional


class TextRequest(BaseModel):
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class ExtractResponse(BaseModel):
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    msg: str = Field("success", description="状态信息")
    data: List[dict] = Field(..., description="抽取结果列表")

