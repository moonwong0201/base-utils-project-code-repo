from pydantic import BaseModel, Field  # BaseModel是Pydantic的基础类；Field用于字段描述和约束
from typing import Dict, List, Any, Union, Optional  # 类型提示工具：
# Optional[str] 表示可以是str类型或None
# Union[str, List[str]] 表示可以是str类型或列表类型


class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    # 第一个字段：请求ID
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    # 第二个字段：请求文本
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")
