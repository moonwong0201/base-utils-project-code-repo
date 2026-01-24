import json
import openai
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Literal
from datetime import datetime

client = openai.OpenAI(
    api_key="sk-078ae61448344f53b3cb03bcc85ff7cd",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    """信息抽取智能体：用于调用大模型，完成文本的领域、意图、实体抽取"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 构造工具调用参数
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()["title"],
                    "description": response_model.model_json_schema()["description"],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()["properties"],
                        "required": response_model.model_json_schema().get("required", [])
                    }
                }

            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            result = response_model.model_validate_json(arguments)
            return result
        except Exception as e:
            print(f"处理失败，异常信息：{str(e)}")
            return None


class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    # 领域类别
    domain: Literal[
        'music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather',
        'match', 'map', 'website', 'news', 'message', 'contacts', 'translation',
        'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone',
        'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'
    ] = Field(description="用户提问的所属类型领域")
    # 意图类型
    intent: Literal[
        'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
        'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
        'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
        'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'
    ] = Field(description="用户提问的意图")

    # 实体字段
    Src: Optional[str] = Field(default=None, description="来源")
    startDate_dateOrig: Optional[str] = Field(default=None, description="开始日期原始值")
    film: Optional[str] = Field(default=None, description="电影")
    endLoc_city: Optional[str] = Field(default=None, description="目的地城市")
    artistRole: Optional[str] = Field(default=None, description="艺术家角色")
    location_country: Optional[str] = Field(default=None, description="位置国家")
    location_area: Optional[str] = Field(default=None, description="位置区域")
    author: Optional[str] = Field(default=None, description="作者")
    startLoc_city: Optional[str] = Field(default=None, description="出发地城市")
    season: Optional[str] = Field(default=None, description="季节")
    dishNamet: Optional[str] = Field(default=None, description="菜品名称类型")
    media: Optional[str] = Field(default=None, description="媒体")
    datetime_date: Optional[datetime] = Field(default=None, description="日期时间-日期")
    episode: Optional[str] = Field(default=None, description="剧集")
    teleOperator: Optional[str] = Field(default=None, description="电信运营商")
    questionWord: Optional[str] = Field(default=None, description="疑问词")
    receiver: Optional[str] = Field(default=None, description="接收者")
    ingredient: Optional[str] = Field(default=None, description="食材")
    name: Optional[str] = Field(default=None, description="名称")
    startDate_time: Optional[datetime] = Field(default=None, description="开始时间")
    startDate_date: Optional[datetime] = Field(default=None, description="开始日期")
    location_province: Optional[str] = Field(default=None, description="位置省份")
    endLoc_poi: Optional[str] = Field(default=None, description="目的地兴趣点")
    artist: Optional[str] = Field(default=None, description="艺术家")
    dynasty: Optional[str] = Field(default=None, description="朝代")
    area: Optional[str] = Field(default=None, description="区域")
    location_poi: Optional[str] = Field(default=None, description="位置兴趣点")
    relIssue: Optional[str] = Field(default=None, description="相关问题")
    Dest: Optional[str] = Field(default=None, description="目的地")
    content: Optional[str] = Field(default=None, description="内容")
    keyword: Optional[str] = Field(default=None, description="关键词")
    target: Optional[str] = Field(default=None, description="目标")
    startLoc_area: Optional[str] = Field(default=None, description="出发地区域")
    tvchannel: Optional[str] = Field(default=None, description="电视频道")
    type: Optional[str] = Field(default=None, description="类型")
    song: Optional[str] = Field(default=None, description="歌曲")
    queryField: Optional[str] = Field(default=None, description="查询字段")
    awayName: Optional[str] = Field(default=None, description="客队名称")
    headNum: Optional[str] = Field(default=None, description="人数")
    homeName: Optional[str] = Field(default=None, description="主队名称")
    decade: Optional[str] = Field(default=None, description="年代")
    payment: Optional[str] = Field(default=None, description="支付方式")
    popularity: Optional[str] = Field(default=None, description="流行度")
    tag: Optional[str] = Field(default=None, description="标签")
    startLoc_poi: Optional[str] = Field(default=None, description="出发地兴趣点")
    date: Optional[str] = Field(default=None, description="日期")
    startLoc_province: Optional[str] = Field(default=None, description="出发地省份")
    endLoc_province: Optional[str] = Field(default=None, description="目的地省份")
    location_city: Optional[str] = Field(default=None, description="位置城市")
    absIssue: Optional[str] = Field(default=None, description="绝对问题")
    utensil: Optional[str] = Field(default=None, description="厨具")
    scoreDescr: Optional[str] = Field(default=None, description="分数描述")
    dishName: Optional[str] = Field(default=None, description="菜品名称")
    endLoc_area: Optional[str] = Field(default=None, description="目的地区域")
    resolution: Optional[str] = Field(default=None, description="分辨率")
    yesterday: Optional[str] = Field(default=None, description="昨天")
    timeDescr: Optional[str] = Field(default=None, description="时间描述")
    category: Optional[str] = Field(default=None, description="类别")
    subfocus: Optional[str] = Field(default=None, description="子焦点")
    theatre: Optional[str] = Field(default=None, description="剧院")
    datetime_time: Optional[datetime] = Field(default=None, description="日期时间-时间")

    # 将 Pydantic 模型实例转换为目标嵌套格式的 JSON 字符串
    def to_target_format(self, indent: int = 2) -> str:
        model_dict = self.model_dump()  # 将 Pydantic 模型实例转换为标准 Python 字典
        slots = {}
        for field_name, field_value in model_dict.items():
            if field_name not in ["domain", "intent"] and field_value is not None:
                slots[field_name] = field_value

        target_dict = {
            "domain": self.domain,
            "intent": self.intent,
            "slots": slots
        }

        return json.dumps(target_dict, ensure_ascii=False, indent=indent, default=str)


if __name__ == "__main__":
    text_list = [
        "糖醋鲤鱼怎么做啊？",
        "帮我查下北京到上海的高铁票",
        "明天上海的天气怎么样？",
        "推荐几首周杰伦的经典歌曲"
    ]

    agent = ExtractionAgent(model_name="qwen-plus")
    for idx, text in enumerate(text_list):
        print(f"正在处理第 {idx + 1} 条文本：{text}")
        result = agent.call(text, IntentDomainNerTask)  # 已经完成了实例化
        if result:
            print(result.to_target_format())
            print()
        else:
            print(f"第 {idx + 1} 条文本未匹配到有效 json\n")
