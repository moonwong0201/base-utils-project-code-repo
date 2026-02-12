import json
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from typing_extensions import Literal

import openai
from config import Args

args = Args()
client = openai.OpenAI(
    api_key=args.API_KEY,
    base_url=args.BASE_URL,
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        system_prompt = f"""
        你是专业的意图槽位抽取助手，处理规则如下：
        1. 领域选择：必须从 [{', '.join(response_model.model_fields['domain'].annotation.__args__)}] 中选1个，优先匹配文本核心场景；
        2. 意图选择：必须从 [{', '.join(response_model.model_fields['intent'].annotation.__args__)}] 中选1个；
        3. 实体抽取：只抽取文本中明确的核心实体，无实体则slots为空列表；
        4. 必须调用指定工具，返回标准JSON格式，不添加任何额外解释、备注、换行。
        """

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": f"请严格按照指定格式抽取以下文本的领域、意图和实体标签：{user_prompt}"
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": response_model.model_json_schema()
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "ToolsExtraction"}},
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class EntityAndSlots(BaseModel):
    """实体及对应标签"""
    label: Literal["code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country", "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time", "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi", "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel", "type", "song", "queryField", "awayName", "headNum", "homeName", "decade", "payment", "popularity", "tag", "startLoc_poi", "date", "startLoc_province", "endLoc_province", "location_city", "absIssue", "utensil", "scoreDescr", "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr", "category", "subfocus", "theatre", "datetime_time"] = Field(description="实体")
    entity: str = Field(description="标签")


class ToolsExtraction(BaseModel):
    """对文本进行抽取领域类别、意图类别和实体标签"""
    domain: Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"] = Field(description="领域类别")
    intent: Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(description="意图类别")
    slots: List[EntityAndSlots] = Field(description="实体标签")


def tools_extraction(text_list: Union[str, List[str]]):
    results = []
    if isinstance(text_list, str):
        text_list = [text_list]

    agent = ExtractionAgent(model_name="qwen-plus")

    for text in text_list:
        result_dict = {}
        try:
            result = agent.call(text, ToolsExtraction)

            if result:
                slot_dict = {}
                for slot in result.slots:
                    slot_dict[slot.label] = slot.entity

                    result_dict = {
                        "text": text,
                        "domain": result.domain,
                        "intent": result.intent,
                        "slots": slot_dict
                    }
            else:
                result_dict = {
                    "text": text,
                    "domain": None,
                    "intent": None,
                    "slots": {}
                }
        except Exception as e:
            print(f"文本 {text} 处理失败：{str(e)}")
            result_dict = {
                "text": text,
                "domain": None,
                "intent": None,
                "slots": {}
            }
        results.append(result_dict)

    return results
