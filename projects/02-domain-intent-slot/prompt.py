import json

from typing import List, Optional, Union
import re
import openai
from config import Args

args = Args()
client = openai.OpenAI(
    api_key=args.API_KEY,
    base_url=args.BASE_URL,
)


def prompt_Extraction(text_list: Union[str, List[str]]):
    try:
        texts = "\n".join(text for text in text_list)
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本分别抽取领域类别、意图类别和实体标签
                 待选的领域类别有：music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
                 待选的意图类别有：OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
                 待选的实体标签有：code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest/content/keyword/target/startLoc_area/tvchannel/type/song/queryField/awayName/headNum/homeName/decade/payment/popularity/tag/startLoc_poi/date/startLoc_province/endLoc_province/location_city/absIssue/utensil/scoreDescr/dishName/endLoc_area/resolution/yesterday/timeDescr/category/subfocus/theatre/datetime_time
        
                 输出要求：
                 1. 所有结果必须放在一个JSON数组中，数组中的每个元素对应一条输入文本（按输入顺序排列）。
                 2. 每条结果的格式为：
                 {
                    "text": "输入文本", 
                    "domain": "领域标签", 
                    "intent": "意图标签", 
                    "slots": {
                        "实体标签": "实体值"
                    }
                 }。
                 3. 必须用```json和```包裹整个JSON数组，不允许添加任何额外说明文字。
        
        
                 要求最终输出的格式是如下的json格式，每一个文本都按照如下格式输出：text是输入文本，domain是领域标签，intent是意图标签，slots是实体识别结果和标签
                 ```json
                 [
                    {
                        "text": "播放周杰伦的歌",
                        "domain": "music",
                        "intent": "PLAY",
                        "slots": {
                            "artist": "周杰伦"
                        }
                    }
                ]
                 ```
                 """},

                {"role": "user", "content": texts},
            ],
        )


        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', completion.choices[0].message.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                results = json.loads(json_str)
                return results
                # print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                print("ERROR：未找到符合格式的 JSON 结果")
        except json.JSONDecodeError:
            print("ERROR：JSON 格式错误")
    except Exception as e:
        print(f"处理失败：{str(e)}")

# {
#     "text": "查询许昌到中山的汽车。",
#     "domain": "bus",
#     "intent": "QUERY",
#     "slots": {
#         "startLoc_city": "许昌",
#         "endLoc_city": "中山"
#     }
# }


