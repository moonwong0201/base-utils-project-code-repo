# 用提示词的方法对文本进行领域类别、意图类型、实体标签的抽取
import json
import openai
import re

client = openai.OpenAI(
    api_key=API-KEY,
    base_url=URL,
)

text_list = [
    "糖醋鲤鱼怎么做啊？",
    "帮我查下北京到上海的高铁票",
    "明天上海的天气怎么样？",
    "推荐几首周杰伦的经典歌曲"
]

system_prompt = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website 
               / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke 
               / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH 
               / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD 
               / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country 
               / location_area / author / startLoc_city / season / dishNamet / media / datetime_date 
               / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time 
               / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi 
               / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song 
               / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi 
               / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr 
               / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

```json
{
    "domain": ,
    "intent": ,
    "slots": {
      "待选实体": "实体名词",
    }
}
```
"""

if __name__ == "__main__":
    for idx, text in enumerate(text_list):
        print(f"正在处理第 {idx + 1} 条文本：{text}")
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )

            result = completion.choices

            # ```json\s*(.*?)\s*``` 匹配规则
            # result[0].message.content 正则表达式要搜索的目标文本
            # re.DOTALL 让正则的 . 匹配包括换行符在内的所有字符
            json_match = re.search(r'```json\s*(.*?)\s*```', result[0].message.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result_str = json.loads(json_str)
                formatted_json = json.dumps(result_str, ensure_ascii=False, indent=4)
                print(formatted_json)
                print()
            else:
                print(f"第 {idx} 条文本未匹配到有效 json\n")
        except Exception as e:
            print(f"第 {idx} 条文本处理失败，错误信息：{str(e)}\n")

