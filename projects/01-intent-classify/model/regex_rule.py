import re  # Python自带的正则表达式库，用于字符串匹配
from typing import Union, List

from config import REGEX_RULE  # 从配置文件导入正则规则（这是分类的核心依据）

# 预编译的目的是提高效率———正则表达式编译一次后可以反复使用，避免每次匹配都重新解析规则
REGEX_RULE_COMPILED = {}
# 1. 取出该类别下的所有关键词/模式列表（比如REGEX_RULE["售后问题"] = ["退款", "退货", "换货"]）
# 2. 用"|"拼接成一个正则表达式字符串（比如"退款|退货|换货"）
# 3. 用re.compile()编译成正则对象（预编译可以提高后续匹配效率）
for category in REGEX_RULE.keys():
    REGEX_RULE_COMPILED[category] = re.compile("|".join(REGEX_RULE[category]))


# 定义分类结果变量，可能是字符串或列表（与输入格式对应）
def model_for_regex(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if request_text is None:
        raise ValueError("请求文本不能为空")

    classify_result: Union[str, List[str]] = []

    if isinstance(request_text, str):
        for category in REGEX_RULE_COMPILED.keys():
            # 用该类别的正则对象匹配文本，findall()返回所有匹配的结果
            if REGEX_RULE_COMPILED[category].findall(request_text):
                classify_result.append(category)
                break
        # 如果没有任何类别匹配成功，就归类为"Other"
        if not classify_result:
            classify_result.append("Other")
    elif isinstance(request_text, list):
        classify_result = []
        for text in request_text:
            is_classified = False  # 标记该文本是否已分类
            for category in REGEX_RULE_COMPILED.keys():
                if REGEX_RULE_COMPILED[category].findall(text):
                    classify_result.append(category)
                    is_classified = True
                    break
            if not is_classified:
                classify_result.append("Other")
    else:
        raise Exception("格式不支持")

    return classify_result


if __name__ == "__main__":
    texts = [
        '查询下周三我的日程安排是什么',
        '帮我播放《狂飙》第15集',
        '今天天气不错，适合出门散步',
        '给我播放张敬轩的歌曲《春秋》',
        '帮我打开客厅的空调，设置温度26度',
        '帮我播放湖南卫视的《快乐大本营》最新一期',
        '收听FM93.8交通广播的实时路况',
        '查询明天上海的天气情况，是否有雨',
        '把明天早上7点的闹钟修改为7点30分',
        '播放《明朝那些事儿》的有声小说第10章',
        '帮我找到电影《流浪地球2》并播放',
        '查询从北京到上海的高铁票，明天下午出发的'
    ]

    result = model_for_regex( texts)
    for i, text in enumerate(texts):
        print(f"文本：{text}")
        print(f"意图：{result[i]}\n")
