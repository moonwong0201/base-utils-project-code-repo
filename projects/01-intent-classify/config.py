import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

REGEX_RULE = {
    "Travel-Query": ["导航到", "去.*地方", "最近的.*加油站", "路线"],
    "Music-Play": ["播放.*音乐", "听.*歌", "切歌", "暂停音乐", "歌曲"],
    "FilmTele-Play": ["播放.*电视剧", "看.*电视剧"],
    "Video-Play": ["播放.*视频", "看.*视频"],
    "Radio-Listen": ["听.*广播", "打开.*电台"],
    "HomeAppliance-Control": ["调.*空调", "开.*空调", "关.*空调", "调节.*温度"],
    "Weather-Query": ["查.*天气", "今天.*温度", "明天.*下雨"],
    "Alarm-Update": ["设置.*闹钟", "取消.*闹钟", "修改.*闹钟"],
    "Calendar-Query": ["查.*日历", "今天.*星期几", "明天.*日期", "提醒"],
    "TVProgram-Play": ["播放.*电视", "看.*电视节目"],
    "Audio-Play": ["播放.*音频", "听.*有声书"],
    "Other": []
}


CATEGORY_NAME = [
    'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
    'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
    'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play',
    'Other'
]

TFIDF_MODEL_PKL_PATH = "assets/weights/tfidf_ml.pkl"

BERT_MODEL_PKL_PATH = "assets/weights/bert.pt"
BERT_MODEL_PERTRAINED_PATH = "models/google-bert/bert-base-chinese"

LABEL_MAP_PATH = 'assets/label2id.json'

DATA = 'assets/dataset/dataset.csv'

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1"  # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "qwen3:0.6b"
