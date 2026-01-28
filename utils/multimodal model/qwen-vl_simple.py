import ast  # 用于解析字符串为Python对象（如列表、字典）
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from IPython.display import Markdown, display  # 在IPython环境中显示Markdown内容

import os
os.environ["MULTIPROCESSING_DISABLE_SEMAPHORE_CLEANUP"] = "1"  # 消除信号量警告
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 禁用 MPS 回退
os.environ["DEVICE"] = "cpu"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
import json

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    use_fast=False
)

# ==================================================================
# =========================== 简单图文推理 ===========================
# ==================================================================
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./data/4.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
# process_vision_info：从messages中提取图像信息并预处理：
# 读取本地图像。
# 按模型要求进行图像预处理（如缩放、归一化等），结果存入image_inputs。
# 由于消息中没有视频，video_inputs为空。

inputs = processor(
    text=[text],  # 处理好的文本（列表形式支持批量输入）
    images=image_inputs,  # 预处理后的图像
    videos=video_inputs,  # 视频输入（此处为空）
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cpu")

# Inference: Generation of the output
# model.generate：调用模型生成对图像的描述文本。
# **inputs：将处理好的输入（文本张量、图像张量等）传入模型。
# max_new_tokens=128：限制生成的最大 token 数量为 128（控制回复长度）
# 返回generated_ids：包含输入 token ID + 新生成 token ID 的张量
generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,   # 要解码的新生成token ID
    skip_special_tokens=True,    # 忽略模型内部的特殊符号（如<|endoftext|>）
    clean_up_tokenization_spaces=False  # 不自动调整空格（保留原始生成的空格格式）
)
print(output_text)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./data/2002fe341d758e41e7d6cd239fe23ac2.jpeg",
            },
            {"type": "text", "text": "识别图片中有擦痕的区域，以json的格式返回区域的位置。"},
        ],
    }
]

# Preparation for inference
# 文本格式化
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 图像预处理
image_inputs, video_inputs = process_vision_info(messages)

# 输入整合
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cpu")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./data/xh44z6ajf6noc_22c15d2af82e42a290dac1dde66bc685.png",
            },
            {"type": "text", "text": "Extract content from the image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cpu")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


# ==================================================================
# =========================== 复杂任务演示 ===========================
# ==================================================================
# @title inference function
# 调用多模态模型进行推理
def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
    """
    封装「图片预处理→文本格式化→模型推理→结果解码」的完整流程，方便后续重复调用
    :param image_path:
    :param prompt:
    :param sys_prompt: 系统提示（定义模型角色）
    :param max_new_tokens: 最大生成 token 数
    :param return_input: 是否返回模型输入数据
    :return:
    """
    image = Image.open(image_path)
    image_local_path = "file://" + image_path  # 构建本地文件URL
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},  # 用户文本提示
            {"image": image_local_path},  # 图像路径
        ]
         },
    ]
    # 将对话消息列表messages格式化为模型能识别的文本格式
    # processor.apply_chat_template：将对话消息转换为模型要求的模板格式（如添加角色前缀、生成提示）
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("text:", text)
    # 用processor整合「格式化文本」和「PIL 图片」，转换为模型能直接运行的 PyTorch 张量（输入数据）
    inputs = processor(
        text=[text],  # 传入格式化后的文本
        images=[image],  # 传入 PIL 图片对象
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cpu')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 裁剪输入部分，只保留模型新生成的token ID
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # 将token ID解码为人类可读文本
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # 自动清理 token 化过程中产生的多余空格，让文本更整洁
    )
    if return_input:
        return output_text[0], inputs  # 返回生成文本和输入数据
    else:
        return output_text[0]  # 只返回生成文本


def plot_text_bounding_boxes(image_path, raw_response, input_width, input_height):
    import json, re
    from PIL import Image, ImageDraw

    # ---- 调试可开 ----
    print('---- raw_response type:', type(raw_response))
    print('---- raw_response repr:', repr(raw_response[:200]))

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 1. 去围栏
    json_str = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_response, flags=re.I).strip()
    # 2. 解析（支持双层转义）
    data = json.loads(json_str)
    while isinstance(data, str):          # 如果又被转义成字符串，再剥一次
        data = json.loads(data)

    # 3. 画框
    for box in data:                      # data 就是 list[dict]
        x1, y1, x2, y2 = box['bbox_2d']
        abs_x1 = int(x1 / input_width * width)
        abs_y1 = int(y1 / input_height * height)
        abs_x2 = int(x2 / input_width * width)
        abs_y2 = int(y2 / input_height * height)
        draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='green', width=2)
        if 'text_content' in box:
            draw.text((abs_x1, abs_y2 + 2), box['text_content'], fill='green')

    save_path = os.path.join(os.path.dirname(image_path),
                             f"{os.path.splitext(os.path.basename(image_path))[0]}_with_bboxes.png")
    img.save(save_path, quality=100)
    print("已保存：", save_path)
    img.show()


def parse_json(raw: str):
    """
    去掉 ```json ... ``` 或 ``` ... ``` 后返回纯 JSON 字符串
    """
    # 贪婪匹配 ```json 开头 到 ``` 结尾
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.I)
    if m:
        return m.group(1).strip()
    # 没找到 fences，直接返回原串
    return raw.strip()


# ============================ 提取文本并可视化 ============================
image_path = "./data/xh44z6ajf6noc_22c15d2af82e42a290dac1dde66bc685.png"
# prompt = "Spotting all the text in the image with line-level, and output in JSON format."
prompt = (
   """You are an OCR detector.

Task:
1) Detect ALL text lines in the image (line-level, do not merge lines).
2) For each line, output one JSON object with:
   - bbox_2d: [x1,y1,x2,y2] normalized to [0,1000]
   - text_content: the exact text in that line

Rules:
- Output MUST be valid JSON only (no markdown).
- bbox_2d must tightly cover ONLY that line.
- Do NOT output a single bbox covering multiple lines.
- The number of objects must equal the number of detected lines.
- Sort lines from top to bottom.

Return JSON list only.
"""
)

## Use a local HuggingFace model to inference.
response, inputs = inference(image_path, prompt, return_input=True)
display(Markdown(response))
input_height = inputs['image_grid_thw'][0][1] * 14
input_width = inputs['image_grid_thw'][0][2] * 14
print(input_height, input_width)
# 在图片上绘制文本边界框
plot_text_bounding_boxes(image_path, response, input_width, input_height)


# ============================ 提取图片中的关键信息 ============================
image_path = "./data/xh44z6ajf6noc_22c15d2af82e42a290dac1dde66bc685.png"
prompt = "Extract the key-value information in the format:{\"company\": \"\", \"date\": \"\", \"address\": \"\", \"total\": \"\"}"

image = Image.open(image_path)
display(image.resize((800, 400)))

## Use a local HuggingFace model to inference.
response = inference(image_path, prompt)
display(Markdown(response))
# text: <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Extract the key-value information in the format:{"company": "", "date": "", "address": "", "total": ""}<|vision_start|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
#
# {
#   "company": "临沧地区中级人民法院",
#   "date": "",
#   "address": "临沧县凤翔镇南屏西路37号",
#   "total": "6000 m²"
# }
