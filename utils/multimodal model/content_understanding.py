import os

# 添加环境变量，彻底禁用 MPS/CUDA，强制使用 CPU，避免内存报错
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["DEVICE"] = "cpu"
os.environ["MPS_DEVICE_MAX_MEMORY"] = "0"

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

device = torch.device("cpu")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float32,  # CPU 专用精度，稳定无兼容问题
    device_map="cpu",  # 强制加载到 CPU
    low_cpu_mem_usage=True,  # 低内存占用优化
    trust_remote_code=True  # 加载 Qwen 自定义结构必需
)

# 加载配套处理器
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    use_fast=False
)

# 简单图文推理
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)  # 统一转到 CPU，与模型设备匹配

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"描述：{output_text[0]}") 


def qwen_inference(prompt, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
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
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 解码为文本
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    result_text = output_text[0]  # 提取整洁的文本结果
    # print(f"模型输出：{result_text}")
    image_obj = Image.open(image_path)  # 加载图片对象（可选）
    return result_text, image_obj  # 返回 文本结果 + 图片对象，满足后续使用


# 调用函数，获取结果并打印（匹配修正后的返回值）
# 推理 1：中文描述图片
response1_text, response1_image = qwen_inference(
    "中文描述这个图片",
    "./data/Xnip2026-01-28_22-00-16.jpg"
)
print(f"图片描述结果：{response1_text}")

# 推理 2：消防安全隐患识别
response2_text, response2_image = qwen_inference(
    """你是专业消防专家，基于提供的图片视觉内容，识别图片中的场景是否为楼道，并判断消防安全隐患的风险等级。
风险等级判定规则：
1. 高风险：楼道中出现电动车、电瓶、飞线充电等可能起火的元素；
2. 中风险：楼道中存在大量堆积物严重影响通行，或堆放大量纸箱、木质家具等易造成火势蔓延的堵塞物；
3. 低风险：楼道存在堆物现象但不严重，不影响通行；
4. 无风险：楼道环境干净，无任何堆放物品；
5. 非楼道：图片场景与楼道无关。
【输出要求】
1. 严格基于图片的实际视觉内容判断，不主观臆断；
2. 禁止复述任何判定规则，禁止添加任何解释、说明、标点符号；
3. 仅输出5个结果词中的其中一个：高风险、中风险、低风险、无风险、非楼道；
4. 输出内容仅限上述一个词，无任何其他文字。""",
    "./data/068f25bbb9c549919385bf2afc0553aa.jpg"
)
print(f"消防安全风险评价：{response2_text}")
