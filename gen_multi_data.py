import os
import json
import random
import torch
import warnings
import argparse
import difflib
import signal
import sys
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import swanlab

# ===========================================================
# Module 0: 参数解析
# ===========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="InternVL Dataset Generator (Dual Mode: Image & Text)")
    
    # [核心] 数据类型选择
    parser.add_argument("--data_type", type=str, default="image", choices=["image", "text"], 
                        help="数据生成模式: 'image' (多模态) 或 'text' (纯文本)")
    
    # 核心模式
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "refusal", "normal"])

    # 路径参数
    parser.add_argument("--image_dir", type=str, default=None, help="[Image模式] 图片文件夹路径")
    parser.add_argument("--text_file", type=str, default=None, help="[Text模式] 文本输入文件路径 (.json list)")
    
    # 标签与输出
    parser.add_argument("--img_prefix", type=str, default="Privacy", help="子类名称/ID前缀, e.g. 'weapon'")
    parser.add_argument("--category", type=str, default="Safety", help="数据集大类标签, e.g. 'Violence'")
    parser.add_argument("--output_json", type=str, default="output.json", help="输出路径")
    parser.add_argument("--json_prefix", type=str, default="", help="JSON中图片路径的前缀")
    
    # Prompt 配置
    parser.add_argument("--prompt_config", type=str, default="prompts/all_prompts.json", help="Prompt 配置文件路径")

    # 模型与硬件
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--gpu_id", type=str, default="0", help="指定使用的 GPU ID")
    parser.add_argument("--sim_threshold", type=float, default=0.3, help="相似度阈值 (Text模式下通常用于判断是否是拒绝回复)")

    return parser.parse_args()

# ===========================================================
# Module A: 文件与数据处理
# ===========================================================

def get_and_rename_images(directory, prefix):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(directory):
        print(f"❌ [Error] Image directory not found: {directory}")
        return []
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)])
    final_file_list = []
    for idx, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{prefix}_{idx}{ext}"
        full_old = os.path.join(directory, old_name)
        full_new = os.path.join(directory, new_name)
        if old_name != new_name:
            try:
                if not os.path.exists(full_new):
                    os.rename(full_old, full_new)
                    final_file_list.append(new_name)
                else:
                    final_file_list.append(new_name)
            except OSError:
                final_file_list.append(old_name)
        else:
            final_file_list.append(new_name)
    return final_file_list

def load_text_inputs(filepath):
    """
    加载纯文本输入。支持简单列表或带元数据的对象列表。
    """
    if not os.path.exists(filepath):
        print(f"❌ [Error] Text file not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"🔍 Loaded {len(data)} text inputs.")
    return data

def load_existing_progress(output_json_path):
    if not os.path.exists(output_json_path):
        return [], set()
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容不同ID格式，提取基础ID
            processed_ids = {entry['id'].replace("_normal", "").replace("_sensitive", "") for entry in data}
            print(f"📥 Loaded {len(data)} entries. Processed base IDs: {len(processed_ids)}")
            return data, processed_ids
    except Exception:
        return [], set()

def load_prompts_from_file(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ===========================================================
# Module B: Prompt 构造 (含层级回退逻辑)
# ===========================================================

def construct_full_input(user_prompt_config, sys_prompt_config, 
                         is_refusal_mode=True, 
                         subcategory=None, 
                         category=None, 
                         forced_reason=None, 
                         text_input=None):
    
    # 内部类：上下文感知字典 (核心逻辑所在)
    class ContextAwareSlotDict(dict):
        def __init__(self, slots, subcategory, category, forced_reason=None):
            self.slots = slots
            self.subcategory = subcategory
            self.category = category
            self.forced_reason = forced_reason

        def __missing__(self, key):
            # 1. 理由生成的层级回退逻辑
            if key == "reason":
                # 优先级 A: 强制注入 (通常来自模型自生成的具体理由)
                if self.forced_reason: return self.forced_reason
                
                # 优先级 B: 精准子类模板 (reason_Bomb)
                if self.subcategory:
                    if f"reason_{self.subcategory}" in self.slots: return random.choice(self.slots[f"reason_{self.subcategory}"])
                
                # 优先级 C: 大类模板 (reason_Violence) -> 这是最常用的
                if self.category:
                    if f"reason_{self.category}" in self.slots: return random.choice(self.slots[f"reason_{self.category}"])
                
                # 优先级 D: 默认模板
                if "reason_default" in self.slots: return random.choice(self.slots["reason_default"])
            
            # 2. 基础元数据插槽
            if key == "subcategory": return self.subcategory if self.subcategory else "sensitive content"
            if key == "category": return self.category if self.category else "safety policies"

            # 3. 普通插槽回退
            if f"{key}_default" in self.slots: return random.choice(self.slots[f"{key}_default"])
            if key in self.slots: return random.choice(self.slots[key])
            return "{" + key + "}"

    # 构造 System Prompt
    if isinstance(sys_prompt_config, list):
        sys_instruction = random.choice(sys_prompt_config)
    else:
        s_temp = random.choice(sys_prompt_config.get("templates", ["You are a helpful assistant."]))
        s_slots = sys_prompt_config.get("slots", {})
        sys_instruction = s_temp.format_map(ContextAwareSlotDict(s_slots, subcategory, category, forced_reason))

    # 构造 User Prompt
    if text_input:
        # [Text Mode] 直接根据配置决定是裸奔还是套壳
        if isinstance(user_prompt_config, list): # 简单的列表
            u_temp = random.choice(user_prompt_config)
        else:
            u_temp = random.choice(user_prompt_config.get("templates", ["{text_input}"]))
        
        # 将 text_input 填入
        user_prompt = u_temp.replace("{text_input}", text_input)
    else:
        # [Image Mode]
        if isinstance(user_prompt_config, list):
            user_prompt = random.choice(user_prompt_config)
        else:
            u_temp = random.choice(user_prompt_config.get("templates", ["Describe this image."]))
            u_slots = user_prompt_config.get("slots", {})
            user_prompt = u_temp.format_map(ContextAwareSlotDict(u_slots, subcategory, category))

    # 拼接
    suffix = "ASSISTANT RESPONSE (Follow the mandatory opening style):" if is_refusal_mode else "ASSISTANT RESPONSE:"
    full_prompt = f"{sys_instruction}\n\nUSER REQUEST: {user_prompt}\n\n{suffix}"
        
    return user_prompt, full_prompt

def get_template_reason_candidate(prompt_data, subcategory, category):
    """用于在模型生成失败时，获取备用的模板理由"""
    if isinstance(prompt_data, dict) and "slots" in prompt_data:
        slots = prompt_data["slots"]
        if subcategory and f"reason_{subcategory}" in slots: return random.choice(slots[f"reason_{subcategory}"])
        if category and f"reason_{category}" in slots: return random.choice(slots[f"reason_{category}"]) # [重点] 回退到大类
        if "reason_default" in slots: return random.choice(slots["reason_default"])
    return "it violates safety guidelines."

# ===========================================================
# Module C: 模型推理
# ===========================================================

def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0]/r[1]))
    target_width, target_height = image_size * best_ratio[0], image_size * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    blocks = best_ratio[0] * best_ratio[1]
    for i in range(blocks):
        box = ((i % best_ratio[0]) * image_size, (i // best_ratio[0]) * image_size, ((i % best_ratio[0]) + 1) * image_size, ((i // best_ratio[0]) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def init_model(model_path):
    warnings.filterwarnings('ignore')
    print(f"🔄 Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
    return tokenizer, model

def chat_inference(model, tokenizer, full_prompt, full_image_path=None, temperature=0.9):
    try:
        pixel_values = None
        if full_image_path:
            if not os.path.exists(full_image_path): return "Error: Image Missing"
            image = Image.open(full_image_path).convert("RGB")
            transform = build_transform(input_size=448)
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
        
        generation_config = dict(
            max_new_tokens=512,
            do_sample=True, temperature=temperature, top_p=0.95, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
        )
        response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
        return response
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return f"Error: {e}"

# ===========================================================
# Module D: 主逻辑
# ===========================================================

GLOBAL_DATA = []
OUTPUT_PATH = ""

def save_checkpoint(signum=None, frame=None):
    print(f"\n💾 Saving {len(GLOBAL_DATA)} entries to {OUTPUT_PATH}...")
    if GLOBAL_DATA:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)
    try:
        swanlab.finish()
    except: pass
    sys.exit(0)

def main():
    global GLOBAL_DATA, OUTPUT_PATH
    args = parse_args()
    OUTPUT_PATH = args.output_json
    signal.signal(signal.SIGINT, save_checkpoint)

    # SwanLab 初始化
    run_name = f"{args.category}_{args.img_prefix}_{args.data_type}"
    swanlab.init(project="InternVL_Data_Gen", name=run_name, config=args)

    # 1. 统一数据源处理
    items_to_process = []
    
    if args.data_type == "image":
        if not args.image_dir: print("❌ Image mode requires --image_dir"); return
        files = get_and_rename_images(args.image_dir, args.img_prefix)
        for f in files:
            base_id = os.path.splitext(f)[0]
            items_to_process.append({"id": base_id, "content": f, "type": "image", "category": args.category, "subcategory": args.img_prefix})
            
    elif args.data_type == "text":
        if not args.text_file: print("❌ Text mode requires --text_file"); return
        raw_texts = load_text_inputs(args.text_file)
        for idx, t in enumerate(raw_texts):
            # 支持纯字符串列表或对象列表
            if isinstance(t, dict):
                base_id = t.get("id", f"{args.img_prefix}_text_{idx+1}")
                content = t.get("text", "")
                cat = t.get("category", args.category) # 优先用数据里的 category
                sub = t.get("subcategory", args.img_prefix)
            else:
                base_id = f"{args.img_prefix}_text_{idx+1}"
                content = t
                cat = args.category
                sub = args.img_prefix
            
            items_to_process.append({"id": base_id, "content": content, "type": "text", "category": cat, "subcategory": sub})

    # 加载状态与模型
    GLOBAL_DATA, processed_ids = load_existing_progress(args.output_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tokenizer, model = init_model(args.model_path)
    
    # 动态加载 Prompt
    all_prompts = load_prompts_from_file(args.prompt_config)
    prefix = "image" if args.data_type == "image" else "text"
    prompts_db = {
        'refusal_user': all_prompts.get(f'{prefix}_refusal_user', {}),
        'refusal_sys':  all_prompts.get(f'{prefix}_refusal_system', {}),
        'normal_user':  all_prompts.get(f'{prefix}_normal_user', {}),
        'normal_sys':   all_prompts.get(f'{prefix}_normal_system', {})
    }

    processed_count = 0
    total_items = len(items_to_process)

    # 2. 循环处理
    for idx, item in enumerate(items_to_process, start=1):
        if item["id"] in processed_ids: continue
        
        # 准备变量
        full_image_path = os.path.join(args.image_dir, item["content"]) if item["type"] == "image" else None
        text_content = item["content"] if item["type"] == "text" else None
        cur_cat = item["category"]
        cur_sub = item["subcategory"]

        # ---------------- Normal Generation ----------------
        if args.mode in ["mixed", "normal"]:
            u_prompt, full_input = construct_full_input(
                prompts_db['normal_user'], prompts_db['normal_sys'], 
                is_refusal_mode=False, subcategory=cur_sub, category=cur_cat,
                text_input=text_content
            )
            resp = chat_inference(model, tokenizer, full_input, full_image_path)
            
            entry = {
                "id": f"{item['id']}_normal",
                "conversations": [{"from": "human", "value": u_prompt}, {"from": "gpt", "value": resp}],
                "label": "normal", "category": cur_cat, "subcategory": cur_sub
            }
            if item["type"] == "image": entry["image"] = os.path.join(args.json_prefix, item["content"])
            GLOBAL_DATA.append(entry)
            swanlab.log({"normal/resp_len": len(resp)})

        # ---------------- Refusal Generation ----------------
        if args.mode in ["mixed", "refusal"]:
            
            # 2.1 理由生成 (Reason Generation)
            if item["type"] == "image":
                reason_q = f"Analyze this image. Briefly explain specifically why generating content about it might be restricted due to safety guidelines regarding {cur_sub}."
            else:
                # 纯文本：让模型解释为什么这句话不能回
                reason_q = (f"Analyze the following user request: '{text_content}'.\n"
                            f"Provide a brief, single-sentence explanation of why answering this request is restricted "
                            f"under safety policies regarding {cur_sub} ({cur_cat}).\n"
                            f"Start your explanation with 'it' or 'this request'.")

            # 让模型尝试生成理由
            draft_reason = chat_inference(model, tokenizer, reason_q, full_image_path, temperature=0.5)
            draft_reason_clean = draft_reason.replace("\n", " ").strip().strip('"')

            # 2.2 验证与回退 (Validation & Fallback)
            # 简单的关键词检查：如果模型生成了拒绝开头（I cannot），说明它误解了指令，这时强制回退到模板
            refusal_keywords = ["apologize", "cannot", "unable", "sorry"]
            is_valid_reason = (len(draft_reason_clean) > 10) and not any(k in draft_reason_clean.lower() for k in refusal_keywords)

            if is_valid_reason:
                final_reason = draft_reason_clean
                source = "MODEL"
            else:
                # 获取模板理由（自动支持大类回退）
                final_reason = get_template_reason_candidate(prompts_db['refusal_sys'], cur_sub, cur_cat)
                source = "TEMPLATE"

            # 2.3 生成拒绝回复
            u_prompt, full_input = construct_full_input(
                prompts_db['refusal_user'], prompts_db['refusal_sys'], 
                is_refusal_mode=True, subcategory=cur_sub, category=cur_cat,
                forced_reason=final_reason, text_input=text_content
            )
            resp = chat_inference(model, tokenizer, full_input, full_image_path)
            
            entry = {
                "id": f"{item['id']}_sensitive",
                "conversations": [{"from": "human", "value": u_prompt}, {"from": "gpt", "value": resp}],
                "label": "sensitive", "category": cur_cat, "subcategory": cur_sub, "reason_source": source
            }
            if item["type"] == "image": entry["image"] = os.path.join(args.json_prefix, item["content"])
            GLOBAL_DATA.append(entry)
            
            swanlab.log({"refusal/source": 1 if source == "MODEL" else 0})

        processed_count += 1
        if processed_count % 10 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)

    save_checkpoint()

if __name__ == "__main__":
    main()