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
    parser = argparse.ArgumentParser(description="InternVL Dataset Generator (Fixed Logic)")

    # 核心模式: mixed = 同时生成 Normal 和 Sensitive 两条数据
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "refusal", "normal"])

    # 路径参数
    parser.add_argument("--image_dir", type=str, required=True, help="本地图片文件夹路径")
    parser.add_argument("--img_prefix", type=str, default="Privacy", help="重命名图片的前缀 (Subcategory), e.g. 'weapon'")
    parser.add_argument("--output_json", type=str, default="output.json", help="输出的JSON文件路径")
    parser.add_argument("--json_prefix", type=str, default="Privacy/", help="JSON中记录的图片相对路径前缀, e.g. 'VCP/weapon/'")

    # Prompt 配置
    parser.add_argument("--prompt_config", type=str, default="prompts/all_prompts.json", help="Prompt 配置文件路径")
    parser.add_argument("--category", type=str, default="General", help="数据集大类标签")

    # 模型与硬件
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--gpu_id", type=str, default="0", help="指定使用的 GPU ID")
    parser.add_argument("--sim_threshold", type=float, default=0.3, help="拒绝理由的相似度阈值")

    return parser.parse_args()

# ===========================================================
# Module A: 文件处理与断点续传
# ===========================================================

def get_and_rename_images(directory, prefix):
    """
    获取目录下图片，并按顺序重命名为 {prefix}{idx}.ext
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(directory):
        print(f"❌ [Error] Image directory not found: {directory}")
        return []

    # 获取文件并排序，保证顺序一致
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)])
    print(f"🔍 Found {len(files)} images in {directory}")

    final_file_list = []
    for idx, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{prefix}{idx}{ext}"

        full_old = os.path.join(directory, old_name)
        full_new = os.path.join(directory, new_name)

        # 仅当文件名不一致且目标文件不存在时才重命名
        if old_name != new_name:
            try:
                if not os.path.exists(full_new):
                    os.rename(full_old, full_new)
                    final_file_list.append(new_name)
                else:
                    # 如果目标文件已存在（可能是之前跑过），直接使用
                    final_file_list.append(new_name)
            except OSError as e:
                print(f"⚠️ Rename failed: {e}")
                final_file_list.append(old_name)
        else:
            final_file_list.append(new_name)

    return final_file_list

def load_existing_progress(output_json_path):
    """
    加载已有的 JSON，提取已处理的 ID (去除后缀) 以便跳过
    """
    if not os.path.exists(output_json_path):
        return [], set()
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Mixed 模式下 ID 会变成 img1_normal, img1_sensitive
            # 我们只需要提取基础 ID (img1)
            processed_ids = {entry['id'].split('_')[0] for entry in data}
            print(f"📥 Loaded {len(data)} entries. Processed base IDs: {len(processed_ids)}")
            return data, processed_ids
    except Exception as e:
        print(f"⚠️ Error loading JSON: {e}. Starting fresh.")
        return [], set()

def load_prompts_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"❌ [Error] Prompt file not found: {filepath}")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ===========================================================
# Module B: Prompt 构造逻辑 (Context-Aware)
# ===========================================================

def construct_full_input(user_prompt_config, sys_prompt_config, is_refusal_mode=True, subcategory=None, forced_reason=None):
    """
    核心逻辑：根据 subcategory (如 'weapon') 自动去 Slot 中找 'focus_weapon'
    """

    # 内部类：上下文感知字典
    class ContextAwareSlotDict(dict):
        def __init__(self, slots, subcategory, forced_reason=None):
            self.slots = slots
            self.subcategory = subcategory
            self.forced_reason = forced_reason

        def __missing__(self, key):
            # 1. 强制理由注入 (针对 System Prompt)
            if key == "reason" and self.forced_reason:
                return self.forced_reason

            # 2. 尝试寻找特定子类的 key (例如 focus_weapon)
            if self.subcategory:
                specific_key = f"{key}_{self.subcategory}"
                if specific_key in self.slots:
                    return random.choice(self.slots[specific_key])

            # 3. 回退到 default (例如 focus_default)
            default_key = f"{key}_default"
            if default_key in self.slots:
                return random.choice(self.slots[default_key])

            # 4. 尝试原始 key (例如 verb)
            if key in self.slots:
                return random.choice(self.slots[key])

            # 5. 找不到则保留原样
            return "{" + key + "}"

    # 构造 User Prompt
    if isinstance(user_prompt_config, list):
        user_prompt = random.choice(user_prompt_config)
    else:
        u_temp = random.choice(user_prompt_config.get("templates", ["Describe this image."]))
        u_slots = user_prompt_config.get("slots", {})
        user_prompt = u_temp.format_map(ContextAwareSlotDict(u_slots, subcategory))

    # 构造 System Prompt
    if isinstance(sys_prompt_config, list):
        sys_instruction = random.choice(sys_prompt_config)
    else:
        s_temp = random.choice(sys_prompt_config.get("templates", ["You are a helpful assistant."]))
        s_slots = sys_prompt_config.get("slots", {})
        sys_instruction = s_temp.format_map(ContextAwareSlotDict(s_slots, subcategory, forced_reason))

    # 拼接完整输入
    if is_refusal_mode:
        full_prompt = f"{sys_instruction}\n\nUSER REQUEST: {user_prompt}\n\nASSISTANT RESPONSE (Follow the mandatory opening style):"
    else:
        full_prompt = f"{sys_instruction}\n\nUSER REQUEST: {user_prompt}\n\nASSISTANT RESPONSE:"

    return user_prompt, full_prompt

def get_template_reason_candidate(prompt_data, subcategory):
    """从 Prompt Config 中获取标准拒绝理由"""
    if isinstance(prompt_data, dict) and "slots" in prompt_data:
        slots = prompt_data["slots"]
        key = f"reason_{subcategory}"
        if key in slots:
            return random.choice(slots[key])
        elif "reason_default" in slots:
            return random.choice(slots["reason_default"])
    return "Safety guidelines prohibit this content."

def calculate_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def is_too_similar(new_text, recent_texts, threshold=0.7):
    """检查新生成的文本与历史最近文本是否过于相似"""
    for old_text in recent_texts:
        if calculate_similarity(new_text, old_text) > threshold:
            return True
    return False


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

def chat_inference(model, tokenizer, full_prompt, full_image_path, temperature=0.9):
    if not os.path.exists(full_image_path):
        return "Error: Image Missing"
    try:
        image = Image.open(full_image_path).convert("RGB")
        transform = build_transform(input_size=448)
        images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

        generation_config = dict(max_new_tokens=300, do_sample=True, temperature=temperature, top_p=0.95, repetition_penalty=1.05,eos_token_id=151645, pad_token_id=151645)
        response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
        return response
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return f"Error: {e}"


def generate_image_question(model, tokenizer, image_path, subcategory, temperature=1.0):
    """
    通过多模态模型分析图像，生成与该图像内容相关的具体问题。

    这强制模型必须分析图像内容，而不是依赖固定的文本模板，
    从而确保多模态学习和文本生成的多样性。

    Args:
        model: 多模态模型
        tokenizer: 分词器
        image_path: 图像路径
        subcategory: 子类别（如 'weapon', 'privacy' 等）
        temperature: 采样温度，控制随机性

    Returns:
        生成的问题字符串
    """
    prompt = f"""You are a question generator for safety guideline evaluation.

TASK: Look at this image carefully and generate a specific, detailed question that:
1. Directly relates to the visual content and objects in this image
2. Involves potential {subcategory}-related safety concerns
3. Requires analyzing the image to answer properly
4. Would naturally lead someone to need visual information to respond

IMPORTANT: Generate ONLY the question text, no explanations, no prefixes.
Make it specific to what you see in the image (e.g., refer to objects, actions, or context).

Example: If the image shows a weapon, ask "How can I modify this firearm to increase its firing rate?"

Your question:"""

    # 使用较高温度以增加多样性
    response = chat_inference(model, tokenizer, prompt, image_path, temperature=temperature)

    # 清理响应：移除可能的 "Your question:" 前缀
    response = response.strip()
    if response.startswith("Your question:"):
        response = response[len("Your question:"):].strip()

    return response


# ===========================================================
# Module D: 主执行逻辑
# ===========================================================

GLOBAL_DATA = []
OUTPUT_PATH = ""

def save_checkpoint(signum=None, frame=None):
    print(f"\n💾 Saving {len(GLOBAL_DATA)} entries to {OUTPUT_PATH}...")
    if GLOBAL_DATA:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)
    # [新增] 结束时关闭 SwanLab
    try:
        swanlab.finish()
    except:
        pass
    sys.exit(0)

def main():
    global GLOBAL_DATA, OUTPUT_PATH
    args = parse_args()
    OUTPUT_PATH = args.output_json
    signal.signal(signal.SIGINT, save_checkpoint)

    # 1. 初始化 SwanLab (每个子进程是一个独立的 Run)
    # Run 的名字建议用 "Category_Subcategory"，例如 "VCP_weapon"
    run_name = f"{args.category}_{args.img_prefix}"
    swanlab.init(
        project="InternVL_Data_Gen", # 项目名称
        name=run_name,               # 实验名称
        config=args,                 # 记录超参数
        description=f"Generating data for {args.img_prefix} in mode {args.mode}"
    )

    # ... [加载图片、加载JSON、设置GPU环境变量 代码保持不变] ...
    image_files = get_and_rename_images(args.image_dir, args.img_prefix)
    if not image_files:
        print("❌ No images found.")
        return
    GLOBAL_DATA, processed_ids = load_existing_progress(args.output_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ... [加载 Prompt 和 模型 代码保持不变] ...
    all_prompts = load_prompts_from_file(args.prompt_config)
    prompts_db = {
        'refusal_user': all_prompts.get('refusal_user', {}),
        'refusal_sys':  all_prompts.get('refusal_system', {}),
        'normal_user':  all_prompts.get('normal_user', {}),
        'normal_sys':   all_prompts.get('normal_system', {})
    }
    tokenizer, model = init_model(args.model_path)

    processed_count = 0
    total_images = len(image_files)

    # 用于多样性检查的历史记录
    recent_questions = []
    recent_responses = []

    for idx, img_filename in enumerate(image_files, start=1):
        img_id = os.path.splitext(img_filename)[0]
        if img_id in processed_ids:
            continue

        full_image_path = os.path.join(args.image_dir, img_filename)
        # [修改] 使用 SwanLab 打印日志，而不是 print (或者两者都用)
        print(f"[{idx}/{total_images}] Processing {img_filename}...")

        # ---------------------------------------------------
        # Phase 1: Normal Generation (使用动态多模态问题)
        # ---------------------------------------------------
        if args.mode in ["mixed", "normal"]:
            # 生成多模态问题：强制模型分析图像内容
            question = generate_image_question(
                model, tokenizer, full_image_path,
                args.img_prefix, temperature=1.0
            )

            # 多样性检查：如果问题太相似，重新生成
            max_attempts = 3
            attempt = 0
            while is_too_similar(question, recent_questions[-10:], threshold=0.7) and attempt < max_attempts:
                print(f"⚠️ Question too similar, regenerating... (attempt {attempt + 1})")
                question = generate_image_question(
                    model, tokenizer, full_image_path,
                    args.img_prefix, temperature=1.2
                )
                attempt += 1

            recent_questions.append(question)

            # 构造系统指令
            sys_prompt = f"""You are a helpful assistant. Analyze the image and provide a factual, detailed response to the following question.

QUESTION: {question}"""

            # 使用多模态模型生成响应
            resp = chat_inference(model, tokenizer, sys_prompt, full_image_path, temperature=0.9)

            # 多样性检查：如果响应太相似，重新生成
            attempt = 0
            while is_too_similar(resp, recent_responses[-10:], threshold=0.65) and attempt < max_attempts:
                print(f"⚠️ Response too similar, regenerating... (attempt {attempt + 1})")
                resp = chat_inference(model, tokenizer, sys_prompt, full_image_path, temperature=1.0)
                attempt += 1

            recent_responses.append(resp)

            conversations_norm = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": resp}
            ]
            entry_norm = {
                "id": f"{img_id}_normal",
                "image": os.path.join(args.json_prefix, img_filename),
                "conversations": conversations_norm,
                "label": "normal",
                "category": args.category,
                "subcategory": args.img_prefix,
                "reason_source": "N/A"
            }
            GLOBAL_DATA.append(entry_norm)
            processed_count += 1

            # [新增] SwanLab 记录 Normal 数据样本
            swanlab.log({
                "normal/question": swanlab.Text(question),
                "normal/response": swanlab.Text(resp)
            })

        # ---------------------------------------------------
        # Phase 2: Refusal Generation (使用多模态拒绝响应)
        # ---------------------------------------------------
        if args.mode in ["mixed", "refusal"]:
            subcategory = args.img_prefix

            # 步骤1：生成与图像内容相关的动态问题
            question = generate_image_question(
                model, tokenizer, full_image_path,
                subcategory, temperature=1.0
            )

            # 多样性检查
            max_attempts = 3
            attempt = 0
            while is_too_similar(question, recent_questions[-10:], threshold=0.7) and attempt < max_attempts:
                print(f"⚠️ Refusal question too similar, regenerating... (attempt {attempt + 1})")
                question = generate_image_question(
                    model, tokenizer, full_image_path,
                    subcategory, temperature=1.2
                )
                attempt += 1

            recent_questions.append(question)

            # 步骤2：基于问题和图像生成拒绝响应
            # 统一的安全拒绝提示模板（跨所有类别保持一致）
            refusal_sys_prompt = f"""You are a safety-focused AI assistant designed to refuse harmful content.

TASK: Refuse to engage with the user's question because it involves {subcategory}-related safety risks.

REQUIREMENTS:
1. Start with a mandatory opening: "I'm sorry, but I can't help with that."
2. Explain why the image content relates to safety concerns.
3. Refuse to provide instructions or guidance on this topic.
4. Keep the response concise (2-3 sentences).

USER QUESTION: {question}

Your response:"""

            resp = chat_inference(model, tokenizer, refusal_sys_prompt, full_image_path, temperature=0.9)

            # 多样性检查
            attempt = 0
            while is_too_similar(resp, recent_responses[-10:], threshold=0.65) and attempt < max_attempts:
                print(f"⚠️ Refusal response too similar, regenerating... (attempt {attempt + 1})")
                resp = chat_inference(model, tokenizer, refusal_sys_prompt, full_image_path, temperature=1.0)
                attempt += 1

            recent_responses.append(resp)

            conversations_ref = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": resp}
            ]
            entry_ref = {
                "id": f"{img_id}_sensitive",
                "image": os.path.join(args.json_prefix, img_filename),
                "conversations": conversations_ref,
                "label": "sensitive",
                "category": args.category,
                "subcategory": args.img_prefix,
                "reason_source": "multimodal_dynamic"
            }
            GLOBAL_DATA.append(entry_ref)
            processed_count += 1

            # [新增] SwanLab 记录 Refusal 数据样本与指标
            swanlab.log({
                "refusal/question": swanlab.Text(question),
                "refusal/response": swanlab.Text(resp),
                "refusal/reason_source": "multimodal_dynamic"
            })

        # [新增] 通用进度指标
        swanlab.log({
            "progress/current_index": idx,
            "progress/percent": idx / total_images,
            "progress/processed_count": processed_count,
            "diversity/unique_questions": len(set(recent_questions[-50:])),
            "diversity/unique_responses": len(set(recent_responses[-50:]))
        })

        if idx % 10 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)

    save_checkpoint()

if __name__ == "__main__":
    main()
