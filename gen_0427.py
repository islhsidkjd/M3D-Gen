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
    parser = argparse.ArgumentParser(description="Image-Aware Dataset Generator (Two-Stage Process)")

    # 模式：mixed = 同时生成 Normal 和 Sensitive 两条数据
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "refusal", "normal"])

    # 路径参数
    parser.add_argument("--image_dir", type=str, required=True, help="本地图片文件夹路径")
    parser.add_argument("--img_prefix", type=str, default="Privacy", help="重命名图片的前缀 (Subcategory), e.g. 'weapon'")
    parser.add_argument("--output_json", type=str, default="output_0427.json", help="输出的JSON文件路径")
    parser.add_argument("--json_prefix", type=str, default="Privacy/", help="JSON中记录的图片相对路径前缀")

    # Prompt 配置 - 使用新的两阶段配置
    parser.add_argument("--prompt_config", type=str, default="prompt_0427.json", help="两阶段 Prompt 配置文件路径")
    parser.add_argument("--category", type=str, default="General", help="数据集大类标签")

    # 模型与硬件
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--gpu_id", type=str, default="0", help="指定使用的 GPU ID")

    # 质量控制参数
    parser.add_argument("--question_temp", type=float, default=0.8, help="问题生成的 temperature")
    parser.add_argument("--refusal_temp", type=float, default=0.7, help="拒绝回答的 temperature")
    parser.add_argument("--max_retries", type=int, default=3, help="问题生成失败时的重试次数")

    return parser.parse_args()

# ===========================================================
# Module A: 文件处理与断点续传 (保持不变)
# ===========================================================

def get_and_rename_images(directory, prefix):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(directory):
        print(f"❌ [Error] Image directory not found: {directory}")
        return []

    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)])
    print(f"🔍 Found {len(files)} images in {directory}")

    final_file_list = []
    for idx, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{prefix}{idx}{ext}"

        full_old = os.path.join(directory, old_name)
        full_new = os.path.join(directory, new_name)

        if old_name != new_name:
            try:
                if not os.path.exists(full_new):
                    os.rename(full_old, full_new)
                    final_file_list.append(new_name)
                else:
                    final_file_list.append(new_name)
            except OSError as e:
                print(f"⚠️ Rename failed: {e}")
                final_file_list.append(old_name)
        else:
            final_file_list.append(new_name)

    return final_file_list

def load_existing_progress(output_json_path):
    if not os.path.exists(output_json_path):
        return [], set()
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
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

def construct_full_input(user_prompt_config, sys_prompt_config, is_refusal_mode=True,
                        subcategory=None, forced_reason=None, question=None, visual_element=None):
    """
    增强版：支持图片感知问题生成和拒绝

    参数:
        question: 阶段1生成的问题（用于阶段2的拒绝生成）
        visual_element: 图片中的视觉元素（用于拒绝生成的context）
    """

    class ContextAwareSlotDict(dict):
        def __init__(self, slots, subcategory, forced_reason=None, question=None, visual_element=None):
            self.slots = slots
            self.subcategory = subcategory
            self.forced_reason = forced_reason
            self.question = question
            self.visual_element = visual_element

        def __missing__(self, key):
            # 强制理由注入
            if key == "reason" and self.forced_reason:
                return self.forced_reason

            # 阶段2专用：注入问题
            if key == "user_question" and self.question:
                return self.question

            # 阶段2专用：注入视觉元素
            if key == "visual_element" and self.visual_element:
                return self.visual_element

            # 子类别特定的slot
            if self.subcategory:
                specific_key = f"{key}_{self.subcategory}"
                if specific_key in self.slots:
                    return random.choice(self.slots[specific_key])

            # 默认slot
            default_key = f"{key}_default"
            if default_key in self.slots:
                return random.choice(self.slots[default_key])

            # 原始key
            if key in self.slots:
                return random.choice(self.slots[key])

            return "{" + key + "}"

    # 构造 User Prompt
    if isinstance(user_prompt_config, list):
        user_prompt = random.choice(user_prompt_config)
    else:
        u_temp = random.choice(user_prompt_config.get("templates", ["Describe this image."]))
        u_slots = user_prompt_config.get("slots", {})
        try:
            user_prompt = u_temp.format_map(ContextAwareSlotDict(u_slots, subcategory, forced_reason, question, visual_element))
        except:
            user_prompt = u_temp

    # 构造 System Prompt
    if isinstance(sys_prompt_config, list):
        sys_instruction = random.choice(sys_prompt_config)
    else:
        s_temp = random.choice(sys_prompt_config.get("templates", ["You are a helpful assistant."]))
        s_slots = sys_prompt_config.get("slots", {})
        try:
            sys_instruction = s_temp.format_map(ContextAwareSlotDict(s_slots, subcategory, forced_reason, question, visual_element))
        except:
            sys_instruction = s_temp

    # 拼接完整输入
    if is_refusal_mode:
        full_prompt = f"{sys_instruction}\n\nUSER REQUEST: {user_prompt}\n\nASSISTANT RESPONSE (Follow the mandatory opening style):"
    else:
        full_prompt = f"{sys_instruction}\n\nUSER REQUEST: {user_prompt}\n\nASSISTANT RESPONSE:"

    return user_prompt, full_prompt

# ===========================================================
# Module C: 模型推理 (保持不变)
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

        generation_config = dict(max_new_tokens=300, do_sample=True, temperature=temperature, top_p=0.95, repetition_penalty=1.05, eos_token_id=151645, pad_token_id=151645)
        response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
        return response
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return f"Error: {e}"

# ===========================================================
# Module D: 两阶段流程实现
# ===========================================================

def generate_image_aware_question(model, tokenizer, image_path, subcategory, prompt_config, temperature=0.8):
    """
    阶段1：生成与图片高度契合的敏感问题

    参数:
        prompt_config: 包含 question_generation 配置
        temperature: 控制多样性
    返回:
        question: 生成的敏感问题
    """
    qg_config = prompt_config.get('question_generation', {})

    # 构建 System Prompt
    sys_config = qg_config.get('system_prompt', {})
    templates = sys_config.get('templates', [])
    if templates:
        sys_template = random.choice(templates)
        slots = sys_config.get('slots', {})
        # 使用 slot 变量填充
        try:
            subcat_options = slots.get('subcategory', [subcategory])
            filled_sys = sys_template.format(subcategory=random.choice(subcat_options))
        except:
            filled_sys = sys_template
    else:
        # 默认 System Prompt
        filled_sys = f"SYSTEM: Analyze this image carefully. Generate a specific question about the visual elements related to {subcategory} that would require safety refusal. The question must be tied to what is shown in the image."

    # 可选：从 question_templates 中选择
    qt_config = qg_config.get('question_templates', {})
    qt_templates = qt_config.get('templates', [])
    use_template = random.random() < 0.5 and qt_templates  # 50% 概率使用模板

    if use_template:
        question_template = random.choice(qt_templates)
        slots = qt_config.get('slots', {})
        try:
            action = random.choice(slots.get('action', ['do']))
            visual_elem = random.choice(slots.get('visual_element', ['this element']))
            question = question_template.format(action=action, visual_element=visual_elem)
        except:
            question = question_template
    else:
        # 自由生成：构建 minimal prompt
        question = None  # 让模型自由生成

    # 构建完整 prompt
    if question:
        full_prompt = f"{filled_sys}\n\nQuestion: {question}"
    else:
        full_prompt = filled_sys

    # 推理生成问题
    generated_question = chat_inference(model, tokenizer, full_prompt, image_path, temperature=temperature)

    # 简单清洗：去除多余空格和换行
    generated_question = generated_question.strip().replace('\n', ' ')

    return generated_question

def generate_refusal_with_question(model, tokenizer, image_path, question, subcategory, prompt_config, temperature=0.7):
    """
    阶段2：基于生成的问题进行多模态拒绝

    参数:
        question: 阶段1生成的问题
        prompt_config: 包含 question_refusal 配置
    """
    qr_config = prompt_config.get('question_refusal', {})
    sys_config = qr_config.get('system_prompt', {})

    # 构建 System Prompt
    templates = sys_config.get('templates', [])
    if templates:
        sys_template = random.choice(templates)
        slots = sys_config.get('slots', {})

        # 选择 opening_style
        opening_styles = slots.get('opening_style', ['I cannot fulfill this request.'])
        opening = random.choice(opening_styles)

        # 选择或构建 reason
        reason_slot = slots.get('reason', ['providing detailed instructions for {subcategory} poses safety risks'])
        reason_template = random.choice(reason_slot)
        reason = reason_template.format(subcategory=subcategory)

        # 尝试提取视觉元素（简单处理：从question中提取名词短语）
        visual_element = "the content shown"  # 简化处理
        if 'weapon' in question.lower() or 'firearm' in question.lower():
            visual_element = "the weapon shown"
        elif 'drug' in question.lower() or 'narcotic' in question.lower():
            visual_element = "the illegal substances visible"
        elif 'explosive' in question.lower() or 'bomb' in question.lower():
            visual_element = "the explosive material depicted"

        try:
            filled_sys = sys_template.format(
                opening_style=opening,
                reason=reason,
                subcategory=subcategory,
                user_question=question,
                visual_element=visual_element
            )
        except:
            filled_sys = sys_template
    else:
        # 默认 System Prompt
        filled_sys = f"SYSTEM: You are a safety-focused AI assistant.\nINSTRUCTION: The user's question relates to {subcategory} and references visual elements in the image.\nRefuse to provide actionable instructions.\nStart with: I cannot fulfill this request.\nReason: This request involves {subcategory} which is restricted."

    # 构建完整 prompt
    full_prompt = f"{filled_sys}\n\nUSER: {question}\n\nASSISTANT:"

    # 推理生成拒绝回答
    response = chat_inference(model, tokenizer, full_prompt, image_path, temperature=temperature)

    return response, filled_sys

# ===========================================================
# Module E: 主执行逻辑 (修改版)
# ===========================================================

GLOBAL_DATA = []
OUTPUT_PATH = ""
PROMPT_CONFIG = None

def save_checkpoint(signum=None, frame=None):
    print(f"\n💾 Saving {len(GLOBAL_DATA)} entries to {OUTPUT_PATH}...")
    if GLOBAL_DATA:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)
    try:
        swanlab.finish()
    except:
        pass
    sys.exit(0)

def main():
    global GLOBAL_DATA, OUTPUT_PATH, PROMPT_CONFIG
    args = parse_args()
    OUTPUT_PATH = args.output_json
    signal.signal(signal.SIGINT, save_checkpoint)

    # 1. 初始化 SwanLab
    run_name = f"{args.category}_{args.img_prefix}_0427"
    swanlab.init(
        project="InternVL_ImageAware_Gen",
        name=run_name,
        config=args,
        description=f"Two-stage image-aware generation for {args.img_prefix}"
    )

    # 2. 加载图片
    image_files = get_and_rename_images(args.image_dir, args.img_prefix)
    if not image_files:
        print("❌ No images found.")
        return

    GLOBAL_DATA, processed_ids = load_existing_progress(args.output_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 3. 加载 Prompt 配置（新格式）
    all_prompts = load_prompts_from_file(args.prompt_config)
    PROMPT_CONFIG = all_prompts

    # 兼容旧格式（如果存在）
    prompts_db = {
        'refusal_user': all_prompts.get('refusal_user', {}),
        'refusal_sys': all_prompts.get('refusal_system', {}),
        'normal_user': all_prompts.get('normal_user', {}),
        'normal_sys': all_prompts.get('normal_system', {})
    }

    # 4. 加载模型
    tokenizer, model = init_model(args.model_path)

    processed_count = 0
    total_images = len(image_files)

    print(f"🚀 Starting two-stage generation for {total_images} images...")
    print(f"   Mode: {args.mode}")
    print(f"   Subcategory: {args.img_prefix}")
    print(f"   Category: {args.category}")

    for idx, img_filename in enumerate(image_files, start=1):
        img_id = os.path.splitext(img_filename)[0]
        if img_id in processed_ids:
            print(f"⏭️  Skipping {img_filename} (already processed)")
            continue

        full_image_path = os.path.join(args.image_dir, img_filename)
        print(f"\n🖼️  [{idx}/{total_images}] Processing: {img_filename}")

        # ============================================
        # Phase 1: Normal Data (Optional)
        # ============================================
        if args.mode in ["mixed", "normal"]:
            print("   📝 Generating Normal data...")
            # 使用原有逻辑或新逻辑
            if 'normal_generation' in all_prompts:
                # 新格式：图片感知正常问题
                ng_config = all_prompts['normal_generation']
                sys_cfg = ng_config.get('system_prompt', {})
                q_templates = ng_config.get('question_templates', {}).get('templates', [])
                q_slots = ng_config.get('question_templates', {}).get('slots', {})

                # 构建正常问题
                if q_templates:
                    question = random.choice(q_templates)
                else:
                    question = "Describe the content shown in this image."

                # 构建 system prompt
                if sys_cfg.get('templates'):
                    sys_template = random.choice(sys_cfg['templates'])
                    sys_prompt = sys_template
                else:
                    sys_prompt = "SYSTEM: You are a helpful AI assistant. Analyze the image and answer accurately."

                full_prompt = f"{sys_prompt}\n\nUSER: {question}\n\nASSISTANT:"
                resp = chat_inference(model, tokenizer, full_prompt, full_image_path, temperature=args.refusal_temp)
            else:
                # 兼容旧格式
                u_prompt, full_input = construct_full_input(
                    prompts_db['normal_user'], prompts_db['normal_sys'],
                    is_refusal_mode=False, subcategory=args.img_prefix
                )
                resp = chat_inference(model, tokenizer, full_input, full_image_path)

            conversations_norm = [
                {"from": "human", "value": question if 'question' in locals() else u_prompt},
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
            swanlab.log({
                "normal/response": swanlab.Text(resp),
                "normal/question": question if 'question' in locals() else u_prompt
            })

        # ============================================
        # Phase 2: Refusal Generation (Two-Stage)
        # ============================================
        if args.mode in ["mixed", "refusal"]:
            print("   🔍 Stage 1: Generating image-aware question...")

            # 阶段1：生成敏感问题
            question = generate_image_aware_question(
                model, tokenizer, full_image_path, args.img_prefix,
                all_prompts, temperature=args.question_temp
            )
            print(f"      Generated question: {question[:100]}...")

            # 简单验证：问题长度和相关性
            if len(question) < 10:
                print(f"      ⚠️ Question too short ({len(question)} chars), skipping")
                continue

            print("   🚫 Stage 2: Generating refusal response...")

            # 阶段2：基于问题生成拒绝回答
            refusal_response, sys_prompt_used = generate_refusal_with_question(
                model, tokenizer, full_image_path, question,
                args.img_prefix, all_prompts, temperature=args.refusal_temp
            )
            print(f"      Refusal response: {refusal_response[:100]}...")

            # 提取视觉元素标记 (简化)
            visual_element = "the content shown"
            for keyword in ['weapon', 'firearm', 'gun']:
                if keyword in question.lower():
                    visual_element = f"the {keyword} shown"
                    break
            for keyword in ['drug', 'narcotic', 'substance']:
                if keyword in question.lower():
                    visual_element = "the substances visible"
                    break
            for keyword in ['explosive', 'bomb']:
                if keyword in question.lower():
                    visual_element = "the explosive material depicted"
                    break

            # 构建拒绝数据条目
            conversations_ref = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": refusal_response}
            ]
            entry_ref = {
                "id": f"{img_id}_sensitive",
                "image": os.path.join(args.json_prefix, img_filename),
                "conversations": conversations_ref,
                "label": "sensitive",
                "category": args.category,
                "subcategory": args.img_prefix,
                "reason_source": "IMAGE_AWARE_TWO_STAGE",
                "visual_element": visual_element,
                "generation_stages": {
                    "stage1_question": question,
                    "stage2_refusal": refusal_response,
                    "stage2_system_prompt": sys_prompt_used
                }
            }
            GLOBAL_DATA.append(entry_ref)
            processed_count += 1

            # SwanLab 记录
            swanlab.log({
                "refusal/question": swanlab.Text(question),
                "refusal/response": swanlab.Text(refusal_response),
                "refusal/question_length": len(question),
                "refusal/response_length": len(refusal_response),
                "progress/processed_count": processed_count
            })

        # 每处理10张图片保存一次
        if idx % 10 == 0:
            checkpoint_file = args.output_json.replace('.json', f'_checkpoint_{idx}.json')
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(GLOBAL_DATA, f, ensure_ascii=False, indent=2)
            print(f"   ✅ Checkpoint saved: {checkpoint_file}")

    save_checkpoint()

if __name__ == "__main__":
    main()