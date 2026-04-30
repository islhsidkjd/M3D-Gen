import json
import os

# ================= 配置区域 =================
# 1. 输入目录 (merge_and_split.py 的输出目录)
INPUT_DIR = "/home/disk/q3s2/data_generation/Our_Dataset/final_dataset"

# 2. 输出目录
OUTPUT_DIR = "/home/disk/q3s2/data_generation/Our_Dataset/sharegpt_dataset"

# 3. [关键] 图片的根目录 (必须与你之前的 BASE_DATA_ROOT 保持一致)
# 脚本会将相对路径拼接到这个路径后面
IMAGE_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"

# 4. 角色映射 (ShareGPT/LLaVA 格式)
ROLE_MAPPING = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
    "human": "human",
    "gpt": "gpt"
}
# ===========================================

def to_absolute_path(path):
    """
    将路径转换为绝对路径
    """
    if not path:
        return ""
    
    # 如果已经是绝对路径 (以 / 开头)，直接返回
    if os.path.isabs(path):
        return path
    
    # 如果是相对路径，进行拼接
    full_path = os.path.join(IMAGE_ROOT, path)
    
    # 规范化路径 (去除多余的 //, ./ 等)
    return os.path.abspath(full_path)

def process_conversations(conversations):
    new_convs = []
    
    for idx, turn in enumerate(conversations):
        # 1. 角色映射
        original_role = turn.get('from', '').lower()
        new_role = ROLE_MAPPING.get(original_role, original_role)
        
        content = turn.get('value', '')

        # 2. 第一轮 Human 对话添加 <image> 占位符
        if idx == 0 and new_role == "human":
            if "<image>" not in content:
                content = "<image>\n" + content
        
        new_convs.append({
            "from": new_role,
            "value": content
        })
    
    return new_convs

def convert_file(input_path, output_path):
    print(f"🔄 正在处理: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_path}")
        return

    sharegpt_data = []
    processed_count = 0
    
    for item in data:
        # 必须包含 image 字段
        if 'image' not in item:
            continue

        raw_image_path = item.get('image')
        
        # [核心修改] 转换为绝对路径
        abs_image_path = to_absolute_path(raw_image_path)
        
        # 可选：检查文件是否存在 (这一步会稍微拖慢速度，但能保证数据质量)
        # if not os.path.exists(abs_image_path):
        #     print(f"⚠️ 警告: 图片不存在 {abs_image_path}")
        
        conversations = item.get('conversations', [])
        if not conversations:
            continue

        new_conversations = process_conversations(conversations)

        entry = {
            "id": str(item.get('id', '')),
            "image": abs_image_path,  # 这里写入的是绝对路径
            "conversations": new_conversations
        }
        
        sharegpt_data.append(entry)
        processed_count += 1

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存至: {output_path} (共 {processed_count} 条，均为绝对路径)")

def main():
    # 处理 train.json 和 test.json
    files_to_process = ["train.json", "test.json"]
    
    print(f"🔧 图片根目录设定为: {IMAGE_ROOT}")
    print("-" * 50)

    for filename in files_to_process:
        input_full = os.path.join(INPUT_DIR, filename)
        output_full = os.path.join(OUTPUT_DIR, filename)
        convert_file(input_full, output_full)
    
    print("-" * 50)
    print("🎉 转换完成。请检查输出文件中的 'image' 字段是否为绝对路径。")

if __name__ == "__main__":
    main()