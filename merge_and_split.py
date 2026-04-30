import os
import json
import random
import glob
from collections import defaultdict

# ================= 配置区域 =================
# 必须与 run_audit.py 中的 BASE_DATA_ROOT 保持一致
BASE_DATA_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"

# 输出保存路径
OUTPUT_DIR = "/home/disk/q3s2/data_generation/Our_Dataset/final_dataset/20260309"

# 随机种子，保证每次划分结果一致
RANDOM_SEED = 42
# ===========================================

def load_and_merge(base_root):
    all_data = []
    seen_ids = set()
    
    # 将单一统计字典改为嵌套字典：大类 -> 子类 -> 数量
    stats_sensitive = defaultdict(lambda: defaultdict(int))
    stats_skipped = defaultdict(lambda: defaultdict(int))

    print(f"📂 开始遍历目录: {base_root}")
    print(f"🔍 寻找路径模式: */audit_results/*_valid.json")

    for category in os.listdir(base_root):
        cat_path = os.path.join(base_root, category)
        if not os.path.isdir(cat_path):
            continue
        
        audit_dir = os.path.join(cat_path, "audit_results")
        if not os.path.exists(audit_dir):
            continue

        json_files = glob.glob(os.path.join(audit_dir, "*_valid.json"))
        
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            # 提取子分类名称，例如从 "hardware_tampering_valid.json" 提取出 "hardware_tampering"
            subcategory = file_name.replace("_valid.json", "") 
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if not isinstance(data, list):
                        print(f"❌ 格式错误 (非列表): {json_file}")
                        continue

                    count_sensitive = 0
                    count_skipped = 0
                    
                    for item in data:
                        current_label = item.get('label')
                        
                        if current_label != 'sensitive':
                            count_skipped += 1
                            continue
                        
                        # ==========================================
                        # 【新增】将 image 字段转换为绝对路径
                        # ==========================================
                        if 'image' in item and item['image']:
                            img_path = item['image']
                            # 如果当前路径不是绝对路径，则与 BASE_DATA_ROOT 拼接
                            if not os.path.isabs(img_path):
                                item['image'] = os.path.join(BASE_DATA_ROOT, img_path)
                        # ==========================================
                        
                        item_id = item.get('id', str(item)) 
                        
                        if item_id in seen_ids:
                            continue
                        
                        seen_ids.add(item_id)
                        all_data.append(item)
                        count_sensitive += 1
                    
                    # 按照 大类[子类] 的结构记录统计信息
                    if count_sensitive > 0 or count_skipped > 0:
                        stats_sensitive[category][subcategory] += count_sensitive
                        stats_skipped[category][subcategory] += count_skipped

            except Exception as e:
                print(f"❌ 读取失败 {json_file}: {e}")

    return all_data, stats_sensitive, stats_skipped

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("-" * 60)
    merged_data, stats_sensitive, stats_skipped = load_and_merge(BASE_DATA_ROOT)
    
    # 计算总数时，需要遍历嵌套字典
    total_sensitive = len(merged_data)
    total_skipped = sum(sum(sub_dict.values()) for sub_dict in stats_skipped.values())
    
    print("-" * 60)
    print(f"📊 数据合并与过滤全局总结:")
    print(f"  ✅ 成功提取 'sensitive' 总量 : {total_sensitive}")
    print(f"  ⏭️ 被跳过的总数据量 (非 sensitive) : {total_skipped}")
    print("-" * 60)
    
    print("📈 各类别详细分布树 (提取 sensitive / 跳过其他):")
    # 获取所有出现过的大类
    all_categories = set(stats_sensitive.keys()).union(set(stats_skipped.keys()))
    
    for cat in sorted(all_categories):
        # 计算当前大类的总和
        cat_extracted_total = sum(stats_sensitive[cat].values())
        cat_skipped_total = sum(stats_skipped[cat].values())
        print(f" 📁 [大类] {cat:<15} | 提取总计: {cat_extracted_total:<5} | 跳过总计: {cat_skipped_total:<5}")
        
        # 获取当前大类下的所有子类
        all_subcats = set(stats_sensitive[cat].keys()).union(set(stats_skipped[cat].keys()))
        subcat_list = sorted(list(all_subcats))
        
        # 树状图格式打印子类
        for i, subcat in enumerate(subcat_list):
            extracted = stats_sensitive[cat].get(subcat, 0)
            skipped = stats_skipped[cat].get(subcat, 0)
            
            # 使用树状符号，如果是最后一个子类就用 └─，否则用 ├─
            branch_char = "└─" if i == len(subcat_list) - 1 else "├─"
            print(f"    {branch_char} [子类] {subcat:<25} : 提取了 {extracted:<5} | 跳过了 {skipped:<5}")
            
    print("-" * 60)

    if total_sensitive == 0:
        print("❌ 未找到有效且 label 为 'sensitive' 的数据。")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(merged_data)

    split_index = int(total_sensitive * 0.8)
    train_data = merged_data[:split_index]
    test_data = merged_data[split_index:]

    print(f"✂️  划分数据集 (80/20):")
    print(f"  🟢 训练集 (Train): {len(train_data)}")
    print(f"  🔵 测试集 (Test) : {len(test_data)}")

    paths = {
        "train": os.path.join(OUTPUT_DIR, "train.json"),
        "test": os.path.join(OUTPUT_DIR, "test.json"),
        "all": os.path.join(OUTPUT_DIR, "all_merged.json")
    }

    print("-" * 60)
    for name, path in paths.items():
        data_to_save = train_data if name == "train" else (test_data if name == "test" else merged_data)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"💾 已保存 {name}: {path}")

    print("\n🎉 全部处理完成。")

if __name__ == "__main__":
    main()