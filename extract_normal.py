import os
import json
import random
import glob
from collections import defaultdict

# ================= 配置区域 =================
BASE_DATA_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"

# 输出保存路径 (建议修改为一个新文件夹以免覆盖之前 sensitive 的数据)
OUTPUT_DIR = "/home/disk/q3s2/data_generation/Our_Dataset/final_dataset/20260330_normal"

RANDOM_SEED = 42
# ===========================================

def load_and_merge(base_root):
    all_data = []
    seen_ids = set()
    
    # 统计字典：大类 -> 子类 -> 数量 (变量名改为 normal)
    stats_normal = defaultdict(lambda: defaultdict(int))
    stats_skipped = defaultdict(lambda: defaultdict(int))

    print(f"📂 开始遍历目录: {base_root}")
    print(f"🔍 寻找路径模式: */json/*.json") # 【修改1 & 修改2】提示信息更新

    for category in os.listdir(base_root):
        cat_path = os.path.join(base_root, category)
        if not os.path.isdir(cat_path):
            continue
        
        # 【修改1】提取的文件夹改为 json 目录
        json_dir = os.path.join(cat_path, "json")
        if not os.path.exists(json_dir):
            continue

        # 【修改2】寻找 json 目录下的所有 .json 文件，不限制 _valid 后缀
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            
            # 【修改3】交互式询问：每次提取前自行决定是否提取
            print("-" * 50)
            user_choice = input(f"❓ 发现待提取文件: [{category}/json/{file_name}]\n   是否提取此文件? (y/n, 默认y): ").strip().lower()
            
            if user_choice == 'n':
                print(f"⏭️ 已跳过提取: {file_name}")
                continue # 直接跳过当前文件，进入下一个循环
            
            # 提取子分类名称，去掉 .json 后缀
            subcategory = file_name.replace(".json", "") 
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if not isinstance(data, list):
                        print(f"❌ 格式错误 (非列表): {json_file}")
                        continue

                    count_normal = 0
                    count_skipped = 0
                    
                    for item in data:
                        current_label = item.get('label')
                        
                        # 【核心修改】目标改为提取 'normal'
                        if current_label != 'normal':
                            count_skipped += 1
                            continue
                        
                        # 修复绝对路径
                        if 'image' in item and item['image']:
                            img_path = item['image']
                            if not os.path.isabs(img_path):
                                item['image'] = os.path.join(BASE_DATA_ROOT, img_path)
                        
                        item_id = item.get('id', str(item)) 
                        
                        if item_id in seen_ids:
                            continue
                        
                        seen_ids.add(item_id)
                        all_data.append(item)
                        count_normal += 1
                    
                    # 记录统计信息
                    if count_normal > 0 or count_skipped > 0:
                        stats_normal[category][subcategory] += count_normal
                        stats_skipped[category][subcategory] += count_skipped

                    print(f"✅ 成功从 {file_name} 中提取了 {count_normal} 条 normal 数据。")

            except Exception as e:
                print(f"❌ 读取失败 {json_file}: {e}")

    return all_data, stats_normal, stats_skipped

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    merged_data, stats_normal, stats_skipped = load_and_merge(BASE_DATA_ROOT)
    
    # 统计全局 normal 数量
    total_normal = len(merged_data)
    total_skipped = sum(sum(sub_dict.values()) for sub_dict in stats_skipped.values())
    
    print("=" * 60)
    print(f"📊 数据合并与过滤全局总结:")
    print(f"  ✅ 成功提取 'normal' 总量 : {total_normal}")
    print(f"  ⏭️ 被跳过的总数据量 (非 normal) : {total_skipped}")
    print("-" * 60)
    
    all_categories = set(stats_normal.keys()).union(set(stats_skipped.keys()))
    
    if all_categories:
        print("📈 各类别详细分布树 (提取 normal / 跳过其他):")
        for cat in sorted(all_categories):
            cat_extracted_total = sum(stats_normal[cat].values())
            cat_skipped_total = sum(stats_skipped[cat].values())
            print(f" 📁 [大类] {cat:<15} | 提取总计: {cat_extracted_total:<5} | 跳过总计: {cat_skipped_total:<5}")
            
            all_subcats = set(stats_normal[cat].keys()).union(set(stats_skipped[cat].keys()))
            subcat_list = sorted(list(all_subcats))
            
            for i, subcat in enumerate(subcat_list):
                extracted = stats_normal[cat].get(subcat, 0)
                skipped = stats_skipped[cat].get(subcat, 0)
                
                branch_char = "└─" if i == len(subcat_list) - 1 else "├─"
                print(f"    {branch_char} [子类] {subcat:<25} : 提取了 {extracted:<5} | 跳过了 {skipped:<5}")
                
    print("-" * 60)

    if total_normal == 0:
        print("❌ 未找到有效且 label 为 'normal' 的数据，或者您选择跳过了所有文件。")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(merged_data)

    split_index = int(total_normal * 0.8)
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