import subprocess
import os
import sys
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# Step 1: 全局通用配置
# ==============================================================================
BASE_DATA_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"
MODEL_PATH = "/home/disk/q3s2/evaluate_models/InternVL3_5-8B"
TARGET_SCRIPT = "./gen_dataset.py"
DRY_RUN = False

# [新增] 可用的 GPU 列表
AVAILABLE_GPUS = ["2", "3"] 

# ==============================================================================
# Step 2: 任务列表 (无需手动指定 GPU，会自动分配)
# ==============================================================================
tasks = [
    {"category": "VCP", "folder": "animal_abuse",       "mode": "mixed"},
    {"category": "VCP", "folder": "hazaradous_labels",       "mode": "mixed"},
    {"category": "PCP", "folder": "logos",      "mode": "mixed"},
    # {"category": "VCP", "folder": "bully",              "mode": "mixed"},
    # {"category": "VCP", "folder": "drugs",              "mode": "mixed"},
    # {"category": "VCP", "folder": "gamble",             "mode": "mixed"},
    # {"category": "VCP", "folder": "terrorism",          "mode": "mixed"},
    # {"category": "VCP", "folder": "theft",              "mode": "mixed"},
    # {"category": "VCP", "folder": "violence",           "mode": "mixed"},
    # {"category": "VCP", "folder": "weapon",             "mode": "mixed"},
    # {"category": "SM",  "folder": "explicit_gestures",  "mode": "mixed"},
    # {"category": "SM",  "folder": "nudity",             "mode": "mixed"},
    # {"category": "SM",  "folder": "sexual_acts",        "mode": "mixed"},
    # {"category": "RC",  "folder": "religious_bias",     "mode": "mixed"},
    # {"category": "RC",  "folder": "religious_scenes",   "mode": "mixed"},
    # {"category": "PR",  "folder": "demonstrations",     "mode": "mixed"},
    # {"category": "PR",  "folder": "politician",         "mode": "mixed"},
    # {"category": "DC",  "folder": "hardware_tampering", "mode": "mixed"},
    # {"category": "DC",  "folder": "malware_ransomware", "mode": "mixed"},
    # {"category": "DC",  "folder": "phishing",           "mode": "mixed"},
    # {"category": "DC",  "folder": "website_hijacking",  "mode": "mixed"},
    # {"category": "PCP", "folder": "contract",           "mode": "mixed"},
    # {"category": "PCP", "folder": "PII",                "mode": "mixed"},
    # {"category": "PCP", "folder": "privacy_leaks",      "mode": "mixed"},
    # {"category": "MP",  "folder": "negative_implications", "mode": "mixed"},
    # {"category": "MP",  "folder": "self_injured",       "mode": "mixed"},
    # {"category": "FDI",  "folder": "academic_dishonesty", "mode": "mixed"},
    # {"category": "FDI",  "folder": "bad_ads",            "mode": "mixed"},
    # {"category": "FDI",  "folder": "financial",          "mode": "mixed"},
]

# ==============================================================================
# Step 3: 并行执行逻辑
# ==============================================================================

# 初始化 GPU 队列
gpu_queue = queue.Queue()
for gpu in AVAILABLE_GPUS:
    gpu_queue.put(gpu)

def process_task(task):
    """
    单个任务的处理函数，运行在独立的线程中
    """
    cat = task["category"]
    folder = task["folder"]
    
    # 1. 获取 GPU (如果没有可用 GPU，这里会阻塞等待，直到有其他任务归还)
    gpu_id = gpu_queue.get()
    
    try:
        # 2. 自动构建路径
        img_dir = os.path.join(BASE_DATA_ROOT, cat, folder)
        output_json = os.path.join(BASE_DATA_ROOT, cat, "src", f"{folder}.json")
        prompt_config_path = os.path.join(BASE_DATA_ROOT, cat, "prompt", "all_prompts.json")
        json_img_prefix = f"{cat}/{folder}/"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        print(f"🟢 [Start] GPU {gpu_id} -> {cat}/{folder}")

        # 检查必要文件
        if not os.path.exists(img_dir):
            return f"❌ Skip: 图片目录不存在 {img_dir}"
        if not os.path.exists(prompt_config_path):
            return f"❌ Skip: Prompt配置不存在 {prompt_config_path}"

        # 3. 组装命令
        cmd = [
            "python", TARGET_SCRIPT,
            "--mode", task["mode"],
            "--image_dir", img_dir,
            "--img_prefix", folder,
            "--output_json", output_json,
            "--json_prefix", json_img_prefix,
            "--category", cat,
            "--prompt_config", prompt_config_path,
            "--model_path", MODEL_PATH,
            "--gpu_id", gpu_id,   # 使用动态获取的 GPU ID
            "--sim_threshold", "0.3"
        ]

        # 4. 执行命令
        if DRY_RUN:
            time.sleep(1) # 模拟运行时间
            return f"✅ [Dry Run] {cat}/{folder} on GPU {gpu_id}"
        else:
            # 这里的 subprocess.run 会阻塞当前线程，直到 Python 脚本跑完
            # 因为我们在 ThreadPoolExecutor 里，所以主程序不会被阻塞
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL) # stdout=DEVNULL 防止多个任务日志混在一起刷屏
            return f"✅ [Done] {cat}/{folder} finished on GPU {gpu_id}"

    except subprocess.CalledProcessError as e:
        return f"❌ [Fail] {cat}/{folder} failed on GPU {gpu_id} (Exit Code: {e.returncode})"
    except Exception as e:
        return f"❌ [Error] {cat}/{folder}: {str(e)}"
    finally:
        # [关键] 无论任务成功还是失败，都要把 GPU 还给队列
        gpu_queue.put(gpu_id)
        # print(f"♻️  GPU {gpu_id} released.")

def run_parallel_tasks():
    total_tasks = len(tasks)
    print(f"🚀 开始并行调度，共 {total_tasks} 个任务，使用 GPU: {AVAILABLE_GPUS}...\n")

    # 使用线程池，最大并发数 = GPU 数量
    # 注意：这里用 ThreadPoolExecutor 是因为 subprocess 主要是 IO 等待（等待显卡跑完），Python 本身不费 CPU
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_task, task): task for task in tasks}

        # 实时处理完成的任务
        for i, future in enumerate(as_completed(future_to_task), start=1):
            result_msg = future.result()
            print(f"[{i}/{total_tasks}] {result_msg}")

    print("\n🎉 所有任务并行处理完毕。")

if __name__ == "__main__":
    run_parallel_tasks()