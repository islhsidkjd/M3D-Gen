import subprocess
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# Step 1: 全局配置
# ==============================================================================
BASE_DATA_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"
MODEL_PATH = "/home/disk/q3s2/evaluate_models/Qwen3-VL-8B-Instruct"

# [关键] 指向新的 Final 脚本
TARGET_SCRIPT = "audit_final.py" 

DRY_RUN = False 
AVAILABLE_GPUS = ["2", "3"] 

# ==============================================================================
# Step 2: 完整任务列表 (34个)
# ==============================================================================
tasks = [
    # # DC
    # {"category": "DC", "folder": "hardware_tampering"},
    # # {"category": "DC", "folder": "malware_ransomware"},
    # # {"category": "DC", "folder": "phishing"},
    # {"category": "DC", "folder": "website_hijacking"},
    # # FDI
    # {"category": "FDI", "folder": "academic_dishonesty"},
    # {"category": "FDI", "folder": "bad_ads"},
    # {"category": "FDI", "folder": "financial"},
    # # # HDH
    # {"category": "HDH", "folder": "disablity_discrimination"},
    # {"category": "HDH", "folder": "gender_discrimination"},
    # {"category": "HDH", "folder": "geographical_discrimination"},
    # {"category": "HDH", "folder": "harass"},
    # {"category": "HDH", "folder": "hate"},
    # {"category": "HDH", "folder": "hatefulMemes"},
    # {"category": "HDH", "folder": "insult"},
    # {"category": "HDH", "folder": "racial"},
    # # MP
    # {"category": "MP", "folder": "negative_implications"},
    # {"category": "MP", "folder": "self_injured"},
    # # PCP
    # {"category": "PCP", "folder": "contract"},
    # {"category": "PCP", "folder": "PII"},
    # {"category": "PCP", "folder": "privacy_leaks"},
    # # PR
    # {"category": "PR", "folder": "demonstrations"},
    # {"category": "PR", "folder": "politician"},
    # # RC
    # {"category": "RC", "folder": "religious_bias"},
    # {"category": "RC", "folder": "religious_scenes"},
    # # SM
    # {"category": "SM", "folder": "explicit_gestures"},
    # {"category": "SM", "folder": "nudity"},
    # {"category": "SM", "folder": "sexual_acts"},
    # # VCP
    {"category": "VCP", "folder": "animal_abuse"},
    # {"category": "VCP", "folder": "bully"},
    # {"category": "VCP", "folder": "drugs"},
    # {"category": "VCP", "folder": "gamble"},
    # {"category": "VCP", "folder": "terrorism"},
    # {"category": "VCP", "folder": "theft"},
    # {"category": "VCP", "folder": "violence"},
    #  {"category": "VCP", "folder": "weapon"},
    # {"category": "VCP", "folder": "hazaradous_labels"},
    # {"category": "PCP", "folder": "logos"}
 ]

# ==============================================================================
# Step 3: 并行执行逻辑
# ==============================================================================

gpu_queue = queue.Queue()
for gpu in AVAILABLE_GPUS:
    gpu_queue.put(gpu)

def process_task(task):
    cat = task["category"]
    folder = task["folder"] 
    
    gpu_id = gpu_queue.get()
    
    try:
        input_json = os.path.join(BASE_DATA_ROOT, cat, "json", f"{folder}.json")
        img_dir = os.path.join(BASE_DATA_ROOT, cat, folder)
        output_dir = os.path.join(BASE_DATA_ROOT, cat, "audit_results")
        os.makedirs(output_dir, exist_ok=True)

        print(f"🟢 [Start] GPU {gpu_id} -> {cat}/{folder}")

        if not os.path.exists(input_json):
            input_json_lower = os.path.join(BASE_DATA_ROOT, cat, "src", f"{folder.lower()}.json")
            if os.path.exists(input_json_lower):
                input_json = input_json_lower
            else:
                return f"❌ Skip: JSON文件不存在 {input_json}"

        if not os.path.exists(img_dir):
            return f"❌ Skip: 图片目录不存在 {img_dir}"

        cmd = [
            "python", TARGET_SCRIPT,
            "--input_json", input_json,
            "--image_root", img_dir,
            "--model_path", MODEL_PATH,
            "--output_dir", output_dir,
            "--gpu_id", gpu_id,
            "--task_name", folder,
            "--max_samples", "2000"
        ]

        if DRY_RUN:
            print(f"   [CMD] {' '.join(cmd)}")
            time.sleep(0.5) 
            return f"✅ [Dry Run] {cat}/{folder} on GPU {gpu_id}"
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL) 
            return f"✅ [Done] {cat}/{folder} finished on GPU {gpu_id}"

    except subprocess.CalledProcessError as e:
        return f"❌ [Fail] {cat}/{folder} crashed (Exit Code: {e.returncode})"
    except Exception as e:
        return f"❌ [Error] {cat}/{folder}: {str(e)}"
    finally:
        gpu_queue.put(gpu_id)

def run_parallel_tasks():
    total_tasks = len(tasks)
    print(f"🚀 开始并行审计 (SwanLab Enhanced Final), GPU: {AVAILABLE_GPUS}")
    
    with ThreadPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}

        for i, future in enumerate(as_completed(future_to_task), start=1):
            result_msg = future.result()
            print(f"[{i}/{total_tasks}] {result_msg}")

    print("\n🎉 所有任务结束。")

if __name__ == "__main__":
    run_parallel_tasks()