import subprocess
import os
import time

# ================= 配置区域 =================
# 请确保这些路径与你之前的配置一致
BASE_DATA_ROOT = "/home/disk/q3s2/data_generation/Our_Dataset/images"
MODEL_PATH = "/home/disk/q3s2/evaluate_models/InternVL3_5-8B"

# 脚本文件名
GEN_SCRIPT = "run_scheduler.py"
AUDIT_SCRIPT = "run_audit.py"

# 测试用的 GPU
TEST_GPU = "0"

# 测试用的任务 (选择一个存在的文件夹)
# 假设 hazardous_labels 文件夹存在，如果不存在请修改为其他存在的文件夹
TEST_CAT = "VCP"
TEST_FOLDER = "hazaradous_labels" 
TEST_MODE = "mixed"

# ================= 验证逻辑 =================

def run_command(cmd, step_name):
    print(f"\n{'='*50}")
    print(f"🧪 [TEST] 开始验证步骤: {step_name}")
    print(f"📝 命令: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    start = time.time()
    try:
        # 实时输出日志，方便你看报错
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ [PASS] {step_name} 验证成功! (耗时: {time.time() - start:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [FAIL] {step_name} 验证失败! (Exit Code: {e.returncode})")
        return False
    except Exception as e:
        print(f"\n❌ [ERROR] {step_name} 发生异常: {str(e)}")
        return False

def main():
    print("🚀 开始环境与代码冒烟测试...\n")

    # 1. 检查路径是否存在
    img_dir = os.path.join(BASE_DATA_ROOT, TEST_CAT, TEST_FOLDER)
    prompt_path = os.path.join(BASE_DATA_ROOT, TEST_CAT, "prompt", "all_prompts.json")
    
    if not os.path.exists(img_dir):
        print(f"❌ 错误: 图片目录不存在: {img_dir}")
        print("💡 建议: 请修改脚本中的 TEST_FOLDER 为一个真实存在的目录。")
        return

    # 2. 构造生成路径 (Output)
    gen_output_json = os.path.join("test_output", "gen_result.json") # 临时输出路径
    os.makedirs("test_output", exist_ok=True)
    json_img_prefix = f"{TEST_CAT}/{TEST_FOLDER}/"

    # =======================================================
    # STEP 1: 验证生成脚本 (gen_dataset.py)
    # =======================================================
    gen_cmd = [
        "python", GEN_SCRIPT,
        "--mode", TEST_MODE,
        "--image_dir", img_dir,
        "--img_prefix", TEST_FOLDER,
        "--output_json", gen_output_json,
        "--json_prefix", json_img_prefix,
        "--category", TEST_CAT,
        "--prompt_config", prompt_path,
        "--model_path", MODEL_PATH,
        "--gpu_id", TEST_GPU,
        "--sim_threshold", "0.3",
        "--max_samples", "2"  # <--- 关键：只生成 2 条，速度快
    ]

    if not run_command(gen_cmd, "生成数据 (Generation)"):
        print("⛔ 测试终止：生成步骤失败。")
        return

    # 检查生成结果是否存在
    if not os.path.exists(gen_output_json):
        print(f"❌ 错误: 生成脚本运行成功，但未找到输出文件: {gen_output_json}")
        return

    # =======================================================
    # STEP 2: 验证评估脚本 (audit_pipeline.py)
    # =======================================================
    audit_output_dir = "test_output/audit_results"
    
    audit_cmd = [
        "python", AUDIT_SCRIPT,
        "--input_json", gen_output_json, # 使用刚才生成的文件作为输入
        "--image_root", img_dir,
        "--model_path", MODEL_PATH,
        "--output_dir", audit_output_dir,
        "--gpu_id", TEST_GPU,
        "--task_name", "test_run",
        "--max_samples", "2" # <--- 关键：只评估 2 条
    ]

    if run_command(audit_cmd, "评估数据 (Audit)"):
        print("\n🎉🎉🎉 [SUCCESS] 所有代码验证通过！系统准备就绪。")
        print(f"📂 测试结果已保存在 ./test_output 目录下，你可以去检查一下内容是否符合预期。")
        print("👉 现在你可以放心地运行 run_all_script.py 了。")

if __name__ == "__main__":
    main()