import subprocess
import time

def run_script(script_name):
    print(f"🚀 [Master] 开始执行: {script_name}")
    start_time = time.time()
    
    try:
        # check=True 表示如果脚本报错（返回码不为0），则抛出异常，停止后续步骤
        subprocess.run(["python", script_name], check=True)
        print(f"✅ [Master] {script_name} 执行完毕，耗时 {time.time() - start_time:.2f}秒\n")
    except subprocess.CalledProcessError:
        print(f"❌ [Master] {script_name} 执行失败！已终止后续任务。")
        exit(1) # 退出总控脚本

if __name__ == "__main__":
    print("="*40)
    print("🎬 全流程启动")
    print("="*40)

    # 1. 运行生成脚本
    run_script("run_scheduler.py")

    # 2. 运行评估脚本
    run_script("run_audit.py")

    print("="*40)
    print("🎉 所有步骤执行完毕！")