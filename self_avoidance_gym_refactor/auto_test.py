import subprocess
import csv
import re
import os
import sys
import numpy as np

def main():
    # 测试的模型及其对应的权重文件名
    # models_and_checkpoints = {
    #     "SAC": "SAC_only_mean_good.pt",
    #     "SACWithoutGate": "without_gate.pt",
    #     "SACWithoutTransformerAndGate": "without_transformer_and_gate.pt",
    #     "PointTransSAC": "PointTransSAC.pt",
    #     "SetTransSAC": "SetTransSAC.pt",
    #     "DPRL": "DPRL.pt"
    # }
    models_and_checkpoints = {
        "withQ2": "withQ2.pt"
    }

    # 测试的环境等级
    levels = [5, 8, 10, 15]
    # 在每个等级下重复测试的轮次
    rounds = 10
    
    # 结果输出文件
    csv_file = "baseline_benchmark_results_withQ2.csv"

    # 匹配控制台输出数据的正则表达式
    success_re = re.compile(r"成功[：:]\s*(\d+)")
    collision_re = re.compile(r"碰撞[：:]\s*(\d+)")
    power_re = re.compile(r"耗尽能量[：:]\s*(\d+)")
    over_re = re.compile(r"超过步长[：:]\s*(\d+)")

    print(f"=== 开始全自动化批量评估 ===")
    print(f"结果将实时写入: {csv_file}")
    print("=" * 40)

    # 准备 csv 写入流程（加上 encoding 避免乱码）
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["Model", "EnvLevel", "Round", "Success", "Collision", "PowerEmpty", "StepOver"])

        for model, ckpt in models_and_checkpoints.items():
            for lvl in levels:
                # 记录该等级下所有轮次的各自指标，以供计算均值和方差
                stats_s, stats_c, stats_p, stats_o = [], [], [], []
                
                for r in range(1, rounds + 1):
                    print(f"正在运行 -> 模型: {model:<30} | 难度: {lvl:<2} | 轮次: {r}/{rounds} ...", end="", flush=True)
                    
                    # 组合 hydra 命令行参数
                    cmd = [
                        sys.executable, "test/test.py",
                        f"navigationModel={model}",
                        f"SACloadModel={ckpt}",
                        f"envLevel={lvl}",
                        "testEpisodes=100",
                        "testUavNums=10"
                    ]
                    
                    try:
                        # 开启子进程，并捕获标准输出和标准错误
                        # 使用 bytes 模式截获，并通过系统环境变量强制指定 Python 输出为 utf-8，解决 Windows 下的控制台乱码导致正则失效问题
                        env = os.environ.copy()
                        env["PYTHONIOENCODING"] = "utf-8"
                        
                        result = subprocess.run(
                            cmd, 
                            capture_output=True,
                            env=env
                        )
                        
                        # 全部使用 UTF-8 安全解码
                        stdout_text = result.stdout.decode('utf-8', errors='ignore')
                        stderr_text = result.stderr.decode('utf-8', errors='ignore')
                        
                        # 开始正则解析
                        s, c, p, o = 0, 0, 0, 0
                        matched = False
                        
                        if m := success_re.search(stdout_text): 
                            s = int(m.group(1))
                            matched = True
                        if m := collision_re.search(stdout_text): c = int(m.group(1))
                        if m := power_re.search(stdout_text): p = int(m.group(1))
                        if m := over_re.search(stdout_text): o = int(m.group(1))
                        
                        if matched:
                            # 成功解析，写入CSV并刷新流
                            writer.writerow([model, lvl, r, s, c, p, o])
                            f.flush()
                            print(f" 完成 -> 成功:{s}\t碰撞:{c}\t耗能:{p}\t越步:{o}")
                            # 添加到记录列表
                            stats_s.append(s)
                            stats_c.append(c)
                            stats_p.append(p)
                            stats_o.append(o)
                        else:
                            # 没找正则结果表示程序可能发生错误了
                            print(f" 失败!\n[stderr]: {result.stderr.strip()[:200]}...")
                            
                    except Exception as e:
                        print(f" 子进程启动或执行异常: {e}")

                # 当前等级的 rounds 测试完成，计算均值和标准差
                if stats_s:
                    mean_s, std_s = np.mean(stats_s), np.std(stats_s)
                    mean_c, std_c = np.mean(stats_c), np.std(stats_c)
                    mean_p, std_p = np.mean(stats_p), np.std(stats_p)
                    mean_o, std_o = np.mean(stats_o), np.std(stats_o)
                    
                    # 按照审稿人要求的格式 [mean±std] 记录
                    str_s = f"[{mean_s:.2f}±{std_s:.2f}]"
                    str_c = f"[{mean_c:.2f}±{std_c:.2f}]"
                    str_p = f"[{mean_p:.2f}±{std_p:.2f}]"
                    str_o = f"[{mean_o:.2f}±{std_o:.2f}]"
                    
                    writer.writerow([model, lvl, "Mean±Std", str_s, str_c, str_p, str_o])
                    # 用一个空行作为分隔
                    writer.writerow([])
                    f.flush()
                    print(f"[{model} - 等级 {lvl} 汇总] -> 成功:{str_s} 碰撞:{str_c} 耗能:{str_p} 越步:{str_o}")
                    print("-" * 50)

    print("=" * 40)
    print(f"所有测试已自动化完成, 数据已保存至 {csv_file}")

if __name__ == "__main__":
    main()