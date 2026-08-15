"""
=============================================================================
模型复杂度与推理时间基准测试脚本 (Model Complexity & Inference Latency Benchmark)
=============================================================================

用途:
    根据审稿人意见 R1-Q9（M3-VI-D-01..03），系统性地测量以下指标：
    - 参数量 (Params, K)
    - 理论计算量 (FLOPs, M)
    - GPU 推理延迟 (ms)
    - CPU 推理延迟 (ms)

覆盖模型变体:
    1. MTrans-SAC Full (SAC.py)           — 完整模型
    2. MTrans-SAC w/ Hetero-Q (differentQ.py) — 异构双Q
    3. w/o Gating (SACwithoutGate.py)     — 消融：无门控融合
    4. Vanilla SAC (SACwithoutTransformerAndGate.py) — 消融：无Transformer+无门控
    5. DPRL (baselineDPRL.py)             — 1D-CNN基线
    6. PointTransSAC (baselinePointTransSAC.py) — 可学习位置编码 Transformer 基线
    7. SetTransSAC (baselineSetTransSAC.py) — 置换不变 Transformer 基线
    8. DDPG (DDPG.py)                     — DDPG基线

对每个模型变体，分别测量:
    - StateProcessor（状态编码器）子模块
    - RadarProcessor（雷达编码器）子模块
    - GatedFusion（门控融合/特征拼接）子模块（如适用）
    - Actor（完整策略网络）
    - Critic（完整价值网络，含 Q1+Q2）
    - Total Inference（仅Actor推理，实际部署所需）

硬件:
    GPU = NVIDIA RTX 3090 24GB; CPU = Intel i9-12900K (single thread)
    如无GPU则仅测CPU。

输出:
    - 控制台格式化表格
    - benchmark_results.csv

依赖:
    pip install torch torchinfo thop

用法:
    cd self_avoidance_gym_refactor
    python benchmark_model_complexity.py
    python benchmark_model_complexity.py --cpu-only    # 仅CPU模式
    python benchmark_model_complexity.py --runs 200    # 自定义重复次数
=============================================================================
"""

import sys
import os
import time
import math
import argparse
import warnings
from collections import OrderedDict

# 将当前目录加入路径，确保能导入 navigation 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# 尝试导入第三方库
try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("[WARN] torchinfo 未安装，将使用手动参数计数。安装: pip install torchinfo")

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[WARN] thop 未安装，FLOPs 将估算为 N/A。安装: pip install thop")

warnings.filterwarnings("ignore")

# =============================================================================
# 配置参数
# =============================================================================
BATCH_SIZE = 1           # 推理批量大小
STATE_DIM = 7            # 无人机状态维度
RADAR_DIM = 512          # 雷达波束数
ACTION_DIM = 3           # 动作维度
DIFFICULTY = 0.5         # 归一化难度等级 [0, 1]
WARMUP_RUNS = 10         # 预热次数
BENCH_RUNS = 100         # 正式测量次数
FP32 = True              # 使用 FP32 精度

# =============================================================================
# 工具函数
# =============================================================================

def count_params(model: nn.Module) -> int:
    """统计模型总参数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_thop(model: nn.Module, *inputs, **kwargs) -> float:
    """使用 thop 估算 FLOPs（MACs × 2）"""
    if not HAS_THOP:
        return float('nan')
    try:
        macs, _ = thop_profile(model, inputs=inputs, verbose=False, **kwargs)
        return macs * 2  # FLOPs ≈ 2 × MACs
    except Exception:
        return float('nan')


class LatencyBenchmark:
    """推理延迟测量器"""

    def __init__(self, device: torch.device, warmup: int = 10, runs: int = 100):
        self.device = device
        self.warmup = warmup
        self.runs = runs

    def measure(self, model: nn.Module, forward_fn) -> float:
        """
        测量模型前向传播延迟（毫秒）
        :param model: 已设置 eval() 的模型
        :param forward_fn: 无参可调用对象，执行一次前向传播
        :return: 平均延迟（ms）
        """
        model.eval()
        is_gpu = next(model.parameters()).device.type == 'cuda'

        # 预热
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = forward_fn()

        # 同步 + 计时
        if is_gpu:
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.runs):
                _ = forward_fn()
        if is_gpu:
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        return (t_end - t_start) / self.runs * 1000.0  # ms


# =============================================================================
# 模型构建工厂
# =============================================================================

def _get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_mtrans_sac_actor(device: torch.device):
    """MTrans-SAC 完整 Actor（SAC.py）"""
    from navigation.SAC import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_mtrans_sac_critic(device: torch.device):
    """MTrans-SAC 完整 Critic（SAC.py）"""
    from navigation.SAC import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_mtrans_state_processor(device: torch.device):
    """StateProcessor 子模块（SAC.py）"""
    from navigation.SAC import StateProcessor
    return StateProcessor(inputDim=STATE_DIM, hiddenDim=64, outputDim=32).to(device)


def make_mtrans_radar_transformer(device: torch.device):
    """RadarTransformer 子模块（SAC.py）"""
    from navigation.SAC import RadarTransformer
    return RadarTransformer(inputDim=RADAR_DIM, featureDim=64, outputDim=32).to(device)


def make_mtrans_gated_fusion(device: torch.device):
    """GatedFusion 子模块（SAC.py）"""
    from navigation.SAC import GatedFusion
    return GatedFusion(stateDim=32, radarDim=32, fusedDim=64).to(device)


def make_heteroq_actor(device: torch.device):
    """异构双Q SAC Actor（differentQ.py）"""
    from navigation.differentQ import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_heteroq_critic(device: torch.device):
    """异构双Q SAC Critic（differentQ.py）"""
    from navigation.differentQ import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_wogate_actor(device: torch.device):
    """w/o Gating Actor（SACwithoutGate.py）"""
    from navigation.SACwithoutGate import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_wogate_critic(device: torch.device):
    """w/o Gating Critic（SACwithoutGate.py）"""
    from navigation.SACwithoutGate import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_vanilla_sac_actor(device: torch.device):
    """Vanilla SAC Actor（SACwithoutTransformerAndGate.py, MLP+concat）"""
    from navigation.SACwithoutTransformerAndGate import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_vanilla_sac_critic(device: torch.device):
    """Vanilla SAC Critic（SACwithoutTransformerAndGate.py）"""
    from navigation.SACwithoutTransformerAndGate import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_vanilla_radar_mlp(device: torch.device):
    """Vanilla SAC 的 RadarMLP 子模块"""
    from navigation.SACwithoutTransformerAndGate import RadarMLP
    return RadarMLP(inputDim=RADAR_DIM, hiddenDim=256, outputDim=32).to(device)


def make_dprl_actor(device: torch.device):
    """DPRL Actor（baselineDPRL.py, 1D-CNN + GatedFusion + SAC）"""
    from navigation.baselineDPRL import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_dprl_critic(device: torch.device):
    """DPRL Critic（baselineDPRL.py）"""
    from navigation.baselineDPRL import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_dprl_cnn_processor(device: torch.device):
    """DPRL 1D-CNN 雷达处理器子模块"""
    from navigation.baselineDPRL import DPRLCNNProcessor
    return DPRLCNNProcessor(inputDim=RADAR_DIM, featureDim=64, outputDim=32).to(device)


def make_pointtrans_actor(device: torch.device):
    """PointTransSAC Actor（baselinePointTransSAC.py, 可学习PE + MHSA池化）"""
    from navigation.baselinePointTransSAC import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_pointtrans_critic(device: torch.device):
    """PointTransSAC Critic"""
    from navigation.baselinePointTransSAC import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_pointtrans_radar(device: torch.device):
    """PointTransSAC MHSARadarTransformer 子模块"""
    from navigation.baselinePointTransSAC import MHSARadarTransformer
    return MHSARadarTransformer(inputDim=RADAR_DIM, featureDim=64, outputDim=32).to(device)


def make_settrans_actor(device: torch.device):
    """SetTransSAC Actor（baselineSetTransSAC.py, 无PE + PMA池化）"""
    from navigation.baselineSetTransSAC import Actor
    return Actor(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_settrans_critic(device: torch.device):
    """SetTransSAC Critic"""
    from navigation.baselineSetTransSAC import Critic
    return Critic(stateDim=STATE_DIM, radarDim=RADAR_DIM, actionDim=ACTION_DIM).to(device)


def make_settrans_radar(device: torch.device):
    """SetTransSAC SetTransRadarTransformer 子模块"""
    from navigation.baselineSetTransSAC import SetTransRadarTransformer
    return SetTransRadarTransformer(inputDim=RADAR_DIM, featureDim=64, outputDim=32).to(device)


def make_ddpg_actor(device: torch.device):
    """DDPG Actor（DDPG.py）"""
    from navigation.DDPG import Actor as DDPGActor
    return DDPGActor(numIn=13, numOut=ACTION_DIM, hiddenDim=128,
                     activation=nn.PReLU(), outFunction=lambda x: torch.tanh(x) * 2.0).to(device)


def make_ddpg_critic(device: torch.device):
    """DDPG Critic（DDPG.py）"""
    from navigation.DDPG import Critic as DDPGCritic
    return DDPGCritic(numIn=24, numOut=1, hiddenDim=128,
                      activation=nn.PReLU(), outFunction=lambda x: x).to(device)


# =============================================================================
# 前向传播函数工厂
# =============================================================================

def forward_state_processor(model, device, state):
    def fn():
        return model(state)
    return fn


def forward_radar_transformer(model, device, radar):
    def fn():
        return model(radar)
    return fn


def forward_gated_fusion(model, device, state_f, radar_f):
    def fn():
        return model(state_f, radar_f, DIFFICULTY)
    return fn


def forward_sac_actor(model, device, state, radar):
    """SAC 系列 Actor 前向传播"""
    def fn():
        return model(state, radar, DIFFICULTY)
    return fn


def forward_sac_critic(model, device, state, radar, action):
    """SAC 系列 Critic 前向传播（同时输出Q1和Q2）"""
    def fn():
        return model(state, radar, action, DIFFICULTY)
    return fn


def forward_ddpg_actor(model, device, ddpg_input):
    """DDPG Actor 前向传播: input (1, 1, 13)"""
    def fn():
        return model(ddpg_input)
    return fn


def forward_ddpg_critic(model, device, critic_input):
    """DDPG Critic 前向传播: input (1, 24)"""
    def fn():
        return model(critic_input)
    return fn


# =============================================================================
# 单模型基准测试
# =============================================================================

def benchmark_module(device, make_fn, forward_fn_builder, prep_inputs_fn,
                     module_name, latency_bench):
    """
    对单个模块进行完整基准测试
    :return: dict {params_k, flops_m, gpu_ms, cpu_ms}
    """
    result = {"name": module_name, "params_k": 0, "flops_m": 0.0,
              "gpu_ms": 0.0, "cpu_ms": 0.0}

    try:
        model = make_fn(device)
        model.eval()

        # 参数量
        params = count_trainable_params(model)
        result["params_k"] = round(params / 1000.0, 2)

        # 准备输入
        inputs = prep_inputs_fn(device)
        fwd_fn = forward_fn_builder(model, device, *inputs)

        # GPU 延迟
        if device.type == 'cuda':
            gpu_lat = latency_bench.measure(model, fwd_fn)
            result["gpu_ms"] = round(gpu_lat, 3)

        # CPU 延迟（将模型和数据移到CPU）
        model_cpu = model.cpu()
        inputs_cpu = prep_inputs_fn(torch.device("cpu"))
        fwd_fn_cpu = forward_fn_builder(model_cpu, torch.device("cpu"), *inputs_cpu)
        cpu_lat = latency_bench.measure(model_cpu, fwd_fn_cpu)
        result["cpu_ms"] = round(cpu_lat, 3)

        # FLOPs estimation via thop
        # thop needs to trace model.forward() directly; SAC models need extra 
        # difficultyLevel arg which may not be in *inputs — try with and without
        try:
            flops = estimate_flops_thop(model, *inputs)
            if math.isnan(flops):
                # Retry with difficultyLevel added for SAC-style models
                diff_tensor = torch.tensor([DIFFICULTY], device=inputs[0].device)
                flops = estimate_flops_thop(model, *inputs, diff_tensor)
            if not math.isnan(flops):
                result["flops_m"] = round(flops / 1e6, 2)
        except Exception:
            pass

        # 清理
        del model, model_cpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [ERROR] {module_name}: {e}")
        result["error"] = str(e)

    return result


# =============================================================================
# 输入准备
# =============================================================================

def prep_state_input(device):
    return (torch.randn(BATCH_SIZE, STATE_DIM, device=device),)

def prep_radar_input(device):
    return (torch.randn(BATCH_SIZE, RADAR_DIM, device=device),)

def prep_state_radar_inputs(device):
    return (torch.randn(BATCH_SIZE, STATE_DIM, device=device),
            torch.randn(BATCH_SIZE, RADAR_DIM, device=device))

def prep_state_radar_action_inputs(device):
    return (torch.randn(BATCH_SIZE, STATE_DIM, device=device),
            torch.randn(BATCH_SIZE, RADAR_DIM, device=device),
            torch.randn(BATCH_SIZE, ACTION_DIM, device=device))

def prep_gate_inputs(device):
    # GatedFusion 需要预处理后的状态和雷达特征
    state_f = torch.randn(BATCH_SIZE, 32, device=device)
    radar_f = torch.randn(BATCH_SIZE, 32, device=device)
    return (state_f, radar_f)

def prep_ddpg_actor_input(device):
    return (torch.randn(BATCH_SIZE, 1, 13, device=device),)

def prep_ddpg_critic_input(device):
    return (torch.randn(BATCH_SIZE, 24, device=device),)


# =============================================================================
# 模型配置注册表
# =============================================================================

MODEL_CONFIGS = OrderedDict([
    # ---- MTrans-SAC Full ----
    ("1-MTransSAC-Full", {
        "type": "sac",
        "submodules": [
            ("StateProcessor", make_mtrans_state_processor, forward_state_processor, prep_state_input),
            ("RadarTransformer", make_mtrans_radar_transformer, forward_radar_transformer, prep_radar_input),
            ("GatedFusion", make_mtrans_gated_fusion, forward_gated_fusion, prep_gate_inputs),
        ],
        "actor": (make_mtrans_sac_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_mtrans_sac_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- MTrans-SAC w/ Hetero-Q ----
    ("2-HeteroQ-SAC", {
        "type": "sac",
        "submodules": [],  # 子模块同 Full，不再重复
        "actor": (make_heteroq_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_heteroq_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- w/o Gating ----
    ("3-w-o-Gating", {
        "type": "sac",
        "submodules": [],
        "actor": (make_wogate_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_wogate_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- Vanilla SAC (w/o Transformer + Gate) ----
    ("4-Vanilla-SAC", {
        "type": "vanilla",
        "submodules": [
            ("StateProcessor", make_mtrans_state_processor, forward_state_processor, prep_state_input),
            ("RadarMLP", make_vanilla_radar_mlp, forward_radar_transformer, prep_radar_input),
        ],
        "actor": (make_vanilla_sac_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_vanilla_sac_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- DPRL (1D-CNN) ----
    ("5-DPRL", {
        "type": "sac",
        "submodules": [
            ("StateProcessor", make_mtrans_state_processor, forward_state_processor, prep_state_input),
            ("1D-CNN Processor", make_dprl_cnn_processor, forward_radar_transformer, prep_radar_input),
            ("GatedFusion", make_mtrans_gated_fusion, forward_gated_fusion, prep_gate_inputs),
        ],
        "actor": (make_dprl_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_dprl_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- PointTransSAC ----
    ("6-PointTransSAC", {
        "type": "sac",
        "submodules": [
            ("StateProcessor", make_mtrans_state_processor, forward_state_processor, prep_state_input),
            ("MHSA Radar Trf.", make_pointtrans_radar, forward_radar_transformer, prep_radar_input),
            ("GatedFusion", make_mtrans_gated_fusion, forward_gated_fusion, prep_gate_inputs),
        ],
        "actor": (make_pointtrans_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_pointtrans_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- SetTransSAC ----
    ("7-SetTransSAC", {
        "type": "sac",
        "submodules": [
            ("StateProcessor", make_mtrans_state_processor, forward_state_processor, prep_state_input),
            ("SetTrans Radar Trf.", make_settrans_radar, forward_radar_transformer, prep_radar_input),
            ("GatedFusion", make_mtrans_gated_fusion, forward_gated_fusion, prep_gate_inputs),
        ],
        "actor": (make_settrans_actor, forward_sac_actor, prep_state_radar_inputs),
        "critic": (make_settrans_critic, forward_sac_critic, prep_state_radar_action_inputs),
    }),

    # ---- DDPG ----
    ("8-DDPG", {
        "type": "ddpg",
        "submodules": [],
        "actor": (make_ddpg_actor, forward_ddpg_actor, prep_ddpg_actor_input),
        "critic": (make_ddpg_critic, forward_ddpg_critic, prep_ddpg_critic_input),
    }),
])


# =============================================================================
# 主流程
# =============================================================================

def print_header(title: str, width: int = 100):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_all_benchmarks(device, runs, warmup=10):
    latency_bench = LatencyBenchmark(device, warmup=warmup, runs=runs)

    all_results = OrderedDict()
    first_variant_submodules = None  # 缓存第一个变体的子模块结果供复用

    for variant_name, config in MODEL_CONFIGS.items():
        print_header(f"Benchmarking: {variant_name}")
        variant_results = OrderedDict()

        # 测量子模块（如果存在）
        if config["submodules"]:
            for sub_name, make_fn, fwd_builder, prep_fn in config["submodules"]:
                print(f"  --> {sub_name}...", end=" ", flush=True)
                r = benchmark_module(device, make_fn, fwd_builder, prep_fn,
                                     sub_name, latency_bench)
                variant_results[sub_name] = r
                print(f"Params={r['params_k']}K, GPU={r.get('gpu_ms','N/A')}ms, CPU={r['cpu_ms']}ms")
            # 缓存第一个变体的子模块（StateProcessor/GatedFusion 是共享的）
            if first_variant_submodules is None:
                first_variant_submodules = {k: v for k, v in variant_results.items()}

        # 测量完整 Actor
        print(f"  --> Actor (full)...", end=" ", flush=True)
        actor_make, actor_fwd, actor_prep = config["actor"]
        r_actor = benchmark_module(device, actor_make, actor_fwd, actor_prep,
                                   f"{variant_name}-Actor", latency_bench)
        variant_results["Actor"] = r_actor
        print(f"Params={r_actor['params_k']}K, GPU={r_actor.get('gpu_ms','N/A')}ms, CPU={r_actor['cpu_ms']}ms")

        # 测量完整 Critic
        print(f"  --> Critic (Q1+Q2)...", end=" ", flush=True)
        critic_make, critic_fwd, critic_prep = config["critic"]
        r_critic = benchmark_module(device, critic_make, critic_fwd, critic_prep,
                                    f"{variant_name}-Critic", latency_bench)
        variant_results["Critic (Q1+Q2)"] = r_critic
        print(f"Params={r_critic['params_k']}K, GPU={r_critic.get('gpu_ms','N/A')}ms, CPU={r_critic['cpu_ms']}ms")

        all_results[variant_name] = variant_results
        torch.cuda.empty_cache()

    return all_results


def print_format(results):
    """打印格式化结果表格"""
    print_header("BENCHMARK RESULTS SUMMARY")

    # 设备信息
    device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU only"
    print(f"\n  Hardware: {device_str} | Batch Size: {BATCH_SIZE} | Warmup: {WARMUP_RUNS} | Runs: {BENCH_RUNS}")
    print(f"  Precision: FP32 | Input: State({STATE_DIM}D) + Radar({RADAR_DIM}D) | Action: {ACTION_DIM}D")

    has_gpu = torch.cuda.is_available()

    # ---- Table IV: 计算复杂度总览 ----
    print_header("Table IV: Computational Footprint and Inference Latency")
    header = f"  {'Module':<30} {'Params (K)':>12} {'FLOPs (M)':>12}"
    if has_gpu:
        header += f" {'GPU (ms)':>10}"
    header += f" {'CPU (ms)':>10}"
    print(header)
    print("  " + "-" * (30 + 12 + 12 + (10 if has_gpu else 0) + 10))

    for variant_name, modules in results.items():
        print(f"\n  [{variant_name}]")
        for mod_name, r in modules.items():
            p = r.get("params_k", 0)
            f = r.get("flops_m", 0)
            row = f"  |-- {mod_name:<28} {p:>12.2f} {f:>12.2f}"
            if has_gpu:
                g = r.get("gpu_ms", 0)
                row += f" {g:>10.3f}"
            c = r.get("cpu_ms", 0)
            row += f" {c:>10.3f}"
            print(row)

    # ---- Comparison Summary ----
    print_header("Inference-Only Comparison (Actor Only)")
    header2 = f"  {'Model':<35} {'Params (K)':>12} {'FLOPs (M)':>12}"
    if has_gpu:
        header2 += f" {'GPU (ms)':>10} {'Speedup':>10}"
    header2 += f" {'CPU (ms)':>10}"
    print(header2)
    print("  " + "-" * (35 + 12 + 12 + (20 if has_gpu else 0) + 10))

    # 收集所有 Actor 数据
    actor_data = []
    for variant_name, modules in results.items():
        if "Actor" in modules:
            actor_data.append((variant_name, modules["Actor"]))

    # 找最轻量模型作 baseline
    if actor_data:
        baseline_lat = None
        baseline_params = None
        for name, r in actor_data:
            if name == "4-Vanilla-SAC":
                baseline_lat = r.get("gpu_ms", r.get("cpu_ms", 0))
                baseline_params = r.get("params_k", 0)
                break
        if baseline_lat is None:
            baseline_lat = actor_data[0][1].get("gpu_ms", actor_data[0][1].get("cpu_ms", 0))
            baseline_params = actor_data[0][1].get("params_k", 0)

        for name, r in actor_data:
            p = r.get("params_k", 0)
            f = r.get("flops_m", 0)
            row = f"  {name:<35} {p:>12.2f} {f:>12.2f}"
            if has_gpu:
                g = r.get("gpu_ms", 0)
                speedup = baseline_lat / max(g, 1e-6) if baseline_lat > 0 else 1.0
                row += f" {g:>10.3f} {speedup:>9.2f}x"
            c = r.get("cpu_ms", 0)
            row += f" {c:>10.3f}"
            print(row)
            # Print param ratio
            if baseline_params > 0:
                ratio = p / baseline_params
                print(f"  {'':35} {'Param ratio:':>12} {ratio:>11.2f}x")


def save_csv(results, filepath="benchmark_results.csv"):
    """保存结果到 CSV 文件"""
    has_gpu = torch.cuda.is_available()
    columns = ["Variant", "Module", "Params_K", "FLOPs_M"]
    if has_gpu:
        columns.append("GPU_ms")
    columns.append("CPU_ms")

    rows = []
    for variant_name, modules in results.items():
        for mod_name, r in modules.items():
            row = [variant_name, mod_name,
                   r.get("params_k", 0),
                   r.get("flops_m", 0)]
            if has_gpu:
                row.append(r.get("gpu_ms", 0))
            row.append(r.get("cpu_ms", 0))
            rows.append(row)

    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"\n[OK] Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Model Complexity & Latency Benchmark")
    parser.add_argument("--cpu-only", action="store_true", help="仅测量CPU延迟（不使用GPU）")
    parser.add_argument("--runs", type=int, default=100, help="延迟测量迭代次数 (默认: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数 (默认: 10)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="CSV输出文件路径")
    args = parser.parse_args()

    bench_runs = args.runs
    warmup_runs = args.warmup

    # 设备选择
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = _get_device(prefer_gpu=True)

    print_header("MODEL COMPLEXITY & INFERENCE LATENCY BENCHMARK")
    print(f"  Device: {device}")
    print(f"  torchinfo: {'[OK]' if HAS_TORCHINFO else '[MISSING] (fallback to manual count)'}")
    print(f"  thop: {'[OK]' if HAS_THOP else '[MISSING] (FLOPs estimation unavailable)'}")
    print(f"  Warmup runs: {warmup_runs} | Measurement runs: {bench_runs}")

    # 运行所有基准测试
    results = run_all_benchmarks(device, bench_runs, warmup=warmup_runs)

    # 输出格式化结果
    print_format(results)

    # 保存 CSV
    save_csv(results, args.output)

    print_header("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
