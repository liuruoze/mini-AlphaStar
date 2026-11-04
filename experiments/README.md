# experiments/ —— 实验与验收脚本

> 这些脚本**不是**库代码，不参与 `run_rl.py` / `run_sl.py` 的运行。
> 它们是 batch acting 改造过程中写的一次性探针、验收测试、benchmark 和长跑编排。
> 保留它们是为了**可复现**：日志里的每个数字都对应这里的某条命令。

## 运行约定

- **工作目录必须是仓库根目录**（脚本里 `sys.path.insert(0, os.getcwd())` 依赖它），
  所以命令形如 `python experiments/_test_t0_t1.py` 而不是进目录再跑。
- `_run_regression.sh`、`_soak_run.sh` 这两个要在**有 GPU 和 SC2 的机器**上跑，
  且需要 `ulimit -n 65536` / `OMP_NUM_THREADS=1` / `SC2PATH` / `DISPLAY`（脚本内部已设）。

## 一、验收测试（`_test_*.py`）

| 脚本 | 验什么 |
|---|---|
| `_test_t0_t1.py` | cuda 设备选择不越界 + 纯函数批推理与逐样本等价 |
| `_test_t2_e2e.py` | 同步 barrier 批推理（**负面结论**：吞吐反而更低） |
| `_test_t3b_async.py` | 动态批（异步调度）主循环 |
| `_test_t4_multi.py` | 多 actor 混批 + hidden 隔离 |
| `_test_t5_restart.py` | actor 崩溃后重启、hidden 清零 |
| `_test_t8_logits_to_loss.py` | behavior_logits → `log_prob` → rho == 1 |
| `_test_t9_trajectory_to_loss.py` | 合成 Trajectory → `loss_function` → backward |
| `_test_t21_mpserver.py` | 跨进程 batch server（4 actor × 20 步） |
| `_test_r15_client.py` | 接线层 19 项（含真实训练路径与灵敏度） |
| `_test_r19_worker_obs.py` | 真实 env 观测 8 项 |
| `_test_r58_real_rho.py` | 真实轨迹 rho（3 头，后期被 `_test_r65_delta3` 取代） |
| `_test_r64_sixhead.py` | 六头 rho 尝试（**失败**，数据布局不对，见 ERRATA B10） |
| `_test_r65_kl.py` | **KL 语义的手算参照**（6 条边界，legacy 出现 −0.168） |
| `_test_r65_delta3.py` | 三个已存 head 的未截断 log-prob 差 + 变异测试 |
| `_test_r66_diag.py` | 残差判别：logits 级 + float64（判定为批大小导致的浮点差） |
| `_test_r67_loss.py` | 整批真实轨迹 → 完整 `loss_function`，对比 legacy / renorm |
| `_test_r67_hidden.py` | hidden 契约（非零 / 每步不同 / episode 清零 / 跨轨迹连续） |
| `_test_r70_mapping.py` | **批内 actor 映射 + hidden 生命周期**（注入式，8 组 13 项，含反例自检） |
| `_test_r71_pack.py` | `_pack_actions` 与逐字段版**逐元素等价**（截断 / 大整数 / 混合 dtype） |
| `_test_mask_vec.py` / `_test_cpu_forward.py` | 掩码向量化 / CPU 前向基线 |
| `_test_jit_script.py` / `_test_jit_trace.py` / `_test_compile_21.py` | 图静态化三条路线的**否定性结论** |

## 二、审计与探针（`_audit_*` / `_check_*` / `_diag_*` / `_probe_*` / `_profile_*`）

| 脚本 | 用途 |
|---|---|
| `_audit_batched_hidden.py` | 批推理 4 步 hidden 演化 |
| `_audit_behavior_logits.py` | behavior_logits 与 action 是否同源 |
| `_audit_r14_fixes.py` | 第 14 轮 17 项修复的回归 |
| `_check_batch_logits.py` / `_check_batch_mimic.py` | 批前向与逐样本前向的逐头对比 |
| `_probe_logits.py` / `_probe_state_shape.py` / `_probe_sl_teacher.py` | 形状/数值探针（Trajectory 的 6 个 head 形状就是这里测的） |
| `_diag_compile.py` / `_diag_encoder.py` | compile 与编码器诊断 |
| `_profile_t3a.py` | forward 的 CPU/GPU 时间归因 |

## 三、Benchmark（`_bench_*.py`）

| 脚本 | 量什么 |
|---|---|
| `_bench_acting.py` | 单步 acting 耗时 |
| `_bench_preprocess.py` | `preprocess_state_all` 耗时 |
| `_bench_mp.py` / `_bench_mp_pipe.py` / `_bench_env_mp.py` | 多进程/管道/环境并行度 |
| `_bench_scale.py` | 批大小-吞吐曲线（**注意**：纯采样口径，不含 Learner，勿外推到完整训练） |

## 四、长跑编排（`_soak_*`、`_run_regression.sh`）

| 脚本 | 用途 |
|---|---|
| `_soak_monitor.py` | 10 s 采样：4 卡显存/util、python RSS/线程、SC2 实例数、load |
| `_soak_run.sh` | 单组长跑：清场 → 起监控 → 训练 → 收尾统计（`<tag> <秒> <BATCH> <PAR> <ACT> [ON_SERVER]`） |
| `_soak_series_bc.sh` / `_soak_series_cd.sh` | 多组串行，用于 A/B 对照 |
| `_soak_analyze.py` | 曲线分析（中值滤波 + 分位取点 + 只算后半段斜率） |
| `_run_regression.sh` | **10 项**回归，**只认 rc=0**，不看日志里的 PASS |

## 五、`_fix_del.py`

把 `core/arch/` 里 `del a, b, c` 拆成单行 `del a` 的机械改写脚本，
为 `torch.jit.script` 试探用。**该路线已判定不可行**，但改动留在工作区
（257 行，占 arch diff 的 89%，无语义变化）。保留此脚本是为了说明那批改动的来源。
