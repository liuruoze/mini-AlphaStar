# 环境变量开关索引（由 experiments/env_index.py 自动生成）

> 修改代码后重跑 `python3 experiments/env_index.py --md <此文件>` 即可更新。

## 常用开关（人工维护）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAS_SL_CLASS_WEIGHTS` | 0.15,build_pylon=8,... | 按动作名前缀给损失加权（必须精确写 pylon/gateway，写 build= 会连带 Build_Nexus） |
| `MAS_SL_BALANCE` | 0 | 按动作类型做 batch 内均衡重采样（只改采样不改损失）；压 other 会崩，别乱用 |
| `MAS_SL_LR / MAS_SL_EPOCHS` | 1e-4 / - | 学习率与轮数；cwB 配方是 1e-3 / 4（ep4 是峰值，ep8 崩） |
| `MAS_SL_SEED` | 20261108 | 训练 seed；同配方 7 个 run 双峰（3/7 可用），必须多 run |
| `MAS_DAGGER` | 0 | 声明这是 DAgger 数据；与 manifest 的 dagger 双向校验，不一致直接拒跑 |
| `MAS_SL_LOAD_STATE / MAS_SL_RESTORE_PATH` | 0 / - | 热启动：从指定 ckpt 继续训（[SL-restore] 会打印加载了多少 key） |
| `MAS_ALLOW_MORE` | 0 | 跳过『训练数上限 + 内存』前置检查（可能 OOM，慎用） |
| `MAS_PF_SKIP` | 0 | 跳过全部前置检查（只在你明确知道后果时用） |
| `OFF_DIFFICULTY` | 3 | 内置 AI 难度；采集/评测基准都是 3 |
| `OFF_GAME_STEPS` | 30000 | 单局步数上限；**评测必须 40000**，30k 会把优势局打成平 |
| `OFF_SEED / OFF_MAX_EP` | - / 1 | 评测种子与局数；一进程一局，种子固定不变 |
| `OFF_GPUS / OFF_GPU` | - | 评测用哪几张卡 / 单进程用哪张；超过 8 进程/卡会 CUDNN_STATUS_NOT_INITIALIZED |
| `OFF_ACT_STATS` | 0 | 导出每局的 function 调用计数（诊断动作分布） |
| `OFF_ACT_BIAS` |  | 推理端动作 logits 偏置，如 Attack_pt=+2.0；实测平均无增益，默认关 |
| `OFF_MAX_JOBS / OFF_MAX_PER_GPU` | NG*6 / 8 | 评测并发上限（本周新增），防止打爆单卡显存 |
| `MAS_EXPERT_IMPL` | v5 | 采集用哪个规则专家（v5=随机化 fast_rush，v4 仅回退） |
| `MAS_DUAL_MODE` | append | learner+专家的协作方式：append=专家叠加命令，replace=专家接管 |

## 全量索引（自动生成）

| 变量 | 分组 | 文件数 | 出现次数 | 默认值 |
|---|---|---:|---:|---|
| `CUDA_DEVICE_ORDER` | CUDA_ | 12 | 12 | - |
| `CUDA_LAUNCH_BLOCKING` | CUDA_ | 12 | 12 | - |
| `CUDA_VISIBLE_DEVICES` | CUDA_ | 24 | 24 | 0/1/3 |
| `DISPLAY` | DISPLAY | 2 | 2 | :1024 |
| `MAS_ACTOR_NUMS` | MAS_ | 1 | 1 | ACTOR_NUMS |
| `MAS_ACTOR_TRACE` | MAS_ | 1 | 1 | - |
| `MAS_ACT_STATS` | MAS_ | 1 | 3 | - |
| `MAS_ALLOW_EXPAND` | MAS_ | 2 | 2 | 0 |
| `MAS_ALLOW_MORE` | MAS_ | 1 | 1 | 0 |
| `MAS_ARMY_THR` | MAS_ | 2 | 2 | 0/8 |
| `MAS_ATK_DUMP` | MAS_ | 1 | 2 | - |
| `MAS_ATTACK_AT` | MAS_ | 4 | 4 | 10 |
| `MAS_ATTACK_EXIT` | MAS_ | 4 | 4 | 10 |
| `MAS_BATCH_ACTING` | MAS_ | 3 | 8 | 0 |
| `MAS_BUILD_COOLDOWN` | MAS_ | 2 | 2 | 16 |
| `MAS_DAGGER` | MAS_ | 5 | 7 | 0/1 |
| `MAS_DATA_DIR` | MAS_ | 2 | 2 | ./data/replay_data_v3_90_nonop/ |
| `MAS_DATA_FROM` | MAS_ | 2 | 4 | - |
| `MAS_DATA_NUM` | MAS_ | 2 | 4 | - |
| `MAS_DATA_SMOKE` | MAS_ | 1 | 1 | 0 |
| `MAS_DEBUG_FUNCALL` | MAS_ | 3 | 3 | - |
| `MAS_DEBUG_FUNCALL_OBS` | MAS_ | 3 | 3 | - |
| `MAS_DEFEND_HOLD` | MAS_ | 4 | 4 | 100 |
| `MAS_DEFEND_R` | MAS_ | 1 | 1 | 0 |
| `MAS_DRY_RUN` | MAS_ | 1 | 1 | 0 |
| `MAS_DUMP_ACT` | MAS_ | 1 | 1 | - |
| `MAS_DUMP_FREEZE` | MAS_ | 1 | 1 | 0 |
| `MAS_DUMP_STATE` | MAS_ | 1 | 1 | - |
| `MAS_DUMP_TRAJ` | MAS_ | 1 | 1 | - |
| `MAS_ENTITY_ORDER` | MAS_ | 3 | 3 | - |
| `MAS_EVAL_DETERMINISTIC` | MAS_ | 1 | 1 | 1 |
| `MAS_EVAL_LOGIT_NOISE` | MAS_ | 1 | 1 | 0.0 |
| `MAS_EVAL_NOISE_SEED` | MAS_ | 1 | 1 | 20261107 |
| `MAS_EVAL_TEMP` | MAS_ | 1 | 1 | 1.0 |
| `MAS_FAULT_FILE` | MAS_ | 1 | 1 | - |
| `MAS_FILTER_REJECTED` | MAS_ | 1 | 1 | 0 |
| `MAS_FINISH` | MAS_ | 1 | 2 | - |
| `MAS_FINISH_ARMY` | MAS_ | 1 | 2 | 8 |
| `MAS_FINISH_FORCE` | MAS_ | 1 | 1 | - |
| `MAS_GATE_EP2` | MAS_ | 2 | 2 | 0.40 |
| `MAS_GATE_EP4` | MAS_ | 2 | 2 | 0.55 |
| `MAS_GPU_LIST` | MAS_ | 2 | 3 | - |
| `MAS_GRAD_AUDIT` | MAS_ | 1 | 1 | 0 |
| `MAS_GREEDY` | MAS_ | 1 | 1 | 0 |
| `MAS_HB_FILE` | MAS_ | 1 | 1 | - |
| `MAS_HEARTBEAT_SEC` | MAS_ | 2 | 2 | 0/60 |
| `MAS_JOIN_DELAY` | MAS_ | 2 | 2 | 0.5 |
| `MAS_KL_MODE` | MAS_ | 4 | 4 | legacy |
| `MAS_LABEL_RULE` | MAS_ | 1 | 1 | fallback |
| `MAS_LEARNER_NOOP` | MAS_ | 1 | 1 | 0 |
| `MAS_LOC_STATS` | MAS_ | 1 | 2 | - |
| `MAS_MATCH_BY_ABILITY` | MAS_ | 1 | 1 | 0 |
| `MAS_MAX_ENTITIES` | MAS_ | 3 | 3 | 128/unset |
| `MAS_MAX_EPISODES` | MAS_ | 1 | 1 | MAX_EPISODES |
| `MAS_MAX_FRAMES` | MAS_ | 1 | 1 | MAX_FRAMES |
| `MAS_MAX_FRAMES_PER_EP` | MAS_ | 1 | 1 | 0 |
| `MAS_MAX_STEPS` | MAS_ | 2 | 4 | - |
| `MAS_MAX_TIME` | MAS_ | 1 | 1 | MAX_TIME_FOR_TRAINING |
| `MAS_MIN_SEL` | MAS_ | 1 | 1 | 1 |
| `MAS_MULTI_ACTION` | MAS_ | 2 | 2 | index0 |
| `MAS_NOTIFY` | MAS_ | 5 | 5 | event |
| `MAS_ON_SERVER` | MAS_ | 5 | 5 | 0 |
| `MAS_ORACLE_SEL` | MAS_ | 1 | 1 | - |
| `MAS_PARALLEL` | MAS_ | 1 | 1 | PARALLEL |
| `MAS_PF_SKIP` | MAS_ | 1 | 1 | 0 |
| `MAS_PICK_TIMEOUT` | MAS_ | 1 | 1 | 1800 |
| `MAS_POLL_S` | MAS_ | 1 | 1 | 0.0005 |
| `MAS_PRUNE_CAP` | MAS_ | 1 | 1 | 128 |
| `MAS_PV_LOG` | MAS_ | 1 | 1 | - |
| `MAS_QUEUE_TIMEOUT` | MAS_ | 1 | 1 | 60 * 60 |
| `MAS_REPLAY_OBSERVER` | MAS_ | 4 | 4 | 0 |
| `MAS_REPLAY_PATH` | MAS_ | 2 | 4 | - |
| `MAS_RETURN_LOGITS` | MAS_ | 1 | 1 | 0 |
| `MAS_RL_DEBUG` | MAS_ | 1 | 1 | 0 |
| `MAS_RL_INIT_MODEL` | MAS_ | 3 | 3 | (unset/model/BEST_SL_v82_soup_prior3.pth |
| `MAS_SAVE_EVERY_BLOCK` | MAS_ | 1 | 1 | 0 |
| `MAS_SCALAR_NORM_PATH` | MAS_ | 4 | 4 | - |
| `MAS_SEL_DUMP` | MAS_ | 1 | 1 | - |
| `MAS_SEL_GUARD` | MAS_ | 1 | 1 | - |
| `MAS_SEL_GUARD_CLEAN` | MAS_ | 1 | 1 | - |
| `MAS_SEL_GUARD_MIN` | MAS_ | 1 | 1 | 0 |
| `MAS_SEL_MAX_SEL` | MAS_ | 1 | 1 | 0 |
| `MAS_SERVER_GPU` | MAS_ | 1 | 1 | - |
| `MAS_SKIP_KL` | MAS_ | 1 | 1 | 0 |
| `MAS_SNAPSHOT_INTERVAL` | MAS_ | 1 | 1 | 8.0 |
| `MAS_SS_P` | MAS_ | 1 | 1 | 0 |
| `MAS_START_METHOD` | MAS_ | 1 | 1 | spawn |
| `MAS_STATIC_NUM` | MAS_ | 1 | 1 | 100 |
| `MAS_TAG_CHECK` | MAS_ | 1 | 1 | - |
| `MAS_TAG_DETAIL` | MAS_ | 1 | 1 | 0 |
| `MAS_TAG_MAP` | MAS_ | 1 | 1 | - |
| `MAS_TAG_STAT_OUT` | MAS_ | 1 | 1 | - |
| `MAS_TARGET_BUILDING` | MAS_ | 1 | 1 | - |
| `MAS_TEACHER` | MAS_ | 1 | 1 | sl |
| `MAS_TEST_PUT_GAP` | MAS_ | 1 | 1 | 0 |
| `MAS_THRESHOLD_ALL` | MAS_ | 1 | 1 | 0 |
| `MAS_TIME_STATS` | MAS_ | 1 | 2 | - |
| `MAS_TRACE_REQ` | MAS_ | 2 | 2 | - |
| `MAS_TRANSFORM_SEED` | MAS_ | 1 | 2 | - |
| `MAS_V4_DEBUG` | MAS_ | 5 | 21 | 0 |
| `MAS_V4_DEBUG2` | MAS_ | 4 | 4 | - |
| `MAS_V4_POWER_DELAY` | MAS_ | 4 | 4 | 3 |
| `MAS_VERIFY_ORDER` | MAS_ | 2 | 4 | - |
| `MAS_DUAL_ACTION` | MAS_DUAL_ | 1 | 1 | 1 |
| `MAS_DUAL_MODE` | MAS_DUAL_ | 1 | 1 | append |
| `MAS_EXPERT_AFTER` | MAS_EXPERT_ | 2 | 2 | 16/8 |
| `MAS_EXPERT_COOLDOWN` | MAS_EXPERT_ | 3 | 3 | 0/32 |
| `MAS_EXPERT_DBG` | MAS_EXPERT_ | 1 | 1 | - |
| `MAS_EXPERT_EVERY` | MAS_EXPERT_ | 3 | 3 | 1/8 |
| `MAS_EXPERT_IMPL` | MAS_EXPERT_ | 1 | 1 | v5 |
| `MAS_EXPERT_INTERVENE` | MAS_EXPERT_ | 1 | 1 | - |
| `MAS_EXPERT_LOG` | MAS_EXPERT_ | 1 | 1 | - |
| `MAS_EXPERT_NOOP` | MAS_EXPERT_ | 4 | 4 | 1 |
| `MAS_EXPERT_RANDOM` | MAS_EXPERT_ | 4 | 4 | 1 |
| `MAS_EXPERT_RAND_SEED` | MAS_EXPERT_ | 4 | 4 | - |
| `MAS_EXPERT_RATIO` | MAS_EXPERT_ | 1 | 1 | 0 |
| `MAS_EXPERT_STRAT` | MAS_EXPERT_ | 5 | 5 | ${STRATS[$((i % 7))] |
| `MAS_EXPERT_TAG_DIR` | MAS_EXPERT_ | 1 | 1 | - |
| `MAS_SL_ACTCONF` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_ACTION_PRIOR` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_ALLOW_BAD_FILES` | MAS_SL_ | 2 | 2 | 0 |
| `MAS_SL_ATTACK_UNITS_W` | MAS_SL_ | 1 | 1 | 1.0 |
| `MAS_SL_ATTACK_WEIGHT` | MAS_SL_ | 1 | 1 | 1.0 |
| `MAS_SL_BALANCE` | MAS_SL_ | 2 | 2 | 0 |
| `MAS_SL_BALANCE_MAXW` | MAS_SL_ | 2 | 2 | 30 |
| `MAS_SL_BALANCE_MIX` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_BATCH` | MAS_SL_ | 2 | 2 | str(3 * AHP.batch_size |
| `MAS_SL_BN_FREEZE` | MAS_SL_ | 2 | 2 | - |
| `MAS_SL_CLASS_WEIGHTS` | MAS_SL_ | 4 | 4 | $BASE_W/none |
| `MAS_SL_CLASS_WEIGHTS_VERBOSE` | MAS_SL_ | 4 | 4 | 0 |
| `MAS_SL_DETERMINISTIC` | MAS_SL_ | 1 | 2 | 0 |
| `MAS_SL_DROPOUT` | MAS_SL_ | 1 | 1 | 0.0 |
| `MAS_SL_EMA_DECAY` | MAS_SL_ | 2 | 3 | 0.0 |
| `MAS_SL_EMA_WARMUP` | MAS_SL_ | 2 | 2 | 0 |
| `MAS_SL_EPOCHS` | MAS_SL_ | 2 | 2 | 10 |
| `MAS_SL_ETA_MIN` | MAS_SL_ | 1 | 2 | 1e-6 |
| `MAS_SL_FRAME_SPLIT` | MAS_SL_ | 2 | 2 | 0 |
| `MAS_SL_FREEZE_BACKBONE` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_GAMMA` | MAS_SL_ | 2 | 4 | 0.2 |
| `MAS_SL_L2SP` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_LOAD_STATE` | MAS_SL_ | 2 | 4 | 0 |
| `MAS_SL_LOC_W` | MAS_SL_ | 1 | 1 | 1. |
| `MAS_SL_LR` | MAS_SL_ | 3 | 5 | 1e-4 |
| `MAS_SL_MAX_FILES` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_MODEL_PATH` | MAS_SL_ | 1 | 1 | ./model/ |
| `MAS_SL_NOOP_WEIGHT` | MAS_SL_ | 1 | 1 | 1.0 |
| `MAS_SL_PRIOR_EXP` | MAS_SL_ | 1 | 1 | 0.5 |
| `MAS_SL_PRIOR_HI` | MAS_SL_ | 1 | 1 | 20.0 |
| `MAS_SL_PRIOR_LO` | MAS_SL_ | 1 | 1 | 0.2 |
| `MAS_SL_PRIOR_VERBOSE` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_RESTORE_PATH` | MAS_SL_ | 2 | 2 | MODEL_PATH + RESTORE_NAME + '.pth/none |
| `MAS_SL_SCHED` | MAS_SL_ | 1 | 1 | step |
| `MAS_SL_SEED` | MAS_SL_ | 1 | 1 | str(SLTHP.seed |
| `MAS_SL_SEL_STRICT` | MAS_SL_ | 2 | 2 | 1 |
| `MAS_SL_SEQ_LEN` | MAS_SL_ | 1 | 1 | str(int(AHP.sequence_length * 0.5 |
| `MAS_SL_SMART_REPLACE` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_SMOKE` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_SPLIT_SEED` | MAS_SL_ | 2 | 2 | 20261108 |
| `MAS_SL_STEP_SIZE` | MAS_SL_ | 1 | 1 | 50 |
| `MAS_SL_TARGET_W` | MAS_SL_ | 1 | 1 | 1. |
| `MAS_SL_TRAIN_ALL` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_TRAIN_HEADS` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_UNITS_W` | MAS_SL_ | 1 | 1 | 1. |
| `MAS_SL_VAL_FRAC` | MAS_SL_ | 3 | 4 | 0.2 |
| `MAS_SL_VAL_NUM` | MAS_SL_ | 2 | 2 | 5 |
| `MAS_SL_VAL_ONLY` | MAS_SL_ | 1 | 1 | - |
| `MAS_SL_VAL_VERBOSE` | MAS_SL_ | 1 | 1 | 0 |
| `MAS_SL_WARMUP` | MAS_SL_ | 2 | 2 | 0 |
| `MAS_SL_WEIGHT_DECAY` | MAS_SL_ | 1 | 1 | 1e-5 |
| `MAS_SL_WORKERS` | MAS_SL_ | 2 | 2 | 0/str(args.num_workers |
| `OFF_ACTORS` | OFF_ | 1 | 1 | 6 |
| `OFF_ACT_BIAS` | OFF_ | 2 | 2 | - |
| `OFF_ACT_STATS` | OFF_ | 1 | 2 | 0 |
| `OFF_ACT_STATS_FILE` | OFF_ | 1 | 1 | ./outputs/actstats_%d.txt" % os.getpid( |
| `OFF_BOT_RACE` | OFF_ | 1 | 2 | terran |
| `OFF_CKPT_EACH` | OFF_ | 1 | 1 | 0 |
| `OFF_DAGGER_DUMP` | OFF_ | 2 | 4 | 0 |
| `OFF_DAGGER_DUMP_DIR` | OFF_ | 2 | 4 | /home/liuruoze/bvb/dagger_states |
| `OFF_DATA` | OFF_ | 1 | 1 | ./data/replay_data_v3_nonop/ |
| `OFF_DATA_FROM` | OFF_ | 2 | 2 | 0 |
| `OFF_DATA_NUM` | OFF_ | 2 | 2 | 90 |
| `OFF_DEBUG_BUILD` | OFF_ | 1 | 1 | - |
| `OFF_DIFFICULTY` | OFF_ | 2 | 2 | 1/3 |
| `OFF_DUMP_TARGETS` | OFF_ | 2 | 3 | 0 |
| `OFF_DUMP_TARGETS_FILE` | OFF_ | 1 | 1 | ./outputs/targets_%d.txt" % os.getpid( |
| `OFF_EPOCHS` | OFF_ | 3 | 3 | 10 |
| `OFF_EVAL_MODE` | OFF_ | 1 | 1 | 0 |
| `OFF_EXPERT_TAG` | OFF_ | 2 | 2 | 0 |
| `OFF_EXPERT_TAG_DIR` | OFF_ | 2 | 2 | - |
| `OFF_FIX_DELAY` | OFF_ | 1 | 1 | 0 |
| `OFF_GAME_STEPS` | OFF_ | 3 | 3 | 21000/30000/40000 |
| `OFF_GAMMA` | OFF_ | 1 | 1 | 0.2 |
| `OFF_GLOBAL_MAX` | OFF_ | 1 | 1 | 26 |
| `OFF_GPU` | OFF_ | 5 | 5 | 0/2 |
| `OFF_GPUS` | OFF_ | 2 | 3 | - |
| `OFF_LR` | OFF_ | 4 | 4 | 1e-4 |
| `OFF_MAX_ENTITIES` | OFF_ | 2 | 2 | 512 |
| `OFF_MAX_EP` | OFF_ | 2 | 2 | 1/36 |
| `OFF_MAX_JOBS` | OFF_ | 1 | 1 | $((NG * 6)) |
| `OFF_MAX_PER_GPU` | OFF_ | 1 | 1 | 8 |
| `OFF_MAX_STEPS` | OFF_ | 1 | 1 | 0 |
| `OFF_MODEL_DIR` | OFF_ | 1 | 1 | ./model_off/ |
| `OFF_MODEL_PATH` | OFF_ | 5 | 5 | ./model_a//./model_p3//./model_sltrain/ |
| `OFF_NO_CUSTOM_OVERRIDE` | OFF_ | 1 | 1 | 0 |
| `OFF_OUT` | OFF_ | 3 | 3 | ./outputs/eval_rl_off.txt/./outputs/eval_sl.txt |
| `OFF_PROBE_AT` | OFF_ | 3 | 6 | - |
| `OFF_PROBE_GREEDY` | OFF_ | 3 | 5 | 0 |
| `OFF_PROBE_TAG` | OFF_ | 3 | 3 | probe |
| `OFF_REPLAY_SORT` | OFF_ | 1 | 1 | 0 |
| `OFF_SAVE_REPLAY` | OFF_ | 2 | 2 | 0 |
| `OFF_SAVE_TENSOR` | OFF_ | 1 | 1 | ./data/replay_data_off90/ |
| `OFF_SEED` | OFF_ | 3 | 3 | 1 |
| `OFF_SEED_MODE` | OFF_ | 1 | 1 | fixed |
| `OFF_SKIP_EXISTING` | OFF_ | 1 | 1 | 0 |
| `OFF_STEP_MUL` | OFF_ | 1 | 1 | 8 |
| `OFF_STEP_SIZE` | OFF_ | 1 | 1 | 50 |
| `OFF_SYNC_CUDA` | OFF_ | 3 | 3 | 0 |
| `OFF_TAG` | OFF_ | 3 | 3 | ep/run |
| `OFF_TRACE_FILE` | OFF_ | 1 | 1 | ./outputs/trace_%d.txt" % os.getpid( |
| `OFF_TRACE_POS` | OFF_ | 1 | 1 | 0 |
| `OFF_TRAIN_NUM` | OFF_ | 2 | 2 | 60/90 |
| `OFF_TRAIN_SEED` | OFF_ | 1 | 1 | str(SLTHP.seed |
| `PF_TAG` | PF_ | 1 | 1 | pf |
| `SC2PATH` | SC2 | 17 | 17 | $HOME/StarCraftII/os.path.expanduser('~/StarCraftII |
| `CACHE_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `CACHE_REPLAY` | 其它 | 2 | 2 | Simple64_2021-12-14-09-33-00.SC2Replay |
| `CHK_DEVICE` | 其它 | 1 | 1 | cuda:0 |
| `CHK_PATH` | 其它 | 1 | 1 | ./model/ |
| `CHK_RACE` | 其它 | 1 | 1 | zerg |
| `CUBLAS_WORKSPACE_CONFIG` | 其它 | 4 | 4 | - |
| `DETAIL_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `DETAIL_REPLAYS` | 其它 | 2 | 2 | Simple64_2021-12-13-08-48-02.SC2Replay |
| `EVAL_BOT_RACE` | 其它 | 1 | 1 | terran |
| `EVAL_CUSTOM_KILL` | 其它 | 1 | 1 | 4000 |
| `EVAL_DIFFICULTY` | 其它 | 1 | 1 | 1 |
| `EVAL_EPISODES` | 其它 | 1 | 1 | 1 |
| `EVAL_MAX_LOOPS` | 其它 | 1 | 1 | 21000 |
| `EVAL_MODEL` | 其它 | 1 | 1 | - |
| `EVAL_OUT` | 其它 | 1 | 1 | ./outputs/eval_sl.txt |
| `EVAL_SEED` | 其它 | 1 | 1 | 1 |
| `EVAL_SEED_PER_EP` | 其它 | 1 | 1 | 1 |
| `EVAL_SEED_STRIDE` | 其它 | 1 | 1 | 1 |
| `FL_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `FL_REPLAY` | 其它 | 2 | 2 | Simple64_2021-12-14-09-33-00.SC2Replay |
| `NAME` | 其它 | 1 | 1 | - |
| `OMP_NUM_THREADS` | 其它 | 1 | 1 | - |
| `OUTDIR` | 其它 | 1 | 1 | - |
| `P0_SEED` | 其它 | 3 | 3 | - |
| `PROBE_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `PYTHONHASHSEED` | 其它 | 4 | 4 | - |
| `RULE` | 其它 | 1 | 1 | - |
| `SCAN_LIST` | 其它 | 2 | 2 | - |
| `SCAN_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `SCAN_MODE` | 其它 | 2 | 2 | date |
| `SCAN_N` | 其它 | 2 | 2 | 4 |
| `SCAN_OUT` | 其它 | 2 | 2 | replay_action_scan.json |
| `SEL_MAX_STEPS` | 其它 | 2 | 2 | 12000 |
| `SEL_REPLAY` | 其它 | 2 | 2 | Simple64_2021-12-14-09-33-00.SC2Replay |
