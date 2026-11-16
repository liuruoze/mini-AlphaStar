# 命名规则（模型 / 配方 / 运行 / 权重）

> **Abstract (English).** Naming rules only; experimental results live in `CONCLUSIONS.md`.
> Four layers, each with its own identifier: **model** = lunar-mansion code (currently `Hsin`),
> **recipe** = a word code such as `ExpertGuided` (English: `setting`, never `recipe`),
> **run** = the trainer timestamp `sl_YY-MM-DD_HH-MM-SS`, **weight** = timestamp plus
> `_best` or `_epNN`.
> The code pool holds **28 mansion names** picked in three descending tiers (Wade-Giles
> romanisation, then the mansion animal, then a semantic translation), each at most six
> letters, with no punctuation, no SC2 vocabulary, no common Chinese surname unless listed as
> an accepted exception, and **no two codes inside one quadrant within one edit**.
> The four quadrants are `Dragon`, `Turtle`, `Tiger` and `Phoenix`.
> A 17-entry bilingual glossary fixes one Chinese term (two characters each) and one English
> term per concept; banned words are listed explicitly so the vocabulary cannot drift back.
> Evaluation protocol: **strict win** (`raw_reward == +1`) versus **lenient win**
> (`outcome == 1`, or `outcome == 0` with `killed_points > 4000`), 40,000 steps, seeds split
> 8 left-top and 12 right-bottom, never pooled across corners.
> Relative words such as "official", "latest" and "baseline" are banned as identifiers.

> 本文只规定**怎么起名**，不记录实验结论——结论见 `CONCLUSIONS.md`。
> 建立于 2026-11-16，随命名决策更新。

## 1. 四层结构与标识

| 层 | 是什么 | 标识 | 例子 |
|---|---|---|---|
| **模型** | 架构 + 规模；结构不变就是同一个模型 | 星宿代号 | `Hsin` |
| **配方** | 数据 + 超参 + 流程，**不含随机种子** | 全词代号（随实验临时定，无固定规则） | `ExpertGuided` |
| **运行** | 配方 + 一个种子的一次训练 | 时间戳 | `sl_26-11-15_09-37-33` |
| **权重** | 该运行的某个快照，能直接加载运行 | 时间戳 + `_best` / `_epNN` | `sl_26-11-15_09-37-33_best` |

关系：**1 个模型 → N 个配方 → 每个配方 N 个运行 → 每个运行一组权重**。

引用写法按需要的精度选：

| 要说清什么 | 写法 |
|---|---|
| 哪个模型、哪个配方、哪次训练 | `Hsin / HumanHybrid / sl_26-11-15_09-37-33` |
| 讨论方案（合并该配方所有运行） | `HumanHybrid` |
| 讨论某一次训练 | `sl_26-11-15_09-37-33` |
| 讨论某个文件 | `sl_26-11-15_09-37-33_best` |

## 2. 模型代号：二十八宿

### 2.1 取词顺序（依次降级）

| 档 | 规则 | 例子 |
|---|---|---|
| 1 | **威妥玛拼写**：能直接从拼音认出是哪个宿 | `Hsin` = 心宿 |
| 2 | **禽星动物名**：每宿配一种动物，用于重名/带符号/生僻字 | `Wolf` = 奎木狼 |
| 3 | **意译**：前两档都不可用时 | `Wall` = 壁、`Well` = 井、`Star` = 星 |

### 2.2 约束

- 长度 **≤6 字母**（28 个代号目前最长就是 6：`Maiden`、`Monkey`）；四象标签另放宽到 ≤7（最长 `Phoenix`）
- 不含撇号、变音符、单字母
- 不与 SC2 词汇撞名（Zealot / Pylon / Nexus 一律不用）
- 不与常见中文姓氏同形（已保留的例外见 §2.3）
- 不构成"人群 ↔ 动物"的类比
- **同一四象内，任意两个代号不得只差一个字母**（必须换）；跨四象只差一两个字母可以保留，
  但要在 §2.5 登记接受理由

### 2.3 已知例外（有意保留，不是遗漏）

| 项 | 现状 | 理由 |
|---|---|---|
| 与常见姓氏同形 | `Kang`（康）、`Niu`（牛）、`Xu`（徐/许）、`Shen`（沈/申）、`Hsin`（辛） | 这些词的宿名识别度最高，换掉损失大于收益；若真有同姓合作者再加入再逐个替换 |
| 跨四象近名 | `Wall` / `Well`（壁 / 井，只差一个字母） | 分属玄武与朱雀，且"墙"与"井"语义相差极远，实测未产生混淆 |

### 2.4 代号表

| 东方·青龙 `Dragon` | 北方·玄武 `Turtle` | 西方·白虎 `Tiger` | 南方·朱雀 `Phoenix` |
|---|---|---|---|
| 角 **Jue** | 斗 **Tou** | 奎 **Wolf** | 井 **Well** |
| 亢 **Kang** | 牛 **Niu** | 娄 **Bond** | 鬼 **Ghost** |
| 氐 **Ti** | 女 **Maiden** | 胃 **Quail** | 柳 **Roe** |
| 房 **Rabbit** | 虚 **Void** | 昴 **Hen** | 星 **Star** |
| 心 **Hsin** | 危 **Swift** | 毕 **Crow** | 张 **Deer** |
| 尾 **Tail** | 室 **Boar** | 觜 **Monkey** | 翼 **Wings** |
| 箕 **Chi** | 壁 **Wall** | 参 **Shen** | 轸 **Cart** |

四象代号取**本义动物**，只作分组标签，不与 28 个个体争用。
现用模型：**`Hsin`**（心宿，禽星「心月狐」）。

### 2.5 变更记录（为什么不是直觉上的那个词）

| 宿 | 直觉选择 | 实际代号 | 原因 |
|---|---|---|---|
| 角 | Chiao | **Jue** | 按项目内部读法 jué（天文通读为 jiǎo，二者并存） |
| 尾 | Tiger | **Tail** | `Tiger` 让给白虎（四象优先保住本义） |
| 柳 | Liu | **Roe** | `Liu` 与常见姓氏同形 |
| 女 | Bat（女土蝠） | **Maiden** | 代号会与"女性"并排出现，易被误读为人群类比 |
| 室 | Shih | **Boar** | `Shih` 与 `Shen` 只差两个字母 |
| 星 | Hsing | **Star** | `Hsing` 与 `Hsin` 只差一个字母 |
| 娄 | Lou / Dog | **Bond** | `Lou` 与 `Liu` 近似；`Bond` 更贴"连缀"本义 |
| 虚 | Hsü | **Void** | 变音符不便键入，意译本身就是好名字 |
| 翼 | I / Snake | **Wings** | `I` 是单字母不可用；`Wings` 更贴本义 |
| 鬼 | Kuei / Sheep | **Ghost** | `Kuei` 难读；`Sheep` 丢失了"鬼"的信息 |
| 井 | Ching | **Well** | `Ching` 在英语语境有侮辱性联想；犴无英文对应词 |
| 张 | Chang | **Deer** | `Chang` 撞常见姓氏；意译 Extended Net 过长且泛 |
| 房 | Fang | **Rabbit** | `Fang` 与同属青龙的 `Kang` 只差一个字母；房日兔是精确禽星 |

### 2.6 四象不是七宿的上级

二十八宿在前（战国早期已有完整名单），四象在后（汉代定型）。二者是**分区归类 + 象征命名**，
不是行政隶属；且 28 宿宽度极不均匀（觜宿 2 度、井宿 33 度），说明是按月亮每日行程划分的"驿站"，
不是平均切四份。四象编号也不代表代际顺序——**不要用"这一代用青龙七宿"这类轮转规则**。

## 3. 配方：不设命名规则

配方代号**随每次实验临时定**：先做实验，再根据它"和已有配方的区别"起名，不预先分配。
已用过的名字记录在 `CONCLUSIONS.md` 的档案表里。

唯一硬规则：**在同一个模型下，一个配方代号用过就不再复用**（即使该配方被淘汰）。

不同模型之间**可以复用**同一代号——`Hsin / Base` 与未来的 `Kang / Base` 是两个不同的配方，
各自有效。因此完整引用时模型名不可省略；单独写 `Base` 只在"当前模型是 `Hsin`"的语境下成立。

## 4. 运行与权重：时间戳

格式 `sl_<YY-MM-DD>_<HH-MM-SS>`，与上游发布权重同构（`sl_21-12-29_08-15-43`）。

- 一个时间戳 = 一个运行（训练器落盘时自动生成，不需要人工起名）
- `_best` 与 `_ep01..epNN` 是同一个运行的不同快照，共享该时间戳
- 目录名（`cwB_ep4`、`cwH5h_s20261207`）只是**历史别名**，不是标识

## 5. 不用"官方 / 最新 / 基线"这类相对词

它们描述的是"站在谁的位置上看"，会随版本更新漂移（我们推代码之后，这个仓库就成了别人眼里的上游）。
上游那个权重永远叫 `sl_21-12-29_08-15-43`，与谁接管仓库无关。
若历史原因必须用（如 `Official`、`Base`），**首次出现必须带上它的时间戳**。

## 6. 术语表（中英对照）

> 中文一律**两字**便于口语；英文是论文与国际表述的标准写法。两边都要能用。

### 6.1 五个层级词

| 中文 | 英文 | 代码 / 文件里怎么写 | 定义 | 例子 |
|---|---|---|---|---|
| **模型** | model | 只出现在文档 | 架构 + 规模；结构不变就是同一个模型 | `Hsin` |
| **配方** | **setting** | 数据集目录名、训练 tag | 数据 + 超参 + 流程，**不含种子与平台** | `HumanHybrid` |
| **运行** | run | `run_sl.py`、`log/train_<tag>.log` | 配方 × 一个种子的一次训练 | `sl_26-11-15_09-37-33` |
| **权重** | weight | `*.pth` | 能被加载运行的参数文件 | `…_best.pth` |
| **版本** | version | `v_1.09`、git tag | 代码发布版 | `v_1.09` |

> **配方 = setting，不译作 `recipe`。** 首次出现写「配方（setting）」，之后单用。

### 6.2 流程与判据

| 中文 | 英文 | 定义 |
|---|---|---|
| **训练** | training | 从数据到权重的前半程 |
| **验证** | validation | 训练内部留出集上的 action accuracy，**只做门禁，不能跨运行排序** |
| **评测** | evaluation | 在 SC2 上跑真实对局，**唯一有效判据** |

### 6.3 评测相关

| 中文 | 英文 | 定义 |
|---|---|---|
| **条件** | condition | 场景 + 步数 + 角位 + 局数 |
| **判胜** | win rule | 判断"这局算不算赢"的规则，有两套取值 |
| **真胜** | strict win | `raw_reward == +1` |
| **宽胜** | lenient win | `outcome == 1`，或 `outcome == 0` 且 `killed_points > 4000` |
| **对局** | episode | 一次 SC2 对局；`outputs/<tag>s<seed>_<seed>.txt` 一局一行 |
| **角位** | spawn corner | 左上 / 右下 |
| **种子** | seed | 决定出生点位、数据切分与初始化；标准集 `3038–3057` 共 20 个 |
| **快照** | checkpoint | 运行内的参数状态，`_ep01`…`_epNN` 与 `_best` |

### 6.4 环境三词（替代笼统的"环境"）

| 中文 | 英文 | 包含什么 | 出现在哪一侧 |
|---|---|---|---|
| **平台** | platform | OS、CUDA、PyTorch 版本、GPU 型号、驱动 | 训练 + 评测 |
| **场景** | scenario | SC2 客户端版本、地图、对手种族、难度 | 只在评测 |
| **开关** | flag | `MAS_*` / `OFF_*` 环境变量 | 训练 + 评测 |

"环境"这个词不再单独使用——写作时一律替换成平台 / 场景 / 开关中最准确的那个。

### 6.5 不要用的词

| 别用 | 原因 | 改用 |
|---|---|---|
| 轮 | 已被 epoch 占用（`Epoch [1/4]`） | 快照 / 第 N 个 epoch |
| 批 | 已被 batch 占用（`BATCH_SIZE: 24`） | — |
| 局 | 与"对局"重复且更含糊 | 对局 |
| 检查点 | 三字 | 快照 |
| 出生角 | 三字 | 角位 |
| 口径 | 与"条件""判胜"重叠 | 条件 / 判胜 |
| recipe | 配方不译作此词 | setting |
| protocol | 难念，且与通信协议串味 | condition |
| native win / official win | 前者说的是来源不是松紧，后者含相对义 | strict win / lenient win |
| 环境（单独用） | 同时指平台 / 场景 / 开关 | 平台 / 场景 / 开关 |

### 6.6 写法约定

- 所有术语**中文一律两字**；英文只用于代码、文件名与国际表述
- 代码、日志、文件名里仍写 `run`，不改；中文说"运行"
- 统计上说「**运行级**标准差」，不说"run 级 SD"
- 报成绩必须带**条件**（场景 / 步数 / 角位 / 局数）与**判胜**（真胜还是宽胜）

## 7. 评测条件与判胜

> 报成绩必须带全「**场景 / 步数 / 角位 / 局数**」，并写明用的是**真胜**还是**宽胜**；缺一项即视为无效引用。

### 7.1 场景（难度）

| 代号 | 英文 | 说明 |
|---|---|---|
| D1 | very_easy | 官方文档的 0.85 指的是这一档，且只在左上 |
| D2 | easy | — |
| D3 | medium | 本项目主用档位 |

**跨场景不可比**：官方 "0.85" 是 D1 左上，本项目 "70%" 是 D3 双角。

### 7.2 判胜：两套并存，不可混报

两者常差 20–25pp（同一模型在 D3 可以真胜 45%、宽胜 70%），因此**分两列报**：

| 判胜 | 定义 | 何时用 |
|---|---|---|
| **真胜**（strict win） | `raw_reward == +1` | 严格胜负，首选 |
| **宽胜**（lenient win） | `outcome == 1`，或 `outcome == 0` 且 `killed_points > 4000` | 需要与官方文档或历史记录可比时 |

**截断局只认宽胜。** 打满步数上限时 `outcome` 可能为 `+1` 而 `raw_reward = 0`：
SC2 按积分判胜，但 pysc2 因截断返回 0。这类局会让真胜系统性偏低——
`Official` 在 D2 的两次"胜"全部来自这种截断局。

### 7.3 步数

| 项 | 规定 | 依据 |
|---|---|---|
| 标准步数 | **`OFF_GAME_STEPS=40000`** | 30000 会把优势局打成平，历史数字全部偏低 |
| 官方原用步数 | 21000 | 仅用于与官方对照，不用于自身结论 |

### 7.4 局数与运行数

| 项 | 规定 | 依据 |
|---|---|---|
| 每个配方 | **≥3 个独立运行** | 少于 3 个不下配方级结论 |
| 单权重引用 | 必须标注局数 | 20 局的上沿偏差可达 10pp（实测 25% → 15%） |
| 池化 | 必须给 Wilson 区间与运行级标准差 | 单点估计不可用 |

### 7.5 角位分层

20 个标准种子的实测分布：

- **左上 8 个**：3038、3040、3045、3046、3051、3053、3054、3055
- **右下 12 个**：其余

**必须分层报**，混报会掩盖差异：同一模型可以左上 50% / 右下 83%；
`Official` 更极端——左上 88%、右下 8%。

### 7.6 统计只许用脚本

| 工具 | 用途 |
|---|---|
| `experiments/pool_tags.py` | 多运行池化 + Wilson 区间 |
| `experiments/stat_corner.py` | 按角位分层 |

`(胜, 局)` 二元组只能由脚本产出，禁止手算或抄写——单次手算错误的代价是整个结论作废。

### 7.7 验证不能代替评测

| 项 | 事实 |
|---|---|
| 验证是什么 | 训练内部留出集上的 action accuracy |
| 能做什么 | **门禁，且只有上限**：val ≥ 0.57 不评（4 运行 / 80 局，真胜 1.2%）；**不设下限** |
| 不能做什么 | **排序**：同批可用运行里 val 0.2487 → 40%、val 0.2719 → 15%，完全无序 |
| 为什么没有下限 | 0.57 以下四档胜率平坦（37.5%–42.5%，27 运行 / 758 局），任何下限都会误杀——`sl_26-11-13_03-55-05` val 0.5347 → 42.5% 就是被 ≤0.50 闸门挡掉的那个 |
| 结论 | "这个配方好不好"只能由评测回答，不能由 val 回答 |

## 8. 文档语言

三类文档，三种规则。**判据是"读者能不能只用一种语言走完这条路径"**，不是"有没有英文"。

### 8.1 成对镜像（对外文档）

| 英文（入口） | 中文（镜像） |
|---|---|
| [`README.MD`](../../README.MD) | [`doc/README_CHS.MD`](../../doc/README_CHS.MD) |
| [`doc/USAGE.MD`](../../doc/USAGE.MD) | [`doc/USAGE_CHS.MD`](../../doc/USAGE_CHS.MD) |
| [`doc/REPLAY.MD`](../../doc/REPLAY.MD) | [`doc/REPLAY_CHS.MD`](../../doc/REPLAY_CHS.MD) |
| [`doc/HISTORY.MD`](../../doc/HISTORY.MD) | [`doc/HISTORY_CHS.MD`](../../doc/HISTORY_CHS.MD) |

规则：

1. **同结构**：章节数、章节顺序一一对应（可用 `grep '^## '` 对比，数量必须相同）。
2. **同数字**：同一事实在两边必须写成同一个数，连样本量、区间、步数一起写。
3. **不许"精简版"**：中文镜像若省略段落，省略掉的数字就是缺陷——本仓库曾因此把
   `eight fixes` 写成 `7 个补丁`、把 `#23–#52` 写成 `#23–#58`、把 `MAS_QUEUE_TIMEOUT=900`
   写成 `3600`，中文读者拿到的是另一个事实。
4. **改动必须同时改两边**；只改一边视为未完成。

### 8.2 中文正文 + 英文摘要（内部研究文档）

| 文档 | 正文 | 摘要 |
|---|---|---|
| [`docs/exp/CONCLUSIONS.md`](CONCLUSIONS.md) | 中文 | 开头 `> **Abstract (English).**` |
| [`docs/exp/NAMING.md`](NAMING.md) | 中文 | 同上 |
| [`doc/IMPROVEMENTS_2026_11.md`](../../doc/IMPROVEMENTS_2026_11.md) | 中文 | 同上 |

摘要必须**自带关键数字**（配方级胜率、样本量、区间、门禁阈值），不能只写"见正文"。
英文读者读摘要即可判断这份文档是否与自己有关。

### 8.3 单语（自动生成）

`experiments/ENV_KNOBS.md` 由 `experiments/env_index.py` 扫描代码生成，变量名本身是英文，
**不手译**；需要中文说明时写进该变量所在脚本的注释，而不是改这份生成物。

### 8.4 不做的事

- 不在同一个文件里中英并列（会让两份内容互相漂移，且难以 diff）。
- 不为历史日志（`doc/batch_acting/*`）补翻译：它们是当时的记录，改语言等于篡改记录。
