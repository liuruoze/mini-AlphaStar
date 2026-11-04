"""旧名字（soup）的兼容入口 —— 实际实现已改名到 `_avg_models.py`（多版本平均）。

为什么要改名：soup / prior 这类论文术语对读日志的人不友好，
本仓库统一用白话命名：
  · model soup        -> 多版本平均（_avg_models.py）
  · action prior      -> 动作配重（MAS_SL_ACTION_REWEIGHT）
"""
import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_avg_models.py"), run_name="__main__")
