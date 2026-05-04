# [解读]: 该源码文件参与 openpi 的端到端机器人 VLA 流程，负责连接配置、数据、模型或运行时边界。
import os

import pynvml
import pytest


# [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


# [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
