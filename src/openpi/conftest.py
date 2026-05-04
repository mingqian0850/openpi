import os

import pynvml
import pytest
# CN: 模块说明 - 测试运行的全局 fixture 与配置入口。
# EN: Module summary - Global fixtures and configuration entrypoint for tests.



def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
