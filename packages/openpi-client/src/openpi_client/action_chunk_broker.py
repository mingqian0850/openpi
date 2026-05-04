# CN: 实现 action_chunk_broker 的核心逻辑与工具（packages/openpi-client/src/openpi_client/action_chunk_broker.py）。
# EN: Implements core logic and utilities for action_chunk_broker (packages/openpi-client/src/openpi_client/action_chunk_broker.py).

# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    # [解读]: 该函数承载核心学习或推理步骤，把已经标准化的 observation 转换为损失、梯度或动作输出。
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
