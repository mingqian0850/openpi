# CN: 实现 policy_agent 的核心逻辑与工具（packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py）。
# EN: Implements core logic and utilities for policy_agent (packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py).

# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @override
    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def reset(self) -> None:
        self._policy.reset()
