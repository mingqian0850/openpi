# CN: 实现 agent 的核心逻辑与工具（packages/openpi-client/src/openpi_client/runtime/agent.py）。
# EN: Implements core logic and utilities for agent (packages/openpi-client/src/openpi_client/runtime/agent.py).

# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
import abc


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class Agent(abc.ABC):
    """An Agent is the thing with agency, i.e. the entity that makes decisions.

    Agents receive observations about the state of the world, and return actions
    to take in response.
    """

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """Query the agent for the next action."""

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the agent to its initial state."""
