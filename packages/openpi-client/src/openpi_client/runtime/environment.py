# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
import abc


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class Environment(abc.ABC):
    """An Environment represents the robot and the environment it inhabits.

    The primary contract of environments is that they can be queried for observations
    about their state, and have actions applied to them to change that state.
    """

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state.

        This will be called once before starting each episode.
        """

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """Allow the environment to signal that the episode is complete.

        This will be called after each step. It should return `True` if the episode is
        complete (either successfully or unsuccessfully), and `False` otherwise.
        """

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def get_observation(self) -> dict:
        """Query the environment for the current state."""

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """Take an action in the environment."""
