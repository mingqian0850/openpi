# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
import abc


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """Append a step to the episode."""

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends."""
