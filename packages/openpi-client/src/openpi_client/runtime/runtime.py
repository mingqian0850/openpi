# CN: 实现 runtime 的核心逻辑与工具（packages/openpi-client/src/openpi_client/runtime/runtime.py）。
# EN: Implements core logic and utilities for runtime (packages/openpi-client/src/openpi_client/runtime/runtime.py).

# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
