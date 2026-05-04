# CN: 实现 main 的核心逻辑与工具（examples/aloha_real/main.py）。
# EN: Implements core logic and utilities for main (examples/aloha_real/main.py).

# [解读]: 该示例源码展示具体机器人或 benchmark 如何接入 openpi 的数据格式、远程 policy 和动作执行流程。
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000


# [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
