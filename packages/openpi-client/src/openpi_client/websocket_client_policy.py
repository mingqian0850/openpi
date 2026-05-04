# CN: 实现 websocket_client_policy 的核心逻辑与工具（packages/openpi-client/src/openpi_client/websocket_client_policy.py）。
# EN: Implements core logic and utilities for websocket_client_policy (packages/openpi-client/src/openpi_client/websocket_client_policy.py).

# [解读]: 该客户端模块定义机器人侧最小接口和远程调用协议，使控制循环无需直接依赖服务端模型实现。
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


# [解读]: 该运行时类隔离模型推理和外部系统交互，让机器人控制、远程调用和本地模型可以独立演进。
class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    # [解读]: 该函数承载核心学习或推理步骤，把已经标准化的 observation 转换为损失、梯度或动作输出。
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @override
    def reset(self) -> None:
        pass
