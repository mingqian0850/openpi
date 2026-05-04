# CN: 实现 video_display 的核心逻辑与工具（examples/aloha_real/video_display.py）。
# EN: Implements core logic and utilities for video_display (examples/aloha_real/video_display.py).

# [解读]: 该示例源码展示具体机器人或 benchmark 如何接入 openpi 的数据格式、远程 policy 和动作执行流程。
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(self) -> None:
        self._ax: plt.Axes | None = None
        self._plt_img: plt.Image | None = None

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @override
    def on_episode_start(self) -> None:
        plt.ion()
        self._ax = plt.subplot()
        self._plt_img = None

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax is not None

        im = observation["image"][0]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]

        if self._plt_img is None:
            self._plt_img = self._ax.imshow(im)
        else:
            self._plt_img.set_data(im)
        plt.pause(0.001)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()
