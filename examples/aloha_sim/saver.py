# CN: 实现 saver 的核心逻辑与工具（examples/aloha_sim/saver.py）。
# EN: Implements core logic and utilities for saver (examples/aloha_sim/saver.py).

# [解读]: 该示例源码展示具体机器人或 benchmark 如何接入 openpi 的数据格式、远程 policy 和动作执行流程。
import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @override
    def on_episode_start(self) -> None:
        self._images = []

    # [解读]: 该函数是运行时控制点，负责把配置、循环、网络连接或环境交互串成可执行流程。
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @override
    def on_episode_end(self) -> None:
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
        )
