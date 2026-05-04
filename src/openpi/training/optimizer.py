# [解读]: 该模块位于训练层，负责把配置、数据、优化器、分片或 checkpoint 组合成可恢复的训练流程。
import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


# [解读]: 该配置类把分散的模型、数据或训练参数收束到一个稳定对象中，便于命令行覆盖和复现实验。
@runtime_checkable
class LRScheduleConfig(Protocol):
    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(self) -> optax.Schedule: ...


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


# [解读]: 该配置类把分散的模型、数据或训练参数收束到一个稳定对象中，便于命令行覆盖和复现实验。
@runtime_checkable
class OptimizerConfig(Protocol):
    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    # [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


# [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
