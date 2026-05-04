# [解读]: 该模块位于训练层，负责把配置、数据、优化器、分片或 checkpoint 组合成可恢复的训练流程。
from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


# [解读]: 该函数生成约束、掩码或诊断信息，让后续流程能明确数组形状、参数范围和执行边界。
@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


# [解读]: 该函数生成约束、掩码或诊断信息，让后续流程能明确数组形状、参数范围和执行边界。
@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
