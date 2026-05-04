# [解读]: 该模块提供跨训练和推理复用的基础能力，避免图像、下载、归一化和类型逻辑在各处重复实现。
from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


# [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """A higher-order function to JIT-compile `nnx.Module` methods, freezing the module's state in the process.

    Why not `nnx.jit`? For some reason, naively applying `nnx.jit` to `nnx.Module` methods, bound or unbound, uses much
    more memory than necessary. I'm guessing it has something to do with the fact that it must keep track of module
    mutations. Also, `nnx.jit` has some inherent overhead compared to a standard `jax.jit`, since every call must
    traverse the NNX module graph. See https://github.com/google/flax/discussions/4224 for details.

    `module_jit` is an alternative that avoids these issues by freezing the module's state. The function returned by
    `module_jit` acts exactly like the original method, except that the state of the module is frozen to whatever it was
    when `module_jit` was called. Mutations to the module within `meth` are still allowed, but they will be discarded
    after the method call completes.
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclasses.dataclass(frozen=True)
class PathRegex:
    """NNX Filter that matches paths using a regex.

    By default, paths are joined with a `/` separator. This can be overridden by setting the `sep` argument.
    """

    pattern: str | re.Pattern
    sep: str = "/"

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


# [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """Apply a function to the leaves of the state that match the filter."""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
