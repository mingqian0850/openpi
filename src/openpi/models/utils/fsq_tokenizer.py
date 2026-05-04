# [解读]: 该模块位于模型定义层，负责把视觉、语言、状态或动作 token 组织成 VLA 模型可训练和可采样的结构。
import math
from typing import Any, Literal

import chex
from einops import einops
from flax import linen as nn
from flax.linen.module import Module
from flax.linen.module import compact
from flax.struct import dataclass
from flax.typing import Array
import jax
import jax.numpy as jnp


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class FsqCodebook(nn.Module):
    input_dim: int
    target_codebook_size: int
    codebook_type: Literal["fsq", "lfq"]

    _bins_per_dim: tuple[int] | None = None

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @property
    def bins_per_dim(self) -> tuple[int]:
        if self._bins_per_dim is not None:
            return self._bins_per_dim

        if self.codebook_type == "fsq":
            return self._get_bins_fsq(self.target_codebook_size)
        elif self.codebook_type == "lfq":  # noqa: RET505
            return self._get_bins_lfq(self.target_codebook_size)
        elif self.codebook_type == "custom":
            return self._get_bins_custom(self.target_codebook_size)
        else:
            raise ValueError(f"Codebook type {self.codebook_type} not supported.")

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @property
    def place_values(self) -> jnp.ndarray:
        place_values = [1]
        for b in self.bins_per_dim[:-1]:
            place_values.append(place_values[-1] * b)
        return jnp.array(place_values)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @staticmethod
    def _get_bins_fsq(target_codebook_size: int) -> tuple[int]:
        """
        Get bins per dimension based on codebook size, from the original FSQ paper.
        """
        if target_codebook_size == 2**8:
            return (8, 6, 5)
        elif target_codebook_size == 2**10:  # noqa: RET505
            return (8, 5, 5, 5)
        elif target_codebook_size == 2**12:
            return (7, 5, 5, 5, 5)
        elif target_codebook_size == 2**14:
            return (8, 8, 8, 6, 5)
        elif target_codebook_size == 2**16:
            return (8, 8, 8, 5, 5, 5)
        else:
            raise ValueError(f"Codebook size {target_codebook_size} not supported.")

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @staticmethod
    def _get_bins_custom(target_codebook_size: int) -> tuple[int]:
        if target_codebook_size == 2**8:
            return (16, 16)
        elif target_codebook_size == 2**10:  # noqa: RET505
            return (32, 32)
        elif target_codebook_size == 2**12:
            return (64, 64)
        elif target_codebook_size == 2**14:
            return (128, 128)
        elif target_codebook_size == 2**16:
            return (256, 256)
        return None

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @staticmethod
    def _get_bins_lfq(target_codebook_size: int) -> tuple[int]:
        """
        Get bins per dimension according to the Lookup-Free Quantization paper (2 bins per dimension)
        """
        assert target_codebook_size & (target_codebook_size - 1) == 0, "Codebook size should be a power of two for LFQ"

        return (2,) * int(math.log2(target_codebook_size))

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def setup(self):
        self.proj_down = nn.Dense(len(self.bins_per_dim))
        self.proj_up = nn.Dense(self.input_dim)

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __call__(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        tokens, z = self.encode(inputs)
        output = self.decode(tokens, z_grad=z)
        return tokens, output

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def encode(self, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        bases = jnp.array(self.bins_per_dim)

        x = self.proj_down(inputs)
        z = jnp.tanh(x)

        # Quantize
        digits = jnp.round((z + 1) * (bases - 1) / 2).astype(jnp.int32)
        tokens = self.undigitize(digits)

        return tokens, z

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def decode(self, tokens: jnp.ndarray, z_grad: jax.Array | None = None) -> jnp.ndarray:
        bases = jnp.array(self.bins_per_dim)
        digits = self.digitize(tokens)

        z_q = digits / (bases - 1) * 2 - 1

        if z_grad is not None:
            chex.assert_equal_shape([z_q, z_grad])
            z_q = jax.lax.stop_gradient(z_q - z_grad) + z_grad

        return self.proj_up(z_q)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def undigitize(self, digits: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(digits * jnp.array(self.place_values), axis=-1)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def digitize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return (tokens[..., None] // jnp.array(self.place_values)) % jnp.array(self.bins_per_dim)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @property
    def vocab_size(self) -> int:
        return math.prod(self.bins_per_dim)


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class ResNetDownBlock(nn.Module):
    stride: int = 1
    n_filters: int = 64
    dropout_rate: float = 0.0
    group_size: int = 32

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        skip = x

        if self.stride > 1 or x.shape[-1] != self.n_filters:
            skip = nn.Conv(self.n_filters, (self.stride,), (self.stride,), "SAME")(skip)

        x = nn.Conv(self.n_filters, (3,), (self.stride,), "SAME")(x)
        x = nn.GroupNorm(num_groups=self.n_filters // self.group_size)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.Conv(self.n_filters, (3,), (1,), "SAME")(x)

        return skip + x


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class ResNetUpBlock(nn.Module):
    stride: int = 1
    n_filters: int = 64
    dropout_rate: float = 0.0
    group_size: int = 32

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        skip = x

        if self.stride > 1:
            skip = nn.ConvTranspose(self.n_filters, (self.stride,), (self.stride,), "SAME")(skip)

        x = nn.ConvTranspose(self.n_filters, (3,), (self.stride,), "SAME")(x)
        x = nn.GroupNorm(num_groups=self.n_filters // self.group_size)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.n_filters, (3,), (1,), "SAME")(x)

        return skip + x


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
@dataclass
class LfqCodebookOutput:
    tokens: jnp.ndarray
    z: jnp.ndarray
    z_q: jnp.ndarray
    token_log_probs: jnp.ndarray
    commit_loss: jnp.ndarray


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class LookupFreeQuantization(nn.Module):
    num_dims: int
    latent_dim: int

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def setup(self):
        self.codebook = jnp.array([-1, 1])
        self.activation = nn.tanh

        self.project_down = nn.Dense(self.num_dims)
        self.project_up = nn.Dense(self.latent_dim)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def encode(self, z: jnp.ndarray) -> jnp.ndarray:
        z = self.project_down(z)
        token_squared_distances = jnp.square(z[..., None] - self.codebook)
        token_bits = jnp.argmin(token_squared_distances, axis=-1)
        return jnp.sum(token_bits * (2 ** jnp.arange(self.num_dims)), axis=-1)

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def decode(self, tokens: jnp.ndarray) -> jnp.ndarray:
        token_bits = (tokens[..., None] & (2 ** jnp.arange(self.num_dims))).astype(jnp.int32)
        return self.project_up(self.codebook[token_bits])

    # [解读]: 该函数承载核心学习或推理步骤，把已经标准化的 observation 转换为损失、梯度或动作输出。
    def loss(self, x: jnp.ndarray) -> LfqCodebookOutput:
        z = self.project_down(x)
        z = self.activation(z)

        token_squared_distances = jnp.square(z[..., None] - self.codebook)
        tokens = jnp.argmin(token_squared_distances, axis=-1)

        token_bit_log_probs = -token_squared_distances
        # Compute token log probs for tokens 0..2^num_dims-1 by summing corresponding log-probs
        token_bit_expansions = jnp.bitwise_and(
            jnp.arange(2**self.num_dims)[None, :], 2 ** jnp.arange(self.num_dims)[:, None]
        ).astype(jnp.int32)
        token_log_probs = (
            token_bit_log_probs[..., 0] @ (1 - token_bit_expansions)
            + token_bit_log_probs[..., 1] @ token_bit_expansions
        )  # (batch_size, num_tokens, 2 ** num_dims)
        token_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(token_log_probs, axis=-1))
        chex.assert_shape(token_log_probs, (*x.shape[:-1], 2**self.num_dims))

        z_q = self.codebook[tokens]
        commit_loss = jnp.square(z - z_q).mean()
        z_q = jax.lax.stop_gradient(z_q - z) + z

        z_q = self.project_up(z_q)
        z = self.project_up(z)

        tokens = jnp.sum(tokens * (len(self.codebook) ** jnp.arange(self.num_dims)), axis=-1)
        return LfqCodebookOutput(
            tokens=tokens,
            z=z,
            z_q=z_q,
            token_log_probs=jnp.zeros(()),
            commit_loss=commit_loss,
        )


# [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
def make_block_causal_attention_matrix(q: jnp.ndarray, k: jnp.ndarray, bs_q: int, bs_k: int) -> jnp.ndarray:
    return nn.make_attention_mask(q, k, pairwise_fn=lambda x, y: jnp.greater_equal(x // bs_k, y // bs_q))


# [解读]: 该类把相关状态和行为集中在一个边界内，降低训练、推理或示例代码之间的耦合。
class GeGLU(Module):
    """Gated Linear Unit with GELU (GeGLU) activation function.
    GeGLU is a Flax layer that combines a linear transformation with a GELU
    activation function in a gating mechanism. It is often used in Transformer models
    to provide non-linear capabilities while preserving a strong linear component.

    Attributes:
        features: the number of output features (default: None).
    """

    output_dim: int = -1

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies the GeGLU activation to the inputs.
        Args:
            inputs: the nd-array to apply the GeGLU activation function to.
        Returns:
            The transformed input.
        """
        output_dim = inputs.shape[-1] if self.output_dim == -1 else self.output_dim

        x = nn.Dense(output_dim * 2)(inputs)
        x, gate = x[..., :output_dim], x[..., output_dim:]
        return x * nn.gelu(gate)


# [解读]: 该模型类封装一段可复用的网络或编码逻辑，让多模态 token、状态和动作在统一接口下组合。
class CrossAttentionLayer(nn.Module):
    dropout_rate: float = 0.0
    num_heads: int = None
    causal: bool = False
    mlp_ratio: float = 4.0

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        *,
        mask_self: jnp.ndarray | None = None,
        mask_cross: jnp.ndarray | None = None,
        train: bool = True,
    ) -> jnp.ndarray:
        d_embed = x.shape[-1]
        seq_len_q = x.shape[-2]
        seq_len_k = y.shape[-2]

        if self.causal:
            # One block size will be 1
            bs_q = max(seq_len_q // seq_len_k, 1)
            bs_k = max(seq_len_k // seq_len_q, 1)

            mask_self = nn.make_causal_mask(x[..., 0])
            mask_cross = make_block_causal_attention_matrix(x[..., 0], y[..., 0], bs_q, bs_k)

        # Self-attention block
        skip = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads or d_embed // 64,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x, x, x, mask=mask_self)
        x = skip + x

        # Cross-attention block
        skip = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads or d_embed // 64,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x, y, y, mask=mask_cross)
        x = skip + x

        # MLP block
        skip = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(int(d_embed * self.mlp_ratio))(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = GeGLU()(x)
        x = nn.Dense(d_embed)(x)
        return skip + x


# [解读]: 该函数集中创建复杂对象，避免调用方散落地拼接配置、依赖和运行时参数。
def sinusoidal_pe_init(_, shape: tuple[int, int]) -> jnp.ndarray:
    seq_len, d_embed = shape

    position = jnp.arange(0, seq_len, 1)
    div_term = jnp.exp(jnp.arange(0, d_embed, 2) * -(jnp.log(10000.0) / d_embed))
    return jnp.concatenate(
        [
            jnp.sin(position[:, jnp.newaxis] * div_term),
            jnp.cos(position[:, jnp.newaxis] * div_term),
        ],
        axis=-1,
    )


# [解读]: 该模型类封装一段可复用的网络或编码逻辑，让多模态 token、状态和动作在统一接口下组合。
class TokenizerEncoderDecoder(nn.Module):
    num_tokens: int
    num_cross_tokens: int
    num_layers: int
    causal: bool

    mlp_ratio: float = 4.0
    use_state_conditioning: bool = False

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    @nn.compact
    def __call__(
        self,
        y: jnp.ndarray,
        *,
        train: bool = True,
        state_conditioning: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x = self.param("q_embed", sinusoidal_pe_init, (self.num_tokens, y.shape[-1]))
        x = jax.numpy.broadcast_to(x, y.shape[:-2] + x.shape[-2:])

        if mask is not None:
            # mask is (batch_dims..., num_cross_tokens)
            chex.assert_equal_shape([y[..., 0], mask])
            attn_mask = einops.repeat(mask, "... kv -> ... 1 q kv", q=self.num_tokens)
        else:
            attn_mask = jnp.ones((*y.shape[:-2], 1, self.num_tokens, self.num_cross_tokens))

        if self.use_state_conditioning:
            assert state_conditioning is not None, "State conditioning is required for this model."
            state_embed = nn.Dense(y.shape[-1], name="state_proj")(state_conditioning)[..., None, :]
            y = jnp.concatenate([y, state_embed], axis=-2)
            attn_mask = jnp.concatenate([attn_mask, jnp.ones_like(attn_mask[..., 0:1])], axis=-1)

        y = y + self.param("y_pos_enc", sinusoidal_pe_init, y.shape[-2:])

        for _ in range(self.num_layers):
            x = CrossAttentionLayer(causal=self.causal, mlp_ratio=self.mlp_ratio)(
                x, y, train=train, mask_self=None, mask_cross=attn_mask
            )

        return x


# [解读]: 该模型类封装一段可复用的网络或编码逻辑，让多模态 token、状态和动作在统一接口下组合。
class FsqAttentionTokenizer(nn.Module):
    embed_dim: int
    data_dim: int
    data_horizon: int
    num_tokens: int
    num_layers: int
    target_codebook_size: int
    causal: bool = False
    mlp_ratio: float = 2.0

    bound: float | None = None

    use_state_conditioning: bool = False

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    @property
    def vocab_size(self) -> int:
        return math.prod(FsqCodebook._get_bins_fsq(self.target_codebook_size))  # noqa: SLF001

    # [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
    def setup(self):
        self.proj = nn.Dense(self.embed_dim)
        self.encoder = TokenizerEncoderDecoder(
            num_tokens=self.num_tokens,
            num_cross_tokens=self.data_horizon,
            num_layers=self.num_layers,
            causal=self.causal,
            use_state_conditioning=self.use_state_conditioning,
            mlp_ratio=self.mlp_ratio,
        )
        self.codebook = FsqCodebook(
            input_dim=self.embed_dim,
            target_codebook_size=self.target_codebook_size,
            codebook_type="custom",
        )
        self.decoder = TokenizerEncoderDecoder(
            num_tokens=self.data_horizon,
            num_cross_tokens=self.num_tokens,
            num_layers=self.num_layers,
            causal=self.causal,
            use_state_conditioning=self.use_state_conditioning,
            mlp_ratio=self.mlp_ratio,
        )

        self.proj_mean = nn.Dense(self.data_dim)
        self.out_scale = self.param("out_scale", lambda _: jnp.full((), 1.0))

    # [解读]: 该函数位于数据规整路径上，用统一规则消除不同数据来源之间的字段、尺度或形状差异。
    def tokenize(
        self, action: jnp.ndarray, *, obs: jnp.ndarray | None = None, train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.bound is not None:
            action = jnp.clip(action, -self.bound, self.bound)

        x = self.proj(action)
        x = self.encoder(x, train=train, state_conditioning=obs)

        return self.codebook.encode(x)

    # [解读]: 该函数位于数据规整路径上，用统一规则消除不同数据来源之间的字段、尺度或形状差异。
    def detokenize(self, tokens: jnp.ndarray, *, obs: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.decoder(self.codebook.decode(tokens), state_conditioning=obs)
        mean = self.proj_mean(x)
        return mean * self.out_scale

    # [解读]: 该函数承载核心学习或推理步骤，把已经标准化的 observation 转换为损失、梯度或动作输出。
    def loss(
        self, action: jnp.ndarray, *, obs: jnp.ndarray | None = None, train: bool = True
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # Encode
        x = self.proj(action)
        z = self.encoder(x, train=train, state_conditioning=obs)

        # Quantize
        tokens, z = self.codebook(z)

        # Decode
        x = self.decoder(z, train=train, state_conditioning=obs)
        mean = self.proj_mean(x) * self.out_scale

        mse = jnp.mean(jnp.square(action - mean))
        mae = jnp.mean(jnp.abs(action - mean))

        return mse, {
            "mse": mse,
            "mae": mae,
        }

    # [解读]: 该特殊方法维护对象生命周期或协议行为，保证实例能被框架、数据加载器或运行时正确调用。
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Dummy for .init
        """
        return self.loss(*args, **kwargs)
