# [解读]: 该测试源码用于固定关键行为，避免模型、数据转换或客户端协议在重构时悄悄退化。
import numpy as np

from openpi.models import tokenizer as _tokenizer


# [解读]: 该函数位于数据规整路径上，用统一规则消除不同数据来源之间的字段、尺度或形状差异。
def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


# [解读]: 该函数位于数据规整路径上，用统一规则消除不同数据来源之间的字段、尺度或形状差异。
def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
