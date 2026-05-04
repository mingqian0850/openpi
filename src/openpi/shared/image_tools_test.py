# [解读]: 该测试源码用于固定关键行为，避免模型、数据转换或客户端协议在重构时悄悄退化。
import jax.numpy as jnp

from openpi.shared import image_tools


# [解读]: 该函数位于数据规整路径上，用统一规则消除不同数据来源之间的字段、尺度或形状差异。
def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)
