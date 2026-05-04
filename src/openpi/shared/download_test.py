# [解读]: 该测试源码用于固定关键行为，避免模型、数据转换或客户端协议在重构时悄悄退化。
import pathlib

import pytest

import openpi.shared.download as download


# [解读]: 该函数封装一个流程节点，使调用方可以按业务语义组合训练、推理或数据处理步骤。
@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


# [解读]: 该函数处理持久化边界，确保权重、资产或中间状态可以在训练和推理之间稳定复用。
def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


# [解读]: 该函数处理持久化边界，确保权重、资产或中间状态可以在训练和推理之间稳定复用。
def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


# [解读]: 该函数处理持久化边界，确保权重、资产或中间状态可以在训练和推理之间稳定复用。
def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


# [解读]: 该函数处理持久化边界，确保权重、资产或中间状态可以在训练和推理之间稳定复用。
def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
