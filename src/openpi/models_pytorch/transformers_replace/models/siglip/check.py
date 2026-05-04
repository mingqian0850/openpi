import transformers
# CN: 模块说明 - SigLIP Transformers 替换实现与检查逻辑。
# EN: Module summary - SigLIP transformer replacement implementation and checks.


def check_whether_transformers_replace_is_installed_correctly():
    return transformers.__version__ == "4.53.2"
