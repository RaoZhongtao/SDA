from ...import_utils import is_bnb_4bit_available, is_bnb_available, is_eetq_available
from ...utils import register_peft_method


from .moelora_model import MOELoraModel, MOELoraConfig, MOELoraLayer


__all__ = [
    "MOELoraModel",
    "MOELoraConfig",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "MOELoraLinear",
    "MOELoraLayer"
]

register_peft_method(name="moelora", config_cls=MOELoraConfig, model_cls=MOELoraModel)