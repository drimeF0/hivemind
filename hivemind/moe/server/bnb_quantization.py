import torch
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig
from accelerate.utils.bnb import replace_with_bnb_layers





BASE_QUANT_CONFIG = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

def quantization(
    model: torch.nn.Module
):
    """
    This function will quantize the input model with the associated config passed in `bnb_quantization_config`. If the
    model is in the meta device, we will load and dispatch the weights according to the `device_map` passed. If the
    model is already loaded, we will quantize the model and put the model on the GPU,

    Args:
        model (`torch.nn.Module`):
            Input model. The model can be already loaded or on the meta device
        bnb_quantization_config (`BnbQuantizationConfig`):
            The bitsandbytes quantization parameters

    Returns:
        `torch.nn.Module`: The quantized model
    """

    # compatibility with peft
    model.is_loaded_in_4bit = BASE_QUANT_CONFIG.load_in_4bit
    model.is_loaded_in_8bit = BASE_QUANT_CONFIG.load_in_8bit

    with init_empty_weights():
        model = replace_with_bnb_layers(
            model, BASE_QUANT_CONFIG, modules_to_not_convert=[]
        )
        return model