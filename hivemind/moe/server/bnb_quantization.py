import torch
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig
from torch import nn
import torch






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

    model = replace_with_bnb_layers(
        model, BASE_QUANT_CONFIG
    )
    dtype = BASE_QUANT_CONFIG.torch_dtype
    for name, param in model.state_dict().items():
        if torch.is_floating_point(param):
            param.to(dtype)
    return model



def replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=None, current_key_name=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules or by `bnb.nn.Linear4bit`
    modules from the `bitsandbytes`library. The function will be run recursively and replace `torch.nn.Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[str]`):
            Names of the modules to not quantize convert. In practice we keep the `lm_head` in full precision for
            numerical stability reasons.
        current_key_name (`List[str]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert.
    """

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    model, has_been_replaced = _replace_with_bnb_layers(
        model, bnb_quantization_config, modules_to_not_convert, current_key_name
    )
    return model


def _replace_with_bnb_layers(
    model,
    bnb_quantization_config,
    modules_to_not_convert=None,
    current_key_name=None,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    # bitsandbytes will initialize CUDA on import, so it needs to be imported lazily
    import bitsandbytes as bnb

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            proceed = True
            for key in modules_to_not_convert:
                if (
                    (key in current_key_name_str) and (key + "." in current_key_name_str)
                ) or key == current_key_name_str:
                    proceed = False
                    break
            if proceed:
                # Load bnb module with empty weight and replace ``nn.Linear` module
                if bnb_quantization_config.load_in_8bit:
                    bnb_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=bnb_quantization_config.llm_int8_threshold,
                    )
                elif bnb_quantization_config.load_in_4bit:
                    bnb_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        bnb_quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=bnb_quantization_config.bnb_4bit_use_double_quant,
                        quant_type=bnb_quantization_config.bnb_4bit_quant_type,
                    )
                else:
                    raise ValueError("load_in_8bit and load_in_4bit can't be both False")
                bnb_module.weight.data = module.weight.data
                if module.bias is not None:
                    bnb_module.bias.data = module.bias.data
                bnb_module.requires_grad_(False)
                setattr(model, name, bnb_module)
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = _replace_with_bnb_layers(
                module, bnb_quantization_config, modules_to_not_convert, current_key_name
            )
            has_been_replaced = has_been_replaced | _has_been_replaced
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced