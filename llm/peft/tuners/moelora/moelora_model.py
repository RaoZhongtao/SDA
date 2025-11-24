# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
import operator
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import copy
from functools import partial, reduce
from typing import Union, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from ...utils.other import ModulesToSaveWrapper, _get_submodules, get_pattern_key
# from ..lora import bnb
from ...tuners.tuners_utils import _ExcludedModule, BaseTunerLayer, _find_minimal_target_modules, _maybe_include_all_linear_layers, check_adapters_to_merge
from ...utils import (
    PeftType,
    get_quantization_config,
)
from ...utils.constants import DUMMY_TARGET_MODULES, MIN_TARGET_MODULES_FOR_OPTIMIZATION

from ...utils.integrations import (
    gather_params_ctx,
    init_empty_weights,
)

from ..lora import (
    LoraConfig,
    LoraLayer,
    LoraModel
)

from ..lora.layer import Conv1d, Conv2d, Conv3d, Embedding, MultiheadAttention, Linear


from ...import_utils import is_bnb_4bit_available, is_bnb_available

@dataclass
class MOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOELora`]
    """
    # task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    # task_embedding_dim: int = field(default=64)
    task_num: int = field(default=None)
    task_embedding_dim: int = field(default=None)
    expert_num: int = field(default=4)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOELORA

def _item_emb_pre_forward_hook(target, args, kwargs, item_emb):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["item_emb"] = item_emb
    return args, kwargs


def _task_id_pre_forward_hook(target, args, kwargs, moe_task_ids):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["moe_task_ids"] = moe_task_ids
    return args, kwargs

class MOELoraModel(LoraModel):
    prefix: str = "moelora_"
    
    """
    Create MOELoRA (MOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)


    # def add_adapter(self, adapter_name, config=None):
    #     if config is not None:  # get the lora config
    #         model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
    #         config = self._prepare_mmoelora_config(config, model_config)   # load config
    #         self.peft_config[adapter_name] = config # subsititue the original config
    #     self._find_and_replace(adapter_name)
    #     if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
    #         raise ValueError(
    #             "MOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
    #         )

    #     super()._mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
    #     if self.peft_config[adapter_name].inference_mode:
    #         _freeze_adapter(self.model, adapter_name)


    def _create_and_replace(
        self,
        moelora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        target_root_name=None,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(moelora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(moelora_config.alpha_pattern.keys(), current_key)
        r = moelora_config.rank_pattern.get(r_key, moelora_config.r)
        alpha = moelora_config.alpha_pattern.get(alpha_key, moelora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": moelora_config.lora_dropout,
            "fan_in_fan_out": moelora_config.fan_in_fan_out,
            "init_lora_weights": moelora_config.init_lora_weights,
            "use_rslora": moelora_config.use_rslora,
            "use_dora": moelora_config.use_dora,
            "ephemeral_gpu_offload": moelora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": moelora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "expert_num": moelora_config.expert_num,
            "task_num": moelora_config.task_num,
            "task_embedding_dim": moelora_config.task_embedding_dim,
            "target_root_name": target_root_name
        }
        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from ...tuners.adalora import AdaLoraLayer

        if isinstance(target, MOELoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=moelora_config.lora_dropout,
                init_lora_weights=moelora_config.init_lora_weights,
                use_rslora=moelora_config.use_rslora,
                use_dora=moelora_config.use_dora,
                lora_bias=moelora_config.lora_bias,
            )
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(moelora_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
            
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if "moelora_" not in n and "lora_" not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        count += 1
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, MOELoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
                        count += 1
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
            
    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):

        super()._enable_peft_forward_hooks(*args, **kwargs)
        
        
        moe_task_ids = kwargs.pop("moe_task_ids", None)
        hook_handles = []
        for module in self.model.modules():
            if isinstance(module, MOELoraLayer):
                # Add another hook to overwrite the kwargs with the original adapter names -- this is easier than
                # trying to exclude the encoder.
                pre_forward = partial(_task_id_pre_forward_hook, moe_task_ids=moe_task_ids)

                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()
    
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if lora_config._custom_modules:
            # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
            # types by impelementing their own LoRA layers.
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_default
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
            )

        return new_module
    
    def inject_adapter(
        self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    ) -> None:
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the loading process.

        """
        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = self.get_model_config(model)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        key_list = [key for key, _ in model.named_modules()]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            # dummy adapter, we allow not matching any module
            key_list = []

        # update peft_config.target_modules if required
        peft_config = _maybe_include_all_linear_layers(peft_config, model)

        # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
        # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
        # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
        # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
        # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
        # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
        # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
        # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
        # target_modules is sufficiently big.
        if (
            isinstance(peft_config.target_modules, (list, set))
            and len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION
        ):
            names_no_target = [
                name
                for name in key_list
                if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
            ]
            # new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            # if len(new_target_modules) < len(peft_config.target_modules):
            #     peft_config.target_modules = new_target_modules

        for key in key_list:
            if not key:
                continue
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                key.endswith(module_to_save) for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                continue

            result = self._check_target_module_exists(peft_config, key)
            if isinstance(result, _ExcludedModule):
                excluded_modules.append(key)
            elif not result:
                unmatched_modules.append(key)
            else:
                self.targeted_module_names.append(key)
                parent, target, target_name, root_name = self._get_submodules(model, key)
                ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                with ctx():
                    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, target_root_name=root_name, current_key=key)

        if not self.targeted_module_names and not uses_dummy_target_modules:
            if excluded_modules and not unmatched_modules:
                # All targeted modules were excluded
                raise ValueError(
                    "All modules were excluded. This is likely unintended. "
                    "Check your `target_modules` and `exclude_modules` configuration."
                )
            elif not excluded_modules and unmatched_modules:
                # None of the targeted modules matched
                error_msg = (
                    f"Target modules {peft_config.target_modules} not found in the base model. "
                    f"Please check the target modules and try again."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)
            else:
                # Some modules did not match and some matched but were excluded
                error_msg = (
                    "No modules were targeted for adaptation. "
                    "This might be caused by a combination of mismatched target modules and excluded modules. "
                    "Please check your `target_modules` and `exclude_modules` configuration."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)

        elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
            # exclude_modules was passed but was not used
            warnings.warn(
                f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
                "Please check that exclude_modules was set correctly."
            )

        tied_target_modules = self._get_tied_target_modules(model=model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications, for example when merging the adapter "
                "or converting your model to formats other than safetensors. "
                "See for example https://github.com/huggingface/peft/issues/2018."
            )

        # It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is
        # added, and it targets different layer(s) than the first adapter (which is active), then those different
        # layers will be activated, which we don't want.
        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))
    
    def _get_submodules(self, model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        root_name = key.split(".")[0]
        return parent, target, target_name, root_name




class MOELoraLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("moelora_A", "moelora_B", "moelora_embedding_A", "moelora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "moelora_alpha", "scaling", "moelora_dropout")
    
    def __init__(self, base_layer: nn.Module, expert_num, ephemeral_gpu_offload: bool = False, **kwargs):
        
        super().__init__(base_layer, ephemeral_gpu_offload, **kwargs)
        self.expert_num = expert_num
        self.moelora_alpha = {}
        self.moelora_dropout = nn.ModuleDict({})
        self.moelora_A = nn.ModuleDict({})
        self.moelora_B = nn.ModuleDict({})
        # For Embedding layer
        self.moelora_embedding_A = nn.ParameterDict({})
        self.moelora_embedding_B = nn.ParameterDict({})
        self.moelora_bias: dict[str, bool] = {}
        self.moelora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA


    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.moelora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.moelora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.moelora_A[adapter_name] = MOELinearA(self.in_features, r, self.expert_num)
        self.moelora_B[adapter_name] = MOELinearB(r, self.out_features, self.expert_num)
        self.moelora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.moelora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.moelora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.moelora_A[adapter_name].linear_A[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.moelora_B[adapter_name].linear_B[i].mlp.weight)
        if adapter_name in self.moelora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            for i in range(self.expert_num):
                nn.init.zeros_(self.moelora_embedding_A[adapter_name])
                nn.init.normal_(self.moelora_embedding_B[adapter_name])
                


class MOELoraLinear(nn.Module, MOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")
        self.expert_num = kwargs.pop("expert_num")
        self.task_num = kwargs.pop("task_num")
        self.task_embedding_dim = kwargs.pop("task_embedding_dim")
        super().__init__()
        MOELoraLayer.__init__(self, base_layer, self.expert_num, **kwargs)
        
        # init the Gate network
        self.moelora_gate = nn.ModuleDict({})
        self.moelora_task_embedding = nn.ModuleDict({})
        self.moelora_task_embedding.update(nn.ModuleDict({adapter_name: nn.Embedding(self.task_num, self.task_embedding_dim)}))
        self.moelora_gate.update(nn.ModuleDict({adapter_name: Gate(self.task_embedding_dim, self.expert_num)}))
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("MOELoraLinear does not support merging.")

    def unmerge(self) -> None:
        raise NotImplementedError("MOELoraLinear does not support unmerging.")


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        
        self._check_forward_args(x, *args, **kwargs)
        # print(f"debugging moelora kwargs in forward: {kwargs}")
        moe_task_ids = kwargs.pop("moe_task_ids", None)

        if moe_task_ids is None:
            # this would happen when gradient checkpointing is used
            moe_task_ids = self.cached_moe_task_ids
        else:
            self.cached_moe_task_ids = moe_task_ids
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, moe_task_ids=moe_task_ids, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype 
            for active_adapter in self.active_adapters:
                if active_adapter not in self.moelora_A.keys():
                    continue
                expert_weight = self.moelora_gate[active_adapter](self.moelora_task_embedding[active_adapter](moe_task_ids))
                for i in range(self.expert_num):
                    lora_A = self.moelora_A[active_adapter].linear_A[i]
                    lora_B = self.moelora_B[active_adapter].linear_B[i]
                    dropout = self.moelora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    # print(f"lora_A: {lora_A}")
                    # print(f"result: {result.shape}")
                    # print(f"lora_A.weight.shape: {lora_A.weight.shape}")
                    # print(f"lora_B(lora_A(dropout(x))): {lora_B(lora_A(dropout(x))).shape}")
                    # print(f"expert_weight: {expert_weight.shape}")
                    # print(f"expert_weight[..., i].unsqueeze(-1).unsqueeze(0): {expert_weight[..., i].unsqueeze(-1).unsqueeze(0).shape}")
                    # print(f"expert_weight[..., i].unsqueeze(-1).unsqueeze(0): {expert_weight[..., i].unsqueeze(-1).shape}")
                    # print(f"expert_weight[..., i].unsqueeze(-1).unsqueeze(0): {expert_weight[..., i].shape}")
                    # print(f"expert_weight[..., i].unsqueeze(-1).unsqueeze(0): {expert_weight[..., i].unsqueeze(-1).unsqueeze(0).expand(x.shape[0], -1, -1)}")
                    # try:
                    result = result + lora_B(lora_A(dropout(x))) * scaling * expert_weight[..., i].unsqueeze(-1).unsqueeze(-1)
                    # except Exception as e:
                    #     print(f"[MOELoraLinear] x.shape: {x.shape}, moe_task_ids.shape: {moe_task_ids.shape}")
                    #     import traceback; traceback.print_stack(limit=15)
                    #     print(f"result: {result.shape}")
                    #     print(f"lora_A.weight.shape: {lora_A.weight.shape}")
                    #     print(f"lora_B(lora_A(dropout(x))): {lora_B(lora_A(dropout(x))).shape}")
                    #     print(f"expert_weight: {expert_weight.shape}")
                    #     raise ValueError("Error")


            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], moe_task_ids, **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.moelora_A.keys():
                continue
            
            expert_weight = self.moelora_gate[active_adapter](self.moelora_task_embedding[active_adapter](moe_task_ids))
            for j in range(self.expert_num):
                lora_A = self.moelora_A[active_adapter].linear_A[j]
                lora_B = self.moelora_B[active_adapter].linear_B[j]
                dropout = self.moelora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
                lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling * expert_weight[..., j].unsqueeze(-1).unsqueeze(0)
                result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moelora." + rep


class MOELinearA(nn.Module):
    '''MOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.linear_A = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.linear_A.append(Expert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.linear_A[i](x))

        return outputs
    


class MOELinearB(nn.Module):
    '''MOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.linear_B = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.linear_B.append(Expert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.linear_B[i](x[i]))

        return outputs



class Expert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        x = _cast_input_dtype(x, self.mlp.weight.dtype)
        y = self.mlp(x)

        return y



class Gate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):
        x = _cast_input_dtype(x, self.GateL.weight.dtype)
        y = self.GateL(x)
        y = self.act(y)

        return y

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    target_root_name = kwargs["target_root_name"]
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.MultiheadAttention):
        kwargs.update(lora_config.loftq_config)
        new_module = MultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        if target_root_name in ["visual", "vision_model", "vision_encoder", "vision"]:
            new_module = Linear(target, adapter_name, **kwargs)
        else:
            new_module = MOELoraLinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        if target_root_name in ["visual", "vision_model", "vision_encoder", "vision"]:
            new_module = Linear(target, adapter_name, **kwargs)
        else:
            new_module = MOELoraLinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module


def _cast_input_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Whether to cast the dtype of the input to the forward method.

    Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
    layer.cast_input_dtype=False, this can be disabled if necessary.

    Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
    """
    if x.dtype == dtype:
        return x
    return x.to(dtype=dtype)