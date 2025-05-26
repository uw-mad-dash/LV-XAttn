import argparse
import os
import sys
import gc

import torch
import numpy as np
from accelerate import Accelerator
from huggingface_hub import HfFolder
from peft import PeftModel
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention, repeat_kv, apply_rotary_pos_emb, _prepare_cross_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
import transformers
import csv
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch import nn
from transformers import PreTrainedModel


from typing import Optional, Tuple, Union, List
import logging
import time

from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mllama

from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

### lv-xattn
from lv_xattn import _attn_forward as _lv_xattn_forward, \
                            _attn_backward as _lv_xattn_backward, \
                            initialize_distributed as _lv_xattn_initialize_distributed, \
                            reset_global_memory_buffer as _lv_xattn_reset_global_memory_buffer

from ring import _attn_forward as _ring_forward, \
                            _attn_backward as _ring_backward, \
                            initialize_distributed as _ring_initialize_distributed, \
                            reset_global_memory_buffer as _ring_reset_global_memory_buffer

from ring_self import _attn_forward as _ring_self_forward, \
                            _attn_backward as _ring_self_backward, \
                            initialize_distributed as _ring_self_initialize_distributed, \
                            reset_global_memory_buffer as _ring_self_reset_global_memory_buffer

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 1
CROSS_ATTENTION_STATES = None
VISION_ENCODER_TIME_LAPSE = []
CA_TIME_LAPSE = []

def set_global_cross_attention_states(vision_features):
    global CROSS_ATTENTION_STATES
    CROSS_ATTENTION_STATES = vision_features

def get_global_cross_attention_states():
    global CROSS_ATTENTION_STATES
    return CROSS_ATTENTION_STATES

def clear_global_cross_attention_states():
    global CROSS_ATTENTION_STATES
    CROSS_ATTENTION_STATES = None

def clear_ca_time_lapses():
    global CA_TIME_LAPSE
    CA_TIME_LAPSE = []

def add_ca_time_lapse(time_lapse):
    global CA_TIME_LAPSE
    CA_TIME_LAPSE.append(time_lapse)

def get_ca_time_lapses():
    global CA_TIME_LAPSE
    return CA_TIME_LAPSE

def clear_vision_encoder_time_lapses():
    global VISION_ENCODER_TIME_LAPSE
    VISION_ENCODER_TIME_LAPSE = []

def add_vision_encoder_time_lapse(time_lapse):
    global VISION_ENCODER_TIME_LAPSE
    VISION_ENCODER_TIME_LAPSE.append(time_lapse)

def get_vision_encoder_time_lapses():
    global VISION_ENCODER_TIME_LAPSE
    return VISION_ENCODER_TIME_LAPSE


def initialize_distributed(attention_mode):
    assert attention_mode in ['lv_xattn', 'ring', 'ring_self']
    if attention_mode == 'lv_xattn':
        _lv_xattn_initialize_distributed()
    elif attention_mode == 'ring':
        _ring_initialize_distributed()
    elif attention_mode == 'ring_self':
        _ring_self_initialize_distributed()

def reset_global_memory_buffer(attention_mode):
    assert attention_mode in ['lv_xattn', 'ring', 'ring_self']
    if attention_mode == 'lv_xattn':
        _lv_xattn_reset_global_memory_buffer()
    elif attention_mode == 'ring':
        _ring_reset_global_memory_buffer()
    elif attention_mode == 'ring_self':
        _ring_self_reset_global_memory_buffer()


def create_attention_class_save_qkv(attention_mode, no_overlap=False):
    assert attention_mode in ['lv_xattn', 'ring', 'ring_self']
    class _attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, causal, sm_scale, bias_func=None, local_bias_args=[], remote_bias_args=[]):
            torch.cuda.synchronize()
            start_time = time.time()
            comm_mode = None
            backward_engine = 'flash'
            
            if attention_mode == 'lv_xattn':
                forward_func = _lv_xattn_forward
            elif attention_mode == 'ring':
                forward_func = _ring_forward
            elif attention_mode == 'ring_self':
                forward_func = _ring_self_forward

            q, k, v, o, L = forward_func(q, k, v, causal, sm_scale, comm_mode, bias_func=bias_func, local_bias_args=local_bias_args, remote_bias_args=remote_bias_args, no_overlap=no_overlap)

            ctx.save_for_backward(q, k, v, o, L, torch.tensor(len(local_bias_args)), *local_bias_args, *remote_bias_args)
            ctx.sm_scale = sm_scale
            ctx.comm_mode = comm_mode
            ctx.backward_engine = backward_engine
            ctx.causal = causal
            ctx.bias_func = bias_func
            ctx.no_overlap = no_overlap
            torch.cuda.synchronize()
            add_ca_time_lapse(time.time() - start_time)
            return o

        @staticmethod
        def backward(ctx, do):
            torch.cuda.synchronize()
            start_time = time.time()
            q, k, v, o, L, local_bias_args_len, *bias_args = ctx.saved_tensors
            local_bias_args = bias_args[:local_bias_args_len]
            remote_bias_args = bias_args[local_bias_args_len:]
            sm_scale = ctx.sm_scale
            bias_func = ctx.bias_func
            no_overlap = ctx.no_overlap

            if attention_mode == 'lv_xattn':
                backward_func = _lv_xattn_backward
            elif attention_mode == 'ring':
                backward_func = _ring_backward
            elif attention_mode == 'ring_self':
                backward_func = _ring_self_backward
            
            dq, dk, dv = backward_func(do, q, k, v, o, L, ctx.causal, sm_scale, ctx.comm_mode, ctx.backward_engine, bias_func=bias_func, local_bias_args=local_bias_args, remote_bias_args=remote_bias_args, no_overlap=no_overlap)
            torch.cuda.synchronize()
            add_ca_time_lapse(time.time() - start_time)
            return dq, dk, dv, None, None, None, None, None

    return _attention

def create_attention_class(attention_mode, no_overlap=False):
    assert attention_mode in ['lv_xattn', 'ring', 'ring_self']
    class _attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, run_function, is_causal):
            torch.cuda.synchronize()
            start_time = time.time()
            comm_mode = None
            backward_engine = 'flash'
            
            if attention_mode == 'lv_xattn':
                forward_func = _lv_xattn_forward
            elif attention_mode == 'ring':
                forward_func = _ring_forward
            elif attention_mode == 'ring_self':
                forward_func = _ring_self_forward

            q, k, v = run_function(x, get_global_cross_attention_states())
            sm_scale = q.size(-1) ** -0.5

            _, _, _, o, L = forward_func(q, k, v, is_causal, sm_scale, comm_mode, bias_func=None, no_overlap=no_overlap)

            ctx.save_for_backward(x, o, L)
            ctx.causal = is_causal
            ctx.run_function = run_function
            ctx.attention_mode = attention_mode
            ctx.comm_mode = comm_mode
            ctx.backward_engine = backward_engine
            ctx.no_overlap = no_overlap
            
            reset_global_memory_buffer(attention_mode)
            torch.cuda.synchronize()
            add_ca_time_lapse(time.time() - start_time)
            return o

        @staticmethod
        def backward(ctx, do):
            torch.cuda.synchronize()
            start_time = time.time()
            causal = ctx.causal
            run_function = ctx.run_function
            attention_mode = ctx.attention_mode
            comm_mode = ctx.comm_mode
            backward_engine = ctx.backward_engine
            no_overlap = ctx.no_overlap

            x, o, L = ctx.saved_tensors

            if attention_mode == 'lv_xattn':
                backward_func = _lv_xattn_backward
            elif attention_mode == 'ring':
                backward_func = _ring_backward
            elif attention_mode == 'ring_self':
                backward_func = _ring_self_backward

            q, k, v = run_function(x, get_global_cross_attention_states())
            sm_scale = q.size(-1) ** -0.5

            dq, dk, dv = backward_func(do, q, k, v, o, L, causal, sm_scale, comm_mode, backward_engine, bias_func=None, no_overlap=no_overlap)

            reset_global_memory_buffer(attention_mode)
            torch.cuda.synchronize()
            add_ca_time_lapse(time.time() - start_time)
            return None, None, None, None

    return _attention

def freeze_LLM_only(model):
    """
    Freeze self-attention layers in the language_model. vision_model, multi_modal_projector, and cross-attention layers will be fine-tuned
    """
    for name, param in model.language_model.named_parameters():
                param.requires_grad = False
    for i, layer in enumerate(model.language_model.model.layers):
        if i in model.language_model.model.cross_attention_layers:
            for param in layer.parameters():
                # param.requires_grad = True
                param.requires_grad = False

def freeze_Vision_only(model):
    for name, param in model.vision_model.named_parameters():
        param.requires_grad = False

def patch_self_attention_forward(attention_mode):
    attention_class = create_attention_class_save_qkv(attention_mode)
    attention = attention_class.apply

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MllamaModel is using MllamaTextSelfSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.dropout if self.training else 0.0,
        #     is_causal=is_causal,
        # )
        sm_scale = query_states.size(-1) ** -0.5
        # TODO: mask
        attn_output = attention(
            query_states, 
            key_states, 
            value_states, 
            is_causal, 
            sm_scale)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
    
    return forward

def patch_cross_attention_forward(attention_mode):
    # attention_class = create_attention_class_save_qkv(attention_mode)
    attention_class = create_attention_class(attention_mode)
    attention = attention_class.apply

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        def project(hidden_states, cross_attention_states):
            if output_attentions:
                # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
                logger.warning_once(
                    "MllamaModel is using MllamaTextCrossSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                    'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                return super().forward(
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            query_states = self.q_norm(query_states)

            if cross_attention_states is not None:
                key_states = self.k_proj(cross_attention_states)
                value_states = self.v_proj(cross_attention_states)
                key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # if past_key_value is not None:
                #     # if we have a new image + new tokens, we only computed key_states on that new image
                #     # we still update the cross key states, past_image, new_image. And use it!
                #     key_states, value_states = past_key_value.update(
                #         key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                #     )
            elif cache_position[0] != 0:
                key_states, value_states = (
                    past_key_value.key_cache[self.layer_idx],
                    past_key_value.value_cache[self.layer_idx],
                )
            else:
                raise ValueError(
                    "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if attention_mask is None and q_len > 1 else False

            # see if requires grad
            query_states.requires_grad_()
            key_states.requires_grad_()
            value_states.requires_grad_()

            return query_states, key_states, value_states
        
        # query_states, key_states, value_states = project(hidden_states)
        bsz, q_len, _ = hidden_states.size()
        is_causal = True if attention_mask is None and q_len > 1 else False
        

        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=attention_mask,
        #     dropout_p=self.dropout if self.training else 0.0,
        #     is_causal=is_causal,
        # )
        # TODO: mask
        # attn_output = attention(
        #     query_states, 
        #     key_states, 
        #     value_states, 
        #     is_causal, 
        #     sm_scale)
        attn_output = attention(hidden_states, project, is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # print memory usage
        return attn_output, None, past_key_value
    
    return forward

def MllamaForConditionalGeneration_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            start_time = time.time()
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )
            torch.cuda.synchronize()
            add_vision_encoder_time_lapse(time.time() - start_time)

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
        
        set_global_cross_attention_states(cross_attention_states)

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=get_global_cross_attention_states(),
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )
        
        return outputs


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)


def load_model_and_processor(model_name: str):
    """Load model and processor with optional LoRA adapter"""
    hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir="/tmp/",
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map=device,
        token=hf_token,
    )
    # print(model.language_model.model.layers)
    processor = MllamaProcessor.from_pretrained(
        model_name, 
        cache_dir="/tmp/",
        token=hf_token, use_safetensors=True
    )

    # model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image_path: str = None, image=None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB").resize((224, 224))
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB").resize((224, 224))
    raise ValueError("No valid image provided")


def fsdp_auto_wrap_policy(model, transformer_layer_names):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy

def fsdp(model):
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(
                model,
                [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                ],
            )
    
    with enable_wrap(wrapper_cls=FSDP):
        model.language_model.model.layers = nn.ModuleList(
            wrap(layer) for layer in model.language_model.model.layers
        )
        model.language_model = wrap(model.language_model)
    model.vision_model = FSDP(
        model.vision_model,
        auto_wrap_policy=my_auto_wrapping_policy,
        use_orig_params=False,
    )
    
    return model

def benchmark(args, attention_mode, per_sample_num_images, per_sample_text_length,
              num_nodes, total_iter, warmup_iter, benchmark_file,
              is_profile, trace_file=None):
    
    local_num_images = per_sample_num_images // num_nodes
    local_text_length = per_sample_text_length // num_nodes

    assert local_text_length >= local_num_images

    transformers.models.mllama.modeling_mllama.MllamaTextCrossSdpaAttention.forward = patch_cross_attention_forward(attention_mode)
    transformers.models.mllama.modeling_mllama.MllamaTextSelfSdpaAttention.forward = patch_self_attention_forward("ring_self")
    transformers.models.mllama.modeling_mllama.MllamaForConditionalGeneration.forward = MllamaForConditionalGeneration_forward

    if is_profile:
        assert trace_file is not None

    torch.manual_seed(0)
    np.random.seed(0)

    model, processor = load_model_and_processor(args.model_name)
    # apply_liger_kernel_to_mllama(model)
    freeze_Vision_only(model)

    model = fsdp(model)

    model.train()

    time_lapses = {
        "forward": [],
        "backward": [],
        "ca_forward": [],
        "ca_backward": [],
        "vision_encoder_forward": [],
    }
    for i in range(total_iter):
        image = [process_image(image_path=args.image_path)] * local_num_images
        prompt = "<|image|>" * local_num_images
        # padd prompt to local_text_length
        prompt = prompt + " a" * (local_text_length - local_num_images)

        inputs = processor(
            image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
        ).to(device)

        labels = "a " * (local_text_length - 1)
        labels = processor.tokenizer(labels, return_tensors="pt").input_ids
        

        clear_ca_time_lapses()
        clear_vision_encoder_time_lapses()
        torch.cuda.synchronize()
        start_time = time.time()
        # print the labels
        outputs = model(**inputs, labels=labels)
        # output = model.generate(
        #     **inputs, temperature=0.7, top_p=0.9, max_new_tokens=MAX_OUTPUT_TOKENS, use_cache=False
        # )
        torch.cuda.synchronize()
        time_lapses["forward"].append(time.time() - start_time)
        time_lapses["ca_forward"].append(sum(get_ca_time_lapses()))
        time_lapses["vision_encoder_forward"].append(sum(get_vision_encoder_time_lapses()))

        loss = outputs.loss

        reset_global_memory_buffer(attention_mode)
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        clear_ca_time_lapses()
        start_time = time.time()
        loss.backward(retain_graph=False)
        torch.cuda.synchronize()
        time_lapses["backward"].append(time.time() - start_time)
        time_lapses["ca_backward"].append(sum(get_ca_time_lapses()))

        # clear_global_media_offset()
        # clear_global_vision_features()
        clear_global_cross_attention_states()
        reset_global_memory_buffer(attention_mode)
        model.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

        if dist.get_rank() == 0:
            append_row(benchmark_file, attention_mode, num_nodes, per_sample_text_length, per_sample_num_images, i,
                        time_lapses["forward"][-1], time_lapses["ca_forward"][-1], time_lapses["vision_encoder_forward"][-1], time_lapses["backward"][-1], time_lapses["ca_backward"][-1])
    
    forward_time = time_lapses["forward"][warmup_iter:]
    backward_time = time_lapses["backward"][warmup_iter:]
    avg_forward_time = sum(forward_time) / len(forward_time)
    avg_backward_time = sum(backward_time) / max(len(backward_time), 1)

    forward_ca_time = time_lapses["ca_forward"][warmup_iter:]
    backward_ca_time = time_lapses["ca_backward"][warmup_iter:]
    avg_forward_ca_time = sum(forward_ca_time) / len(forward_ca_time)
    avg_backward_ca_time = sum(backward_ca_time) / max(len(backward_ca_time), 1)
    
    forward_vision_encoder_time = time_lapses["vision_encoder_forward"][warmup_iter:]
    avg_forward_vision_encoder_time = sum(forward_vision_encoder_time) / max(len(forward_vision_encoder_time), 1)

    if dist.get_rank() == 0:
        append_row(benchmark_file, attention_mode, num_nodes, per_sample_text_length, per_sample_num_images, "avg", 
                     avg_forward_time, avg_forward_ca_time, avg_forward_vision_encoder_time, avg_backward_time, avg_backward_ca_time)
    
    del model
    del processor
    del inputs
    del image
    del labels
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    


def append_row(file, attention_mode, num_nodes, text_length, num_images, round, forward_time, forward_ca_time, forward_vision_encoder_time, backward_time, backward_ca_time):
    with open(file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([attention_mode, num_nodes, text_length, num_images, round, forward_time, forward_ca_time, forward_vision_encoder_time, backward_time, backward_ca_time])


def main(args):
    attention_modes = ['lv_xattn', 'ring']
    num_nodes = int(os.environ['WORLD_SIZE'])
    per_sample_text_length_num_images = [
        (63 * num_nodes, 12 * num_nodes),
    ]
    actual_iter = 5
    warmup_iter = 2
    total_iter = actual_iter + warmup_iter
    is_profile=False
    benchmark_file = f"llama_experiments/llama_benchmark_{num_nodes}.csv"
    os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)
    trace_dir = "llama_trace"
    if is_profile and not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    header_added = False
    for text_length, num_images in per_sample_text_length_num_images:
        for attention_mode in attention_modes:
            initialize_distributed(attention_mode)
            initialize_distributed("ring_self")
            if dist.get_rank() == 0:
                print(f" *** attention_mode: {attention_mode}, text_length: {text_length}, num_images: {num_images} ***")
            
            if dist.get_rank() == 0 and not header_added:
                append_row(benchmark_file, "attention_mode", "num_nodes", "text_length", "num_images", "iter", "forward_time", "forward_ca_time", "forward_vision_time", "backward_time", "backward_ca_time")
                header_added = True

            benchmark(
                args=args,
                attention_mode=attention_mode,
                per_sample_num_images=num_images,
                per_sample_text_length=text_length,
                num_nodes=num_nodes,
                total_iter=total_iter,
                warmup_iter=warmup_iter,
                benchmark_file=benchmark_file,
                is_profile=is_profile,
                trace_file=f"{trace_dir}/{attention_mode}_{num_nodes}_{text_length}_{num_images}.json",
            )
            # try: 
            #     benchmark(
            #         args=args,
            #         attention_mode=attention_mode,
            #         per_sample_num_images=num_images,
            #         per_sample_text_length=text_length,
            #         num_nodes=num_nodes,
            #         total_iter=total_iter,
            #         warmup_iter=warmup_iter,
            #         benchmark_file=benchmark_file,
            #         is_profile=is_profile,
            #         trace_file=f"{trace_dir}/{attention_mode}_{num_nodes}_{text_length}_{num_images}.json",
            #     )
            # except Exception as e:
            #     # raise
            #     print(f"Error: {e}")
            #     if dist.get_rank() == 0:
            #         append_row(benchmark_file, attention_mode, num_nodes, text_length, num_images, "error", 0, 0, 0, 0, 0)
            # clean all the memory
            reset_global_memory_buffer(attention_mode)
            reset_global_memory_buffer("ring_self")
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-modal inference with optional Gradio UI and LoRA support"
    )
    parser.add_argument("--image_path", type=str, help="Path to the input image", default="dog.jpg")
    parser.add_argument("--prompt_text", type=str, help="Prompt text for the image")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL, help="Model name"
    )

    args = parser.parse_args()

    main(args)