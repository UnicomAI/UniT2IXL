import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    MT5EncoderModel,
    MT5Tokenizer,
    AutoTokenizer,
    ChineseCLIPTextModel,
    ChineseCLIPModel,
    BertTokenizer
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, StableDiffusionMixin
from diffusers import EulerDiscreteScheduler


from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from pipeline_unit2ixl import UniT2IXLPipeline,TextProjection


@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]

from safetensors.torch import load_file as load_safetensors
    
    
def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map

def convert_sdxl_unet_state_dict_to_diffusers(sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_dict = {sd: hf for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(sd, conversion_dict)

def convert_diffusers_unet_state_dict_to_sdxl(du_sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_map = {hf: sd for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(du_sd, conversion_map)

def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for src_key, value in src_sd.items():
        # さすがに全部回すのは時間がかかるので右から要素を削りつつprefixを探す
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix) :]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd

from collections import OrderedDict

    
sd = load_safetensors("/home/jovyan/LMM/zsa/projects/generative-models/models/2024-12-22T11-26-35_example_training-stage3_v3_step_000006000.safetensors")

#### if __name__=="__main__":  cause TextProjection in __main__ bug in model_index.json when from_pretrained, 
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
# print(sd["conditioner.embedders.1.mlp"])

# ***************** txt project *******************
state_0 = OrderedDict()
state_0["mlp.0.weight"] = sd["conditioner.embedders.0.mlp.mlp.0.weight"]
state_0["mlp.0.bias"] = sd["conditioner.embedders.0.mlp.mlp.0.bias"]
state_0["mlp.2.weight"] = sd["conditioner.embedders.0.mlp.mlp.2.weight"]
state_0["mlp.2.bias"] = sd["conditioner.embedders.0.mlp.mlp.2.bias"]
state_2_0 = OrderedDict()
state_2_0["mlp.0.weight"] = sd["conditioner.embedders.1.mlp.mlp.0.weight"]
state_2_0["mlp.0.bias"] = sd["conditioner.embedders.1.mlp.mlp.0.bias"]
state_2_0["mlp.2.weight"] = sd["conditioner.embedders.1.mlp.mlp.2.weight"]
state_2_0["mlp.2.bias"] = sd["conditioner.embedders.1.mlp.mlp.2.bias"]
state_2_1 = OrderedDict()
state_2_1["mlp.0.weight"] = sd["conditioner.embedders.1.mlp2.mlp.0.weight"]
state_2_1["mlp.0.bias"] = sd["conditioner.embedders.1.mlp2.mlp.0.bias"]
state_2_1["mlp.2.weight"] = sd["conditioner.embedders.1.mlp2.mlp.2.weight"]
state_2_1["mlp.2.bias"] = sd["conditioner.embedders.1.mlp2.mlp.2.bias"]

text_projection = TextProjection(input_dim=1024,hidden_dim=2048,output_dim=768)
text_projection.load_state_dict(state_0)
# text_projection.save_pretrained("text_prjoection")
text_projection_2_0 = TextProjection(input_dim=2048,hidden_dim=4096,output_dim=1280)
text_projection_2_0.load_state_dict(state_2_0)
# text_projection_2_0.save_pretrained("text_prjoection_2_0")
text_projection_2_1 = TextProjection(input_dim=2048,hidden_dim=4096,output_dim=1280)
text_projection_2_1.load_state_dict(state_2_1)
# text_projection_2_1.save_pretrained("text_prjoection_2_1")

# text_prjoection = TextProjection.from_pretrained("yuanjing_xl/text_projection")
# text_prjoection_2_0 = TextProjection.from_pretrained("text_prjoection_2_0")
# text_prjoection_2_1 = TextProjection.from_pretrained("text_prjoection_2_1")
# print(tp.state_dict().keys())

# ***************** txt project *******************


# ***************** unet *******************

state_unet= OrderedDict()
for key in sd.keys():
    if "model.diffusion_model" in key:
        new_key = key.replace("model.diffusion_model.","")
        state_unet[new_key] = sd[key]

hf_unet_state = convert_sdxl_unet_state_dict_to_diffusers(state_unet)

# print(unet_conversion_map)
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
config = UNet2DConditionModel.load_config(
    pretrained_model_name_or_path, subfolder="unet"
)
unet = UNet2DConditionModel.from_config(config)

# with open("sdxl_unet2dcondtionmodel.txt","w") as f:
#     for name in unet.state_dict().keys():
#         f.write(name+"\n")

unet.load_state_dict(hf_unet_state)

#     unet.save_pretrained("unet")

# ***************** unet *******************
# ***************** vae *******************
# exit(0)

pretrained_vae_model_name_or_path = None
vae_path = (
    pretrained_model_name_or_path
    if pretrained_vae_model_name_or_path is None
    else pretrained_vae_model_name_or_path
)



vae = AutoencoderKL.from_pretrained(
    vae_path,
    subfolder="vae_1_0" if pretrained_vae_model_name_or_path is None else None,
)

# ae = AutoencoderKL.from_single_file("/home/jovyan/LMM/zsa/projects/generative-models/models/sdxl-vae-fp16-fix.safetensors")

# vae.save_pretrained("vae")

#     print(vae)
# ***************** vae *******************

# ***************** chinese clip *******************

# ch_clip_model=ChineseCLIPModel.from_pretrained("/home/jovyan/LMM/zsa/models/models--OFA-Sys--chinese-clip-vit-huge-patch14/snapshots/503e16b560aff94c1922f13a86a7693d36957a4f/")
# ch_clip_model.text_model.save_pretrained("text_encoder")
# tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/LMM/zsa/models/models--OFA-Sys--chinese-clip-vit-huge-patch14/snapshots/503e16b560aff94c1922f13a86a7693d36957a4f/")
# tokenizer.save_pretrained("tokenizer")
# cp -r /home/jovyan/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/scheduler/ yuanjing_xl/

text_encoder = ChineseCLIPTextModel.from_pretrained("/home/jovyan/LMM/zsa/projects/generative-models/models/text_encoder")
tokenizer = BertTokenizer.from_pretrained("/home/jovyan/LMM/zsa/projects/generative-models/models/tokenizer")

# ***************** mt5 *******************
tokenizer_2 = MT5Tokenizer.from_pretrained("/home/jovyan/LMM/zsa/projects/generative-models/models/tokenizer_2")
text_encoder_2 = MT5EncoderModel.from_pretrained("/home/jovyan/LMM/zsa/projects/generative-models/models/text_encoder_2")

# ***************** mt5 *******************
#     tokenizer_2 = MT5Tokenizer.from_pretrained("/home/jovyan/LMM/zsa/models/models--Tencent-Hunyuan--HunyuanDit/snapshots/b47a590cac7a3e1a973036700e45b3fe457e2239/t2i/mt5/")
#     tokenizer_2.save_pretrained("tokenizer_2")
#     text_encoder_2 = MT5EncoderModel.from_pretrained("/home/jovyan/LMM/zsa/models/models--Tencent-Hunyuan--HunyuanDit/snapshots/b47a590cac7a3e1a973036700e45b3fe457e2239/t2i/mt5/")
    
    # for name, param in text_encoder_2.named_parameters():
    #     if not param.is_contiguous():
    #         print(f"Parameter '{name}' is not contiguous. Making it contiguous...")
    #         # 注意：对于模型参数，通常不需要手动调用.contiguous()，因为它们在大多数情况下都是连续的。
    #         # 这里只是为了演示如何检查并调用.contiguous()，但在实际中这样做可能是不必要的。
    #         param.data = param.data.contiguous()  # 如果确实需要，可以这样修改参数数据（但通常不推荐）
    #         # 由于param是一个Tensor，直接调用contiguous()会返回一个新的Tensor，但不会原地修改param。
    #         # 因此，上面的赋值操作是正确的做法，但通常不需要这样做。
    #         # 为了避免混淆，这里我们不真正修改param，而是仅打印信息。
    #         print(f"NOTE: Above line would have made '{name}' contiguous, but it's usually not needed for model parameters.")
    #     else:
    #         print(f"Parameter '{name}' is already contiguous.")
    # text_encoder_2.save_pretrained("text_encoder_2")
    
scheduler = EulerDiscreteScheduler.from_config("/home/jovyan/LMM/zsa/projects/generative-models/scripts/yuanjing_xl/scheduler")


pipline = UniT2IXLPipeline(vae,
                             text_encoder,
                             text_encoder_2,
                             tokenizer,
                             tokenizer_2,
                             text_projection,
                             text_projection_2_0,
                             text_projection_2_1,
                             unet,
                             scheduler)

pipline.save_pretrained("yuanjing_xl_relastic_2024-12-22T11-26-35")

    # # print(pipline)



    # pipline.save_pretrained("sdxl_unicom")
    
print(0)