from pipeline_unit2ixl import UniT2IXLPipeline

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
import torch
import torch_npu
import time
import os

pipeline = UniT2IXLPipeline.from_pretrained("UnicomAI/UniT2IXL")
pipeline = pipeline.to("npu")

prompt = "在绿色的森林中，隐藏着一座白色的哥特式教堂，教堂的尖塔直指蓝色的天空，教堂周围是五彩斑斓的野花和浅黄色的草坪。"

output_path="results"
os.makedirs(output_path, exist_ok=True)

# warm up
for _ in range(3):
    image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024)).images[0]


t0=time.time()
image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024), unet_cache=True).images[0] #unet_cache，是否启用unet cache加速
print(f"generate image used time:{time.time()-t0}.")
image.save(f"{output_path}/{prompt[:6]}.png")

