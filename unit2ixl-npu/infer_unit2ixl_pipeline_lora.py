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
# add your own LoRA path
lora_path = ""
pipeline.load_lora_weights(lora_path)
pipeline.fuse_lora(lora_scale=0.7)
pipeline = pipeline.to("npu")

# use your own prompt
prompt = ""

output_path="results_lora"
os.makedirs(output_path, exist_ok=True)

# warm up
for _ in range(3):
    image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024)).images[0]

t0=time.time()
image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024),unet_cache=True).images[0] #unet_cache，是否启用unet cache加速
print(f"generate image used time:{time.time()-t0}.")
image.save(f"{output_path}/{prompt}.png")
