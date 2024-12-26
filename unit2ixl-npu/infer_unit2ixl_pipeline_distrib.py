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
import torch.distributed as dist

import random
from diffusers import DiffusionPipeline
from accelerate import PartialState
import numpy as np


pipeline = UniT2IXLPipeline.from_pretrained("UnicomAI/UniT2IXL")

distributed_state = PartialState()
pipeline = pipeline.to(distributed_state.device)

prompts = [  
"在绿色的森林中，隐藏着一座白色的哥特式教堂，教堂的尖塔直指蓝色的天空，教堂周围是五彩斑斓的野花和浅黄色的草坪。",  
"金秋时节，桂花盛开，金黄色的花朵点缀在枝头，香气四溢。月光透过稀疏的云层，洒在铺满落叶的小径上，营造出一种静谧而温馨的氛围。",  
"清澈的月光下，一只雪白的玉兔在桂花树下忙碌地捣着药，周围环绕着仙气缭绕的云雾，寓意着健康与长寿的美好愿望。",  
"古色古香的街道上，各式各样的灯笼高高挂起，散发出柔和而温暖的光芒。人们手提灯笼，漫步其间，享受着节日的喜庆与祥和。",  
]

output_path="results_distrib"
os.makedirs(output_path, exist_ok=True)

# warm up
for _ in range(3):
    image = pipeline(prompt=prompts[0],guidance_scale=7.5,target_size=(1024,1024)).images[0]

with distributed_state.split_between_processes(prompts) as distributed_pairs:
    for prompt in distributed_pairs:
        t0=time.time()
        image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024),unet_cache=True).images[0]  #unet_cache，是否启用unet cache加速
        print(f"generate time:{time.time()-t0}")
        image.save(f"{output_path}/{prompt[:6]}.png")
