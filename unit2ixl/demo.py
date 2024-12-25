from pipeline_unit2ixl import UniT2IXLPipeline

pipeline = UniT2IXLPipeline.from_pretrained(f"UnicomAI/UniT2IXL")
# pipeline.load_lora_weights("lora.safetensors")
pipeline = pipeline.to("cuda:0")
prompt ="在绿色的森林中，隐藏着一座白色的哥特式教堂，教堂的尖塔直指蓝色的天空，教堂周围是五彩斑斓的野花和浅黄色的草坪。"

image = pipeline(prompt=prompt,guidance_scale=7.5,target_size=(1024,1024)).images[0]

image.save(f"{prompt[:50]}.png")
