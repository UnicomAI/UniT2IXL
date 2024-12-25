# UniT2IXL

## 介绍

​	UniT2IXL模型在昇腾智算服务器上的微调与推理说明。

## 环境搭建
【模型开发时推荐使用配套的环境版本】

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
    <th>安装指南</th>
  </tr>
  <tr>
    <td> Python </td>
    <td> 3.10 </td>
  </tr>
  <tr>
    <td> Driver </td>
    <td> AscendHDK 24.1.RC3 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 24.1.RC3 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.0.RC3 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Torch </td>
    <td> 2.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td> Torch_npu </td>
    <td> release v6.0.RC3 </td>
  </tr>
</table>


1、 软件与驱动安装

  torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)


    # 推荐使用python3.10
    conda create -n test python=3.10
    conda activate test
    
    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310m-linux_aarch64.whl
    
    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

2、 依赖安装

 2.1  安装其余依赖

```
pip install -r requirements.txt
```

 2.2  将本代码仓中的attention_processor.py和unet_2d_condition.py复制到diffusers目录下

```
DIFFUSERS_PATH=`python3 -c "import diffusers; print(diffusers.__path__[0])"`

cp ./attention_processor.py ${DIFFUSERS_PATH}/models/
cp ./unet_2d_condition.py ${DIFFUSERS_PATH}/models/unets/
```

## 微调训练
UniT2IXL现支持LoRA微调，在昇腾800T上测试通过。

1、指定训练卡数

​	1.1 修改accelerate_deepspeed_config.yaml的num_processors，值为训练需要的卡数；

​	1.2 修改finetune_unit2ixl_lora_deepspeed_fp16.sh中的num_processors，值为训练需要的卡数，与1.1中的值保持一致；

​	1.3 修改finetune_unit2ixl_lora_deepspeed_fp16.sh中的ASCEND_RT_VISIBLE_DEVICES，指定使用卡的id，注意卡数与前两步设置的卡数保持一致；

​	1.4 修改finetune_unit2ixl_lora_deepspeed_fp16.sh中的model_name、train_data_dir和validation_prompt，model_name可修改为本地路径，train_data_dir为自定义数据集地址，validation_prompt用于训练过程中验证LoRA效果。

​	1.5 数据集格式见[README](../README.md)

2、执行微调训练

    bash finetune_unit2ixl_lora_deepspeed_fp16.sh

## 推理

推理在昇腾800I上测试通过。

	python infer_unit2ixl_pipeline.py
	python infer_unit2ixl_pipeline_lora.py # 带lora的推理

