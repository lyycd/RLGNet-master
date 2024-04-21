# RLGNet



## Environment variables & dependencies

```bash
conda create -n RLGNet python=3.10

conda activate RLGNet

pip install -r requirement.txt
```

## Train models

我们将论文中的超参数设置保存在了args.yaml,所以直接训练模型即可

```bash
python main.py -d <dataset> --train_local --trian_global --train_repeat
```

代码将依次训练global,local和repeat模块. 如果有足够的计算资源建议采用如下做法,这可以减少训练时间:

1.	在两个终端中同时训练local和global模块

```bash
python main.py -d <dataset> --train_local
```

```bash
python main.py -d <dataset> --trian_global
```

2. 上诉命令执行完成后训练repeat模块

   ```bash
   python main.py -d <dataset> --train_repeat
   ```

   

## Evaluate models

评估模型最终的结果

```bash
python main.py -d <dataset> --test_repeat [--multi_step] [--test_global] [--test_local]
```

如果加上--multi_step 参数模型将评估多跳推理的结果,否则为单跳推理结果. 如果需要单独评估local和global模型效果,添加--test_local和--test_global即可

## Model Weight

我们同时上传了模型权重 , 只需下载对应数据集权重即可进行模型评估 . 
