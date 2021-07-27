# AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning (NeurIPS 2020)

## Introduction
![alt text](figures/model.jpg)


AdaShare是一个**创新的**和**可微的**方法，用于高效的多任务学习，学习特征共享模式以达到最佳识别精度，同时尽可能限制内存占用。
我们的主要想法是通过一个特定任务的策略来学习共享模式，该策略有选择地选择多任务网络中的特定任务来执行哪些层。
换句话说，我们的目标是获得一个用于多任务学习的单一网络，支持不同任务的单独执行路径。

Here is [the link](https://arxiv.org/pdf/1911.12423.pdf) for our arxiv version. 

Welcome to cite our work if you find it is helpful to your research.
```
@article{sun2020adashare,
  title={Adashare: Learning what to share for efficient deep multi-task learning},
  author={Sun, Ximeng and Panda, Rameswar and Feris, Rogerio and Saenko, Kate},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

##  Experiment Environment

我们的实现是用Pytorch。我们在1个 "Tesla V100 "GPU上对 "NYU v2 2-task"、"CityScapes 2-task "进行训练和测试，
在 "NYU v2 3-task "和 "Tiny-Taskonomy 5-task "使用2个 "Tesla V100 "GPU。

We use `python3.6` and  please refer to [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) to create a `python3.6` conda environment.

在虚拟环境中安装列出的软件包：
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install -c menpo opencv
conda install pillow
conda install -c conda-forge tqdm
conda install -c anaconda pyyaml
conda install scikit-learn
conda install -c anaconda scipy
pip install tensorboardX
```

# Datasets
2GB
请下载格式化的数据集 `NYU v2` [here](https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view?usp=sharing) 

2GB
可以找到格式化的`citycapes`[here](https://drive.google.com/file/d/1WrVMA_UZpoj7voajf60yIVaS_Ggl0jrH/view?usp=sharing).

11.16 TB
Download `Tiny-Taskonomy` as instructed by its [GitHub](https://github.com/StanfordVL/taskonomy/tree/master/data).

17GB
The formatted `DomainNet` can be found [here](https://drive.google.com/file/d/1qVtPnKX_iuNXcR3JoP4llxflIUEw880j/view?usp=sharing).

记得在`./yamls/`的所有`yaml`文件中把`dataroot`改为你的本地数据集路径。

# Training
## Policy Learning Phase
请使用命令执行“Train.py”以获取策略学习
```
python train.py --config <yaml_file_name> --gpus <gpu ids>
```
例如, `python train.py --config yamls/adashare/nyu_v2_2task.yml --gpus 0`.

## 任务类型
"seg", "sn", "depth", "keypoint", "edge"
语义分割，表面正常检测，深度检测，关键点检测和边检测

示例 `yaml` 文件在`yamls/adashare`

**注意:** 使用 `domainnet` 分支用于训练 DomainNet, i.e. `python train_domainnet.py --config <yaml_file_name> --gpus <gpu ids>`

## Retrain Phase
在策略学习阶段之后，我们对8个不同的架构进行抽样，并执行`re-train.py`进行再训练。
```
python re-train.py --config <yaml_file_name> --gpus <gpu ids> --exp_ids <random seed id>
```
其中，我们使用不同的`--exp_ids`来指定不同的随机种子并产生不同的架构。论文中报告了所有8次运行中的最佳性能。

例如, `python re-train.py --config yamls/adashare/nyu_v2_2task.yml --gpus 0 --exp_ids 0`. 

**注意:** 使用 `domainnet` 分支用于训练 DomainNet,, i.e. `python re-train_domainnet.py --config <yaml_file_name> --gpus <gpu ids>`


# Test/Inference
再训练阶段结束后，执行`test.py`以获得测试集的定量结果。
```
python test.py --config <yaml_file_name> --gpus <gpu ids> --exp_ids <random seed id>
```
例如, `python test.py --config yamls/adashare/nyu_v2_2task.yml --gpus 0 --exp_ids 0`.

我们提供训练好的checkpoint如下：
1. Please download  [our model in NYU v2 2-Task Learning](https://drive.google.com/file/d/1f49uFxHg9W5A3-s96f--QxQKrG1MABBw/view?usp=sharing)
2. Please donwload [our model in CityScapes 2-Task Learning](https://drive.google.com/file/d/1x0g8aOQ-esFXIGhoIKeegcl14zf45Ew_/view?usp=sharing)
3. Please download  [our model in NYU v2 3-Task Learning](https://drive.google.com/file/d/1ERfBiDf36rv0wJkb4BlE8w13IDuamcQ-/view?usp=sharing)

要使用这些提供的checkpoint，请将它们下载到`../experiments/checkpoints/`并在那里解压。使用下面的命令来测试
```
python test.py --config yamls/adashare/nyu_v2_2task_test.yml --gpus 0 --exp_ids 0
python test.py --config yamls/adashare/cityscapes_2task_test.yml --gpus 0 --exp_ids 0
python test.py --config yamls/adashare/nyu_v2_3task_test.yml --gpus 0 --exp_ids 0
```

## Test with our pre-trained checkpoints
我们还提供了一些样本图像，以方便测试我们的模型，以完成nyu v2 3任务。

Please download  [our model in NYU v2 3-Task Learning](https://drive.google.com/file/d/1ERfBiDf36rv0wJkb4BlE8w13IDuamcQ-/view?usp=sharing)

Execute `test_sample.py` to test on sample images in `./nyu_v2_samples`, using the command 
```
python test_sample.py --config  yamls/adashare/nyu_v2_3task_test.yml --gpus 0
```
它将打印样本图像的平均定量结果。

## Note
If any link is invalid or any question, please email sunxm@bu.edu




