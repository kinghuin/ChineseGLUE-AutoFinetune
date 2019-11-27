# ChineseGLUE-AutoFinetune

本项目主要介绍RoBERTa-wwm-ext-large (AutoFinetune)的技术实现，RoBERTa-wwm-ext-large (AutoFinetune)在 RoBERTa-wwm-ext-large 的基础上，采用了 Discriminative Fine-tuning 分层学习率策略；同时，模型采用 PaddleHub AutoDL Finetuner 功能进行超参优化，对学习率，分层策略进行自动化搜索，进一步提升了模型性能，模型在ChineseGLUE XNLI 数据集上的表现达到了 SOTA 的成绩<sup>1</sup>。

## ChineseGLUE简介

[ChineseGLUE (CLUE)](https://github.com/CLUEbenchmark/CLUE) 是由徐亮、Danny Lan、徐国强等人提出的中文自然语言处理测评基准，包含LCQMC、XNLI 、TNEWS 、INEWS 、DRCD、CMRC2018 、BQ、MSRANER、THUCNEWS 、iFLYTEK 等十个数据集，涵盖了文本分类、情感分析、阅读理解、命名实体识别等多个自然语言处理任务。

## Discriminative Fine-tuning策略简介

Discriminative Fine-tuning 是一种学习率逐层递减的策略，通过该策略可以减缓底层的更新速度，让不同层的神经网络以不同的学习率进行学习。PaddleHub 1.4提供了[Discriminative Fine-tuning策略](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Strategy)，其中有2个重要的策略超参分别是factor以及frz_blocks，这两个超参的设置对模型性能有着重要影响，具体可参考[PaddleHub 迁移学习与ULMFiT微调策略](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/tutorial/strategy_exp.md)。

## 使用PaddleHub自动调优、预测

本项目使用[PaddleHub](https://github.com/PaddlePaddle/PaddleHub/) 1.4进行自动调优、预测，实验对象为[ChineseGLUE榜单](http://106.13.187.75:8003/leaderBoard)上的四个数据集（LCQMC、XNLI 、TNEWS 、INEWS）。为了探索更合理的超参设置，本项目使用PaddleHub超参搜索工具进行自动调优。

克隆本项目，通过以下命令启动自动调优：

```bash
hub autofinetune train.py --param_file=hparam.yaml --gpu=0,1,2,3,4,5,6,7 --popsize=16 --round=10 \
 --output_dir="./output" --evaluator=fulltrail --tuning_strategy=hazero dataset $TARGET_DATASET
```

其中`$TARGET_DATASET`可以是INEWS、TNEWS、XNLI、LCQMC中的任意一个，其它参数意义请参考PaddleHub AutoFinetune官方教程：[PaddleHub 超参优化 (AutoDL Finetuner)](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/tutorial/autofinetune.md)。有关train.py源码细节，请参考[PaddleHub文本分类迁移教程](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub文本分类迁移教程))以及[PaddleHub 超参优化（AutoDL Finetuner）——NLP情感分类任务](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/tutorial/autofinetune-nlp.md)。

调优结束后，通过以下命令启动预测：

```bash
export CUDA_VISIBLE_DEVICES=0
python predict.py --dataset $TARGET_DATASET --checkpoint_dir "./output"
```

其中`$TARGET_DATASET`可以是LCQMC、XNLI 、TNEWS 、INEWS中的任意一个。

最终实验结果如下表所示：

| dataset | learning rate          | block | factor             | score |
| ------- | ---------------------- | ----- | ------------------ | ----- |
| TNEWS   | 7.87074615736e-06      | 16    | 1.11736168682      | 90.08 |
| LCQMC   | 7.0320322471924945e-06 | 14    | 1.1048263018713724 | 87.26 |
| XNLI    | 2.38766761764e-05      | 17    | 1.09782046325      | 81.24 |
| INEWS   | 5.040216575441548e-06  | 15    | 1.109245663801446  | 85.4  |

## 免责声明

**本项目仅代表个人作品。** 本项目报告的实验结果可能受到实验设备、系统环境的影响。

</br>
</br>

1: 截至2019.11.27
