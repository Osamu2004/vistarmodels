# RSUniVLM七数据集结果与训练来源标记（2026-11-04）

来源：用户WSL终端完整CSV；19项回归通过，七集20,196张/对全部计分，失败0，complete=True。当前输出不是官方原版解码/多类合并结果。

| 数据集 | 指标 | 精度（%） | 边界WFm（%） | 数量 | 训练来源 |
|---|---|---:|---:|---:|---|
| LoveDA\* | mIoU | 41.34 | 11.21 | 1669 | 明确使用该数据集 |
| FLAIR | mIoU | 18.94 | 15.06 | 15700 | 公开清单未列名 |
| UAVid\* | mIoU | 39.17 | 14.52 | 270 | 明确使用该数据集 |
| xBD-pre | building IoU | 27.25 | 12.15 | 933 | 公开清单未列名 |
| CHN6-CUG | road IoU | 31.05 | 9.85 | 903 | 公开清单未列名 |
| LEVIR-CD\*† | building IoU | 27.65 | -- | 128 | 使用其派生来源LEVIR-MCI |
| SECOND | mIoU_c | 9.49 | -- | 593 | 公开清单未列名 |

\*：训练使用该基准或其源影像的派生数据。†：LEVIR-CD对应训练源LEVIR-MCI，其经LEVIR-CC继承了主要来自LEVIR-CD的影像。标记训练暴露不等于已经证明当前评测test泄漏；未标记也不等于已排除其他指令来源/基础预训练的重叠。

LoveDA/FLAIR/UAVid为本次固定ID升序、后类覆盖的合并结果；原query mIoU已被新标准mIoU取代，仅限本次新目录。其余四项继承原评分，不补造缺失WFm。独立语义/变化协议保持原记录。

表注英文：

> * indicates that training used the benchmark or imagery derived from it. For LEVIR-CD (†), exposure is through LEVIR-MCI, an extension of LEVIR-CC whose images mainly originate from LEVIR-CD. This annotation does not establish overlap with the evaluated test split. An unmarked result only means that the benchmark is not named in the disclosed training sources.

一手来源：

- [RSUniVLM：表10和表12](https://arxiv.org/html/2412.05679v2#S7)。
- [Change-Agent：第III节，LEVIR-MCI扩展LEVIR-CC](https://arxiv.org/html/2403.19646v3#S3)。
- [LEVIR-CC原论文：III.A，影像主要源于LEVIR-CD](https://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf)。

当前只归档终端标量和来源标记，未独立读取WSL原始plan/summary或逐样本训练manifest。没有重新执行模型、解码或评分，也没有自动改正式主表。
