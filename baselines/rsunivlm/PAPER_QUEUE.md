# RSUniVLM七数据集队列

状态：2026-11-02已实现，只有本机静态检查；WSL合成回归、真实CUDA前向、吞吐、显存和完整精度尚待用户回执。命令统一在仓库`RUN_QUEUE.md`。不修改当前论文数值，不新增训练。

## 当前稿件对应的范围

依据`/Users/zhouziyu/code/vistar/paper/sec/4_experiments.tex`及`paper/sec/supplement.tex`：

| 数据集 | 群体 | 指标 | 默认根目录 |
|---|---|---|---|
| LoveDA | Val 1,669 | 七类mIoU/mF1，3px WFm | `/root/data/LoveDA` |
| FLAIR#1 | 官方test 15,700 | 前12类mIoU/mF1，3px WFm | `/root/data/FLAIR-1-2/data/flair#1-test` |
| UAVid | 现有OVSISBench/all 270配对资产 | 八类mIoU/mF1，3px WFm | `/root/data/OVSISBenchDataset/uavid` |
| xBD-pre | test/pre 933 | 建筑IoU/F1，3px WFm | `/root/data/xview2/test` |
| CHN6-CUG | Val 903 | 道路IoU/F1，3px WFm | `/root/data/CHN6-CUG/val` |
| LEVIR-CD | 官方test 128对 | 双方向并集建筑变化IoU/F1 | `/root/data/LEVIR-CD` |
| SECOND | DynamicEarth/Bi-SRNet Val 593对 | 六类变化并集mIoU_c/mF1_c | `/root/data/second_dataset` |

VDD不在当前这份稿件的上述评测集里。本队列只做分割与变化检测，不做生图或FDr⁶；SECOND的1,694对生图test不能代替593对检测集合。

路径可通过`DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json`覆盖。任一集合缺失或群体不合规时，预检会一次列出错误并在加载模型前停止，不悄悄跳过数据集。`--allow_nonstandard_population`只供开发，使用后不能发布为完整基准。MAX_SAMPLES是每数据集全局前N个，再分配到各卡；0为完整集合。

## 两张卡和任意batch

`GPU_IDS=0,1`启动两个独立工作进程，每进程一份相同模型，处理全局样本序号模2的互斥子集。每张卡依次处理七个数据集，模型首次需要前向时加载，之后复用；不是每个数据集重载，也没有把模型切成两半。两份输出分目录，不共享写入同一掩码。

`BATCH_SIZE`是**每张卡的序列batch**，支持任意正整数，如3、5、16、17。每个序列是一块512 RGB，变化任务则是一对配准512 RGB。两个GPU各16时，最多同时32个序列。流式裁块队列跨原图填满batch，不受单张LoveDA只有四裁块的限制；尾批、格式重试子集可以较小。CUDA OOM会把实际batch减半并重试原请求，不跳图；batch1仍OOM时该数据集失败并继续下一个数据集。日志`actual_batch=`与每条原文`generation.inference_batch_size`是实际值，目录名和请求值不是显存适用性证据。

当前采用用户选择的直接批量推理，不再做逐像素单图等价核验，也不会因此永久降成1。已观察到批量与单图差异，二者不能宣称等价；新队列用独立协议目录，不导入旧v4/v5单图缓存。自己同协议的有效PNG/原文可恢复，数据、权重、源码、batch、卡数或空间/提示协议改变会改变身份。

## 空间、标签与输出

已逐项核对Alters实际裁块源码，源图512/stride512、右下补黑、原图拼接规则一致；详见[源码核对记录](ALTERS_TILING.md)。该对齐不改变RSUniVLM官方384视觉输入，也不宣称其RLE输出与Alters内部预测分辨率相同。

所有任务保持源图原生范围、512无重叠裁块、右/下黑色补齐，官方384视觉处理。每块输出原生矩形RLE，最近邻恢复到完整512再裁掉补齐部分，拼回原图后评分。无TTA、无GT输入、无GT选类或选答案。模型仍为原官方GMoE加载器，expert2、fp16/eager、greedy、KV cache和8192新token上限保持。

五个分割集使用现有`segearth_ov.protocols`的发现/标签解码/评分数学：LoveDA排除raw0/255，FLAIR仅评价raw1..12，UAVid保留moving/static car，xBD使用既有WKT rasterizer，CHN6是二类道路。WFm复用既有3px实现与合并规则。不同数据集统一类别清单RLE提示属于VISTAR适配，不能称作RSUniVLM官方已报告协议；未知标签、无计数、行宽错误或无EOS截断都进入至多两次固定格式重试。只选首个合法输出，不填背景或跳过失败样本。

LEVIR双方向二值结果逐像素OR，只保存一张并集预测。SECOND保留forward/reverse两张0..6预测；对每个变化类c分别计算`pred_c=(forward==c)|(reverse==c)`和`gt_c=(label1==c)|(label2==c)`，全样本累计TP/TN/FP/FN后按DynamicEarth/OpenDPR的eps=1e-7计算六类均值，不把方向冲突压成单个类别。SECOND GT采用严格官方0..6 ID/已知RGB颜色，未知编码停止，不能猜测或按最近颜色改GT。

SECOND固定名单已从当前Vistar代码复制到`assets/splits/second_bisrnet_dynamic_earth_val.txt`，593个唯一条目，SHA256为`eed4dc3956022e18f426ba9efd30fd0927cd4fa0c065b7058e86e439624ed346`。根目录依次检查Val/Train/Test等物理位置，只接受能完整覆盖名单的配对语义标签目录；不能用其他593个样本凑数。

## 结果、恢复及失败

- 每集`shards/rank000`和`rank001`保留预测ID掩码、RLE原文/全部尝试、哈希回执、进度JSONL和分片metrics。只存预测图片，不存输入、GT、彩色掩码副本、叠加图。
- 预测先形成并落盘，GT像素随后才进入评分。相同身份恢复时核验PNG形状、类别和哈希并重新评分；不信任旧分数、损坏文件或半写入结果。
- 两卡完成后，父进程核对样本身份、归属、重复和遗漏，合并全局混淆矩阵/SECOND逐类计数及WFm逐图统计，**不平均两卡mIoU**。每集输出`metrics.json`，总览为`paper_summary.json`和CSV。
- 某数据集格式/数据/显存失败时保留错误和部分进度，继续队列剩余集合；最终该集合仍是incomplete。只有七集均完成所选群体才使队列complete；只有全部官方群体且严格校验才使is_full_benchmark=true。开发子集不能充当全量。
- 物理GPU日志在`logs/gpu_0.log`、`gpu_1.log`；`runtime.json`记录本次启动的PID，环境文件记录实际可见GPU与包版本，结束回执记录退出码。父进程Ctrl+C只中断其自行创建的进程组；原有其他任务不在控制范围。
- 预检`PLAN_ONLY=1`仅发现/核对数据和权重元数据，不创建GPU工作进程或加载模型。预检可能为UAVid标签编码审计读取GT，但这些数据不进入模型推理。

此前训练数据核查仍适用：模型明确在LoveDA/UAVid上受过分割训练，LEVIR-MCI/LEVIR-CD重叠未审计。新结果不是“全部数据集未见”的官方零样本结果，详见TRAINING_DATA_AUDIT.md。

## 验证边界

本机按用户要求不运行测试或模型；仅AST、shell、JSON/名单及原数值源码保护检查。WSL合成回归在`tests/test_rsunivlm_paper_queue.py`，覆盖任意batch、跨图、配准双图对应、EOF/EOS、尾批、缓存、互斥分片、全局计数合并、六类变化并集与OOM拆批；真实完整模型和七种数据格式仍待WSL验证。代码bundle校验只证明传输完整。

SECOND协议依据当前Vistar的`docs/second_dynamic_earth_eval_protocol.md`、`eval_flux2_second.py`及其名单；公开来源为[Bi-SRNet名单](https://github.com/DingLei14/Bi-SRNet/blob/main/datasets/val_list.txt)、[DynamicEarth说明](https://github.com/likyoo/DynamicEarth/issues/6)、[DynamicEarth评分](https://github.com/likyoo/DynamicEarth/blob/main/eval/evaluate_second.py)。
