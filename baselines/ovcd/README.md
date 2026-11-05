# 论文变化检测：DynamicEarth 与 OpenDPR

本节点在现有vistarmodels内接入两个公开方法，不做训练。当前完成代码与来源核对，**尚未取得WSL依赖、权重加载、CUDA前向或完整基准回执**。以前表中的DynamicEarth/OpenDPR数值不是本次运行结果。现有DynamicEarth的LEVIR整图入口继续保留。

## 固定的任务与输出

- LEVIR-CD官方test128；原始非零灰度都按前景，和Alters现用GT读取方式一致。
- SECOND与Alters/RSUniVLM相同的593清单及SHA，不能替换为生成的Test1694。
- 默认源tile512、黑色右下补齐、原图拼接后评分；SECOND原图512时即单块。LEVIR默认4块，是明确的VISTAR适配，不冒充作者原1024整图结果；需要作者整图对照可用`TILE_SIZE=0`和独立输出目录。
- 模型保持单次有序双图，不加入反向推理。SECOND输出六张可重叠二值成员图，再逐类累计像素TP/TN/FP/FN、平均六类IoU/F1。WFm仅使用固定类ID升序、后类覆盖的0..6边界视图，GT同规则；复用LoveDA/RSUniVLM的3px函数。
- 保存`pred_mask/*_class_XX.png`、掩码哈希回执及逐图/整体JSON；不复制输入、GT、overlay或彩色图。相同命令按源码/输入/权重/原型/掩码内容身份续跑；损坏掩码重推，GT仅用于评分。
- DataLoader只预读CPU RGB，默认2 workers/prefetch1；每个方法的SAM/DINO官方前向逐对运行，不能把DataLoader进程数说成GPU batch。

## 方法来源

[DynamicEarth](https://github.com/likyoo/DynamicEarth/tree/c9ffd90cafbd791cd75a48a5717a902966c2436c)：固定SAM-DINO-SegEarth-OV的M-C-I变体。SECOND六个官方脚本的背景/前景词组、points_per_side、阈值和instance开关逐项存入`configs/ovcd/dynamicearth_second_mci.json`，包含原脚本SHA。不是APE或DINOv2版本，也不是全六类统一145阈值。共享冻结视觉模型/类别文本特征，保留原归一化与识别函数。完整SimFeatUp仍启用，不把FEATURE_UP=0当正式结果。新Torch的Lightning反序列化兼容只针对已明确指定的SimFeatUp文件，映射CPU读取，不改张量内容或预测公式。

[OpenDPR](https://github.com/guoqi2002/OpenDPR/tree/bdf12a3d3e5adccd4e2c166a0c3a5d9f5a19c313)：固定training-free版，SAM ViT-H + DINOv2-B/14 +作者支持集，不替换成OpenDPR-W/S2C。SECOND按官方infer_all映射作者class0..5到基准1..6；LEVIR采用background/building、最多5个原型，SECOND六类最多10个。DINOv2本地加载绕过上游硬编码占位路径；保留不resize/不center-crop和官方原型/识别函数。共享同一冻结encoder避免重复载入相同权重，不做微调。支持集图片和对应mask必须完整，每类至少3对，实际构建还会核查所有类别原型存在。

## 环境和运行

命令完整记录在根`RUN_QUEUE.md`的OVCD-PAPER-VAL593。建议从已能在5090上工作的环境克隆`vistar_opendpr`和`vistar_dynamicearth`，两者分开安装；保留现有CUDA Torch/torchvision，不按旧论文安装Torch2.1/cu118。OpenDPR需求文件不固定Torch；DynamicEarth沿用已有OpenMMLab/SimFeatUp依赖，编译扩展必须与实际Torch/CUDA一致，导入检查不能代替真实前向。

启动顺序：bootstrap源码/权重→导入/扩展及CUDA检查→全人口plan_only→独立目录2对真实smoke→正式128+593。bootstrap会显式下载SAM、DINO/DINOv2及DynamicEarth上采样权重，已有完整文件复用；OpenDPR官方支持集随源码取得，无需生成新原型图片。CLIP视觉权重由DynamicEarth现有上游加载器解析本地缓存或首次下载，真实smoke会验证这条路径。离线运行前必须完成首次权重解析。

两个方法在独立进程使用同名的上游`dynamic_earth`包，避免相互污染。两张卡并行队列默认DynamicEarth/GPU0和OpenDPR/GPU1。相同协议可调整DataLoader worker数量续跑。具体包导入、编译扩展、权重严格加载或前向任一失败均明确报错，不能以空预测顶替失败来宣称完整运行。

范围状态、来源与待验证项见[WORKLOG.md](WORKLOG.md)。
