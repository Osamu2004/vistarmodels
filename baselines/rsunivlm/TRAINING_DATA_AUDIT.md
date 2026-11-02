# RSUniVLM训练数据与当前稿件评测范围

核对日期：2026-11-02。本记录依据作者论文公开描述，不等同于对已下载checkpoint训练样本清单的逐项审计。用户先要求任意batch及双卡，再要求先说明训练数据；实现工作保留为未完成，不把开发文件当作已交付版本。

## 一手来源

- 原论文arXiv v2（2024-12-10）：https://arxiv.org/html/2412.05679v2 。第3.3节、第4.1节、附录第7节和表10–12。
- 官方仓库：https://github.com/xuliu-cyber/RSUniVLM 。仓库提供公开checkpoint链接，但当前未取得与该权重逐样本对应的训练manifest。
- 当前本地稿件：`/Users/zhouziyu/code/vistar/paper/sec/4_experiments.tex`和`paper/sec/supplement.tex`。这里只核对事实，不修改稿件或填入新分数。

## 论文公开的训练来源

| 任务/来源 | 明列的数据集 |
|---|---|
| 遥感语义分割 | Potsdam、UAVid、LoveDA、OpenEarthMap、landcover |
| 变化描述与变化检测 | LEVIR-MCI |
| 遥感视觉定位 | DIOR-RSVG、OPT-RSVG |
| 遥感分类、问答、描述 | RSVQA-LR、RSVQA-HR、RSITMD、RSICD、UCM、NWPU、fMoW、RSTeller |
| 已有遥感指令数据 | GeoChat、VRSBench、FIT-RS |
| 通用指令与分割 | ShareGPT4V、All-SeeingV2、DVQA、ChartQA、AI2D、DocVQA、COCO-stuff |

表10将LoveDA和UAVid各列为约8k训练条目，LEVIR-MCI约40k；不能把指令条目数当成唯一原图数或据此推断Train/Val/Test样本身份。正文概述约1.2M指令，但附录遥感985k与通用547k的列示合计并非1.2M；采样、去重和计数口径未完全解释，不擅自消除该不一致。

第一阶段混合遥感和通用数据，全参数微调视觉编码器、连接器和语言模型；第二阶段仅保留遥感数据，按图像/区域/像素三个粒度重采样，只微调G-MoE层。LoveDA、UAVid和LEVIR-MCI也明列于第二阶段表12。基础组件为SigLIP-400M、Qwen2-0.5B和两层MLP。以上是论文声明，不能代替下载权重与具体训练运行的溯源。

## 与我们稿件的关系

当前CVPR稿件的分割集是LoveDA、FLAIR、UAVid、xBD-pre和CHN6-CUG；变化检测是LEVIR-CD和SECOND。VDD不在当前这份稿件的这组表述中，不能据其他工作区正在增加VDD适配就替换UAVid。SECOND检测是DynamicEarth的593对验证集、六个变化类别双向合并，不能用1694对生图测试集代替，也不能直接当二值LEVIR-CD。

- LoveDA、UAVid：论文明确包含该数据集的分割训练，不能给RSUniVLM这两项结果标记“未见过该数据集/零样本跨数据集”。这不等于已证明测试集泄漏；训练划分、影像身份和地理关系仍需核查。
- LEVIR-CD：论文明确训练LEVIR-MCI，但本次未审计与LEVIR-CD的样本/地理重叠，不能仅凭名称相近断言重叠，也不能断言完全独立。
- FLAIR、xBD-pre、CHN6-CUG、SECOND：未在上述明列清单中出现；不能由“未列出”直接证明与所有聚合指令来源及基础预训练数据完全无重叠。
- 该模型可作为统一理解模型的对照，表格/实验说明需披露训练暴露差异；不把所有基线一概称为相同零样本设定。

## 当前开发状态

旧提交支持LoveDA/LEVIR-CD。当前已实现新的七数据集队列、任意正整数batch、双GPU互斥分片及统计合并，见PAPER_QUEUE.md。此为实现和静态核查状态，不等于WSL合成测试、完整模型前向或基准精度已通过；本机不跑实验测试或模型，正式运行由用户WSL完成。本文对训练来源和未审计重叠的结论保持。
