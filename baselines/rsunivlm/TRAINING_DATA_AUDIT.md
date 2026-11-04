# RSUniVLM训练数据与当前稿件评测范围

核对日期：2026-11-04（初版2026-11-02）。本记录依据作者论文公开描述，不等同于对已下载checkpoint训练样本清单的逐项审计。最新已收到用户WSL的19项回归通过与七集全量合并评分回执，见results/fused_scores_training_marked_20261104.md；训练来源核查与实际评测回执分别记录。

## 一手来源

- 原论文arXiv v2（2024-12-10）：https://arxiv.org/html/2412.05679v2 。第3.3节、第4.1节、附录第7节和表10–12。
- 官方仓库：https://github.com/xuliu-cyber/RSUniVLM 。仓库提供公开checkpoint链接，但当前未取得与该权重逐样本对应的训练manifest。
- LEVIR-MCI原论文：https://arxiv.org/html/2403.19646v3#S3 ，第III节明确其扩展LEVIR-CC，为相同的双时相影像增加变化掩码。
- LEVIR-CC原论文：https://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf ，第III.A节明确其影像主要来自LEVIR-CD，裁为256×256。
- 当前本地稿件：`/Users/zhouziyu/code/vistar/paper/sec/4_experiments.tex`和`paper/sec/supplement.tex`。本次归档新分数与标记到独立表和论文交接，没有自动修改正式TeX主表。

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

当前CVPR稿件的分割集是LoveDA、FLAIR、UAVid、xBD-pre和CHN6-CUG；变化检测是LEVIR-CD和SECOND。VDD不在当前这份稿件的这组表述中，不能据其他工作区正在增加VDD适配就替换UAVid。SECOND检测是DynamicEarth的593对验证集，不能用1694对生图测试集代替，也不能直接当二值LEVIR-CD。当前RSUniVLM适配为六个可重叠的无方向类别查询、固定一个图像顺序；不因Alters原稿的双向并集描述改变此协议。

- LoveDA、UAVid：论文明确包含该数据集的分割训练，不能给RSUniVLM这两项结果标记“未见过该数据集/零样本跨数据集”。这不等于已证明测试集泄漏；训练划分、影像身份和地理关系仍需核查。
- LEVIR-CD：本次找到可核验的影像来源链：训练LEVIR-MCI → 扩展LEVIR-CC → 影像主要源于LEVIR-CD。故需标明相关训练来源暴露，不能称为完全独立的未见数据。具体已下载checkpoint训练样本与当前128对官方test之间的重叠尚未逐项审计；来源关系不等于测试泄漏已经成立。
- FLAIR、xBD-pre、CHN6-CUG、SECOND：未在上述明列清单中出现；不能由“未列出”直接证明与所有聚合指令来源及基础预训练数据完全无重叠。
- 该模型可作为统一理解模型的对照，表格/实验说明需披露训练暴露差异；不把所有基线一概称为相同零样本设定。

## 当前开发状态

后续边界补齐也已收到WSL完成回执：7822e5d的6项回归通过，LEVIR-CD WFm12.08%、SECOND WFm7.98%，721对失败0、原IoU核对一致、七集complete=True。当前完整显示表改用results/complete_scores_training_marked_20261104.csv/md；原始全精度标量见receipts/wsl_complete_all7_boundary_scores_20261104.json。训练来源结论和星号定义保持。

用户已提供七集完整评分终端回执，LoveDA/FLAIR/UAVid本次固定类别顺序合并后为标准mIoU41.34/18.94/39.17和WFm11.21/15.06/14.52；其余四项继承源评分。19项WSL回归通过，全部20,196张/对计分、失败0。本机没有执行模型或重新评分，也未独立读取WSL原始plan/summary；完整精度标量已从用户CSV归档，不再将这次结果称作只有静态检查。

## 星号标注规则

`*`表示训练使用该基准或其源影像派生数据；因此LoveDA、UAVid标`*`，LEVIR-CD标`*†`。`†`专指LEVIR-MCI→LEVIR-CC→主要LEVIR-CD影像的派生来源，避免误写成明确直接用原LEVIR-CD划分训练。FLAIR、xBD-pre、CHN6-CUG、SECOND在明列来源中未出现，不加星，但不能将其视为已经完成全预训练/聚合指令影像排重。星号按数据集暴露定义，不声称任何当前评测test已经被训练使用。

当前完整标记表和标量：results/complete_scores_training_marked_20261104.csv/md；结构化来源：receipts/wsl_complete_all7_boundary_scores_20261104.json。此前仅三集重计的fused_scores表和wsl_fused回执保留作为历史版本。数值字段保留原完整精度，星号只放在显示/metadata中。
