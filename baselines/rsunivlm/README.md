# RSUniVLM：LoveDA 分割与 LEVIR-CD 变化检测

本目录在同一个官方 `llava-qwen-gmoe` checkpoint 上执行两项任务，不新增训练、LoRA 或任务头。
当前状态：适配代码与 WSL 验证命令已准备；仅做本机静态检查，尚无真实模型加载、GPU 推理或分数。
用户执行入口统一记录在仓库 [RUN_QUEUE.md](../../RUN_QUEUE.md)。

## 来源与适配边界

- [官方源码](https://github.com/xuliu-cyber/RSUniVLM)，固定提交
  `66dce1dc68b81a6c62f02fb8e5b306571092a864`。
- [官方 checkpoint](https://drive.google.com/drive/folders/1TtaoOPmh167gpgHHWRNBMCaA7t_XZ4Vg)，
  默认保存至 `/root/data/weight/rsunivlm/llava-qwen-gmoe`。
- [论文](https://arxiv.org/html/2412.05679)：掩码为 24×24 语义描述符，按行输出 `class*count`。
  官方变化检测定量结果来自 WHU-CD，不能当作 LEVIR-CD 成绩。
- [官方 app.py](https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/app.py)
  支持单图/双图与 `[SEG]`，但演示用解码器把 `others` 以外的标签全部变为白色。
  本适配替换该解码步骤，保留官方加载器、Qwen 对话模板、图像处理和 GMoE 网络。

这两个新评估协议属于 **VISTAR 适配**，不是官方论文已报告的 LoveDA/LEVIR-CD 结果。
训练数据包括 LoveDA 与 LEVIR-MCI；未审计 checkpoint 的实际训练划分及其与 LEVIR-CD
的样本/地理关系，因此不将这两个运行自动标为 zero-shot。

## 推理协议

| 项目 | LoveDA | LEVIR-CD |
|---|---|---|
| 默认数据 | `/root/data/LoveDA/Val/{Urban,Rural}/{images_png,masks_png}` | `/root/data/LEVIR-CD/test/{A,B,label}` |
| 完整数量 | 1,669 张 | 官方原始 test 128 对 |
| 输入 | 一张 RGB | 配准的两张 RGB，按 A→B 顺序 |
| 输出 | 七类互斥语义网格 | 二值建筑变化网格 |
| 类别顺序 | background/building/road/water/barren/forest/agriculture | unchanged/change |
| GT 处理 | 原始 0、255 忽略；1…7→0…6 | 0 未变化；1 或 255 变化 |
| 指标 | mIoU、mF1、mACC、像素准确率、逐类 IoU/F1/ACC、3px 边界 WFm | 变化类 F1/IoU/Precision/Recall、二类 mIoU/OA、3px 边界 WFm |

LoveDA 的新提示词显式列出七类，要求一次输出七类的 24×24 RLE，而不是对二值可视化做颜色映射。
这是本适配新设的多类提示词，实际格式遵循度和分割质量需要 WSL 验证；不以编译或解析器检查代替模型验证。
GT 不进入提示词、图像预处理、网格解码或类别选择。

LEVIR-CD 默认 `CD_DIRECTION=symmetric_union`：用官方演示的建筑变化提示词分别执行
A→B 与 B→A，将两个二值结果取并集；这是为新增和拆除均属于变化而明确设定的协议。
如需官方演示式单方向，设 `CD_DIRECTION=forward`，并使用独立结果目录。
两种协议不混为一个结果。`[CCD]` 用于文字变化描述，不用于生成变化掩码。

两项任务都强制在每个自回归解码步骤保留 `granularity=2`，防止 Transformers 的
`prepare_inputs_for_generation` 丢弃附加参数后退回默认图像级专家；此处是本地适配层绑定，
不改第三方源码或权重。默认 eager attention，无需 FlashAttention 扩展。

图像经过 checkpoint 自带处理器；不增加滑窗、不拼接双时相图像、不使用 SAM 等额外模型。
解析后的 24×24 网格使用最近邻恢复到原始 H×W 后评分，不能将输出上采样解释为原生高精度边界。
显存和速度仍待真实 WSL 运行验证。

## 安装与运行

建议为它单独使用 `vistar_rsunivlm` 环境，保留适合当前显卡的 CUDA 版 torch/torchvision，
安装本仓库 `requirements-rsunivlm.txt`。不要安装官方 `.[train]`：其旧 torch 依赖可能破坏 RTX 50 系列环境。
完整环境准备、测试和依次运行两数据集的可复制命令见 [RUN_QUEUE.md](../../RUN_QUEUE.md)。

```bash
# 先按 RUN_QUEUE 准备独立环境与依赖；两个命令使用完全相同的权重路径。
GPU_IDS=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_loveda.bash
GPU_IDS=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_levircd.bash
# 冒烟运行通过后，省略 MAX_SAMPLES 继续全量，已完成样本会复用。
GPU_IDS=0 bash run_bash/rsunivlm_loveda.bash
GPU_IDS=0 bash run_bash/rsunivlm_levircd.bash
```

也可以执行 `bash run_bash/rsunivlm_eval.bash`：默认 `both`，一个进程先 LoveDA、后 LEVIR-CD，
只加载一次模型。其输出位于 `OUTPUT_DIR/{loveda_val,levircd_test}`。
分别运行的两个脚本则使用各自的独立默认输出目录；如要复用预测，请保持相同目录。
单个输出目录只支持一个进程，用系统文件锁避免两个作业同时写入。

常用覆盖项：`RSUNIVLM_ROOT`、`RSUNIVLM_MODEL_PATH`、`RSUNIVLM_VISION_TOWER`
（本地 SigLIP 目录）、`DATA_ROOT`（单数据集）、`LOVEDA_ROOT`、`LEVIRCD_ROOT`、
`OUTPUT_DIR`、`MAX_SAMPLES`、`CD_DIRECTION`、`DTYPE`、`ATTENTION`、`MAX_NEW_TOKENS`、
`SAVE_IMAGES`、`BOUNDARY_WFM`、`OVERWRITE`。

- `DRY_RUN=1` 仅打印命令，不下载源码/权重。
- `PREFLIGHT=1` 核对源码、权重元数据和数据数量，不实例化模型。
- `BOOTSTRAP_RSUNIVLM=0` 跳过源码/权重准备；`RSUNIVLM_DOWNLOAD_WEIGHTS=0` 只跳过权重下载。
- checkpoint 缺失时通过 gdown 下载官方 Google Drive 文件夹；若无法访问，手动下载该文件夹，
  设置 `RSUNIVLM_MODEL_PATH` 指向直接包含 `config.json`、`tokenizer.json`、完整权重的目录。
  SigLIP 通常还需根据 checkpoint 的 `mm_vision_tower` 从 Hugging Face 下载或提供本地目录。
- 裁块版 LEVIR-CD 的样本数与官方原图不同，需显式 `ALLOW_NONSTANDARD_POPULATION=1`，
  并保持 `is_full_benchmark=false`，不能当作此适配默认原图协议结果。

## 输出与中断恢复

- `metrics.json`：汇总分数、混淆矩阵、数量、`complete`、`is_full_benchmark`；比例分数 0…1。
- `per_class_metrics.csv`：逐类 IoU/F1/ACC。
- `run_config.json` / `input_manifest.json`：固定源码、权重身份、提示词、输入与评价协议。
- `predictions.jsonl`：当前评估逐图分数；`progress.jsonl`：包含重启历史的追加进度记录。
- `pred_mask/*_pred_mask.png`：LoveDA 类别 ID 0…6；LEVIR-CD 为 0/255。
- `pred_rgb/*_pred_rgb.png`：LoveDA 固定调色板；LEVIR-CD 黑白变化图。
- `input`、`gt_mask`、`gt_rgb`、`overlay`：LoveDA 可视化；LEVIR-CD 为
  `input_A`、`input_B`、`gt_mask`、`overlay_A`、`overlay_B`、`error_map`。
- `raw_text`：每个请求的原始模型文本，即使 RLE 解析失败也保留。
- `receipts`：已解码预测的校验摘要；`errors`：错误与堆栈。

默认保存图片。`SAVE_IMAGES=0` 仍保留预测 ID 掩码、原始文本和指标，只跳过额外可视化。
完整的行数、行宽、类别名均须合法，失败立即停止且 `complete=false`，不会用空背景补齐。
重启时检查模型/源码/提示词/完整输入清单以及逐图掩码摘要；损坏预测可从已存原始文本重新解析，
中断的双方向任务只补缺失方向。修改 checkpoint、输入或协议后要用新目录或显式 `OVERWRITE=1`。

权重身份使用小型元数据 SHA-256 以及权重文件大小/mtime；它是恢复兼容性标识，
不是全权重密码学认证。当前没有真实指标可写入论文。
