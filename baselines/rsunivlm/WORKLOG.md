# RSUniVLM 接入记录

## 2026-11-01：LoveDA源图裁块对齐Alters

用户要求使用512裁块。默认改为512×512、步长512，按行优先逐块送入原生SigLIP384处理器，严格解码每块24×24七类RLE；每块最近邻恢复512后拼接、裁回原尺寸并评分。不足整块仅右/下补黑，与Alters公开评估路径一致。保留TILE_SIZE=0旧整图路径；新默认输出目录含tile512/v2，几何协议参与缓存身份，逐块文本带坐标并支持部分恢复，所有块成功后才写完整预测。LEVIR-CD整图与双向并集分支不变。

补充WSL待运行回归，覆盖四块拼接/RGB输入隔离、非整除边缘先还原后裁剪、部分失败恢复、旧整图缓存隔离、CLI默认值及非法尺寸。按用户要求不在本机运行测试或模型；只做AST、bash语法与源代码差异核对，回执见validation_tile512_static_20261101.json。真实CUDA推理、格式遵循度、分数及速度仍待WSL验证；Google Drive不可达和本地SigLIP准备状态没有被此代码修改解决。不改论文分数、不启动GPU作业。

## 2026-11-01：WSL启动回执与推理配置核对

用户LoveDA/GPU1启动先报缺gdown，按requirements安装后已报告Transformers4.45.2、tokenizers0.20.3等成功安装。再次启动进入官方Google Drive权重准备阶段，报Errno101 network unreachable，尚未到依赖完整检查、模型加载或推理，不记为推理失败分数。

读取本机已下载checkpoint配置/下载回执并核对固定官方源码：square处理、SigLIP384输入、24×24行式RLE输出、原图尺度最近邻还原，未使用512源图裁块；当前batch1、FP16/eager、固定pixel expert2、greedy/beam1/max_new_tokens8192。checkpoint含作者机器SigLIP绝对路径，后续需在WSL指定有效视觉塔路径；本机已有完整下载回执不代表文件已复制到WSL。更新README配置说明，原模型源码/权重/推理协议均未改变，未执行本机模型或新推理。

## 2026-10-30：统一 checkpoint 的双数据集推理

用户要求将 RSUniVLM 的 LEVIR-CD 变化检测与 LoveDA 分割实现到 vistarmodels。
工作边界与任务目录均为 `/Users/zhouziyu/code/vistar/vistarmodels`；保留仓库原有未提交修改。

核对官方源码固定提交 `66dce1dc68b81a6c62f02fb8e5b306571092a864`，确认双图 `[SEG]`、
GMoE 像素级专家与行式 RLE。原演示解码器只做二值可视化，不能直接得到 LoveDA 七类。
新增严格七类 RLE 解码与显式七类提示词；LEVIR-CD 默认双向并集，也提供单方向选项。
新增每步固定像素专家的适配，避免上游生成准备接口丢失 `granularity`。

数据与指标复用现有 `segearth_ov.protocols`、`levircd_common`、`binary_boundary_wfm`；
默认 LoveDA Val 1,669 张、LEVIR-CD test 原图 128 对，原尺寸评分且保留预测与文本。
缓存隔离模型、源码、提示词和数据身份；格式错误不填背景，部分样本不标为全量。

新增 bootstrap、依赖检查、独立环境依赖文件、统一与分数据集脚本，以及
`tests/test_rsunivlm_protocols.py` 中的 WSL 待运行契约检查。环境需保留当前 GPU 可用的
CUDA 版 torch，固定 Transformers 4.45.2，不安装上游旧训练依赖。

本机只执行 AST、bash 语法、差异和路径静态核查；不下载权重、不运行测试或模型，
不启动 WSL/校园 GPU 作业。静态回执记录在 `validation_static_20261030.json`。
WSL 测试、依赖导入、真实权重加载、七类提示词格式与速度/显存仍待验证，没有新分数。

结构审计提示既有根目录文件较多、无全仓语义 manifest；本次不做仓库整理，
新增内容均归于 baseline、scripts、tools、run_bash、tests 和当前任务的 RUN_QUEUE。
