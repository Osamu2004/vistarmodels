# 待执行队列

2026-11-01用户规则：以后新训练实验统一20,000个optimizer updates，默认保存10k/20k，具体规则见AGENTS.md。旧训练入口用于新实验前先适配20k并核对实际预算；下方RSUniVLM是公开权重推理，不新增训练。

## RSUniVLM：LoveDA Val 分割与 LEVIR-CD test 变化检测

状态（2026-10-30）：代码已准备，仅本机静态检查；WSL 契约检查、依赖检查、模型加载、
冒烟和全量推理均未启动，没有指标。本队列只给出用户执行命令，不将计划当作运行回执。
协议与输出见 [baseline README](baselines/rsunivlm/README.md)。

工作目录假设为 `/root/code/vistarmodels`；如实际克隆目录不同，先改下面的 `cd`。
LoveDA 保存路径沿用 `/root/data/LoveDA`，LEVIR-CD 默认 `/root/data/LEVIR-CD`；
默认模型目录 `/root/data/weight/rsunivlm/llava-qwen-gmoe`。

### 1. 单独环境与契约检查（先执行）

以下复用用户已有 `vistar_sd3` 环境的 CUDA torch/torchvision，克隆成独立环境，
避免固定的 Transformers 版本影响正在使用的 SD3 环境。仅在新环境不存在时创建。

```bash
cd /root/code/vistarmodels
git pull --ff-only
source /root/anaconda3/etc/profile.d/conda.sh
conda create -n vistar_rsunivlm --clone vistar_sd3 -y
conda activate vistar_rsunivlm
python -m pip install -r requirements-rsunivlm.txt
python -m pytest tests/test_rsunivlm_protocols.py -q
bash scripts/bootstrap_rsunivlm.sh
python tools/check_rsunivlm_deps.py
```

如环境已存在，跳过 `conda create`。如果原环境已有其他 `llava`，请在独立环境内核对冲突；
适配器会拒绝错误包路径。不要在原 SD3 环境安装官方 `.[train]`。
通过条件：契约检查全部通过，依赖/源码/本地 checkpoint 元数据均通过。失败则停在这里。
这些检查不会加载完整模型；后续冒烟才验证真实 CUDA 推理。
首次 bootstrap 需要访问 GitHub、Google Drive；视觉塔按 checkpoint 配置访问 Hugging Face，
可设 `RSUNIVLM_VISION_TOWER=/root/data/weight/<本地SigLIP目录>` 使用本地文件。

### 2. 两样本冒烟，先检查模型输出（单 GPU，串行）

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_loveda.bash
GPU_IDS=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_levircd.bash
```

LoveDA 输出：`/root/data/experiment/rsunivlm_loveda_val_native_rle24_v1`。
LEVIR-CD 输出：`/root/data/experiment/rsunivlm_levircd_test_native_rle24_symmetric_union_v1`。
通过条件：两份 `metrics.json` 均 `complete=true`、`processed_samples=2`、
`is_full_benchmark=false`；检查 `pred_mask`、`pred_rgb` 和 `raw_text` 是否符合预期。
RLE 不合法、显存不足或 CUDA 导入失败则停止，不继续全量。

### 3. 复用冒烟结果，分别跑全量

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=0 bash run_bash/rsunivlm_loveda.bash
GPU_IDS=0 bash run_bash/rsunivlm_levircd.bash
```

上述命令串行。另有空闲 GPU 时可在另一终端独立跑 LEVIR-CD，例如 `GPU_IDS=1`，
不要与正在训练的 SD3 争用显存；每个输出目录只允许一个写入进程。
通过条件：LoveDA 1,669 张、LEVIR-CD 128 对，均 `complete=true` 且
`is_full_benchmark=true`，保存逐图、逐类、汇总指标。输出图保留。
未核查训练数据重叠，不标 zero-shot，不自动更新论文。

### 备选：一个进程加载一次，依次跑两个数据集

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=0 LOVEDA_ROOT=/root/data/LoveDA LEVIRCD_ROOT=/root/data/LEVIR-CD \
OUTPUT_DIR=/root/data/experiment/rsunivlm_shared_checkpoint_both_v1 \
bash run_bash/rsunivlm_eval.bash
```

该模式输出到单独根目录下 `loveda_val`、`levircd_test`，与步骤 2/3 的默认目录不同。
选择一种模式完成全量即可，无需重复跑。没有自动占用校园 GPU 或启动远端进程。
