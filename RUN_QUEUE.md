# 待执行队列

2026-11-01用户规则：以后新训练实验统一20,000个optimizer updates，默认保存10k/20k，具体规则见AGENTS.md。旧训练入口用于新实验前先适配20k并核对实际预算；下方RSUniVLM是公开权重推理，不新增训练。

## RSUniVLM：LoveDA Val 分割与 LEVIR-CD test 变化检测

状态（2026-11-01）：用户WSL已克隆源码并报告安装4.45.2等依赖；LoveDA/GPU1再次启动在Google Drive下载时报Errno101 network unreachable。尚未完成依赖检查、真实模型加载或推理，没有指标。本机只核对配置/源码，不把下载或安装成功当作推理成功。
协议与输出见 [baseline README](baselines/rsunivlm/README.md)。

工作目录假设为 `/root/code/vistarmodels`；如实际克隆目录不同，先改下面的 `cd`。
LoveDA 保存路径沿用 `/root/data/LoveDA`，LEVIR-CD 默认 `/root/data/LEVIR-CD`；
默认模型目录 `/root/data/weight/rsunivlm/llava-qwen-gmoe`。

本次用户授权改为512×512非重叠源图裁块，对齐论文Alters：1024原图→四块512→每块官方SigLIP384输入→每块24×24七类RLE→最近邻还原每块512→拼回1024评分。右/下不足处补黑，先恢复整块再裁掉补齐区。batch1、FP16/eager、pixel expert2、greedy/beam1、最多8192新token保持不变，LEVIR-CD整图双向并集保持不变。

外部阻塞仍待解决：checkpoint内视觉塔为作者机器`/data/liux/siglip-so400m-patch14-384`，需通过`RSUNIVLM_VISION_TOWER`指向本地完整SigLIP目录。本机主权重有完整下载回执，尚未确认复制到WSL。下面默认两套权重已手动准备在指定WSL目录；目录不同时修改变量，不能把公开模型ID填入当前要求本地路径的依赖检查。Google Drive不可达时使用`BOOTSTRAP_RSUNIVLM=0`，不反复触发下载。

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
export RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM
export RSUNIVLM_MODEL_PATH=/root/data/weight/rsunivlm/llava-qwen-gmoe
export RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384
CUDA_VISIBLE_DEVICES=1 python tools/check_rsunivlm_deps.py
```

如环境已存在，跳过 `conda create`。如果原环境已有其他 `llava`，请在独立环境内核对冲突；
适配器会拒绝错误包路径。不要在原 SD3 环境安装官方 `.[train]`。
通过条件：契约检查全部通过，依赖/源码/本地 checkpoint 元数据均通过。失败则停在这里。
这些检查不会加载完整模型；后续冒烟才验证真实 CUDA 推理。
用户WSL已克隆上游源码，当前无需重新bootstrap。新环境若尚无源码/权重，可在网络可达时执行
`bash scripts/bootstrap_rsunivlm.sh`；Google Drive不可达时需先手动准备主权重，另行准备SigLIP。
主权重目录需直接包含`config.json`、`tokenizer.json`、完整权重文件；视觉塔目录需包含其配置与完整权重。

### 2. 两样本冒烟，先检查模型输出（单 GPU，串行）

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_MODEL_PATH=/root/data/weight/rsunivlm/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 SAVE_IMAGES=1 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle24_v2 \
bash run_bash/rsunivlm_loveda.bash
# LoveDA通过后，需要LEVIR-CD时再串行执行：
GPU_IDS=1 BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_levircd.bash
```

LoveDA 输出：`/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle24_v2`，与旧整图目录分开。
LEVIR-CD 输出：`/root/data/experiment/rsunivlm_levircd_test_native_rle24_symmetric_union_v1`。
通过条件：两份 `metrics.json` 均 `complete=true`、`processed_samples=2`、
`is_full_benchmark=false`；检查 `pred_mask`、`pred_rgb` 和 `raw_text` 是否符合预期。
LoveDA两张1024原图应有8份裁块文本，逐图记录`source_tile_size=512,num_tiles=4`；完整预测与GT尺寸相同。
RLE 不合法、显存不足或 CUDA 导入失败则停止，不继续全量。

### 3. 复用冒烟结果，分别跑全量

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_MODEL_PATH=/root/data/weight/rsunivlm/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=0 SAVE_IMAGES=1 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle24_v2 \
bash run_bash/rsunivlm_loveda.bash
# 如需LEVIR-CD，复用步骤1导出的权重路径：
GPU_IDS=1 BOOTSTRAP_RSUNIVLM=0 bash run_bash/rsunivlm_levircd.bash
```

上述命令串行，在用户指定第2张卡（GPU1）运行。另有空闲 GPU 时可在另一终端独立跑 LEVIR-CD，
不要与正在训练的 SD3 争用显存；每个输出目录只允许一个写入进程。
通过条件：LoveDA 1,669 张、LEVIR-CD 128 对，均 `complete=true` 且
`is_full_benchmark=true`，保存逐图、逐类、汇总指标。输出图保留。
未核查训练数据重叠，不标 zero-shot，不自动更新论文。
缓存按几何协议隔离，旧整图预测不能移入新目录复用；中断时只补缺失/无效裁块，全部解码成功后才保存完整原图预测。
`TILE_SIZE=0`可保留整图路径，但需用独立目录，不能标成512裁块结果。本轮只静态核查，WSL契约测试和真实模型推理均待用户执行。

### 备选：一个进程加载一次，依次跑两个数据集

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=1 TILE_SIZE=512 BOOTSTRAP_RSUNIVLM=0 \
LOVEDA_ROOT=/root/data/LoveDA LEVIRCD_ROOT=/root/data/LEVIR-CD \
OUTPUT_DIR=/root/data/experiment/rsunivlm_shared_checkpoint_both_tile512_v2 \
bash run_bash/rsunivlm_eval.bash
```

该模式输出到单独根目录下 `loveda_val`、`levircd_test`，与步骤 2/3 的默认目录不同。
选择一种模式完成全量即可，无需重复跑。没有自动占用校园 GPU 或启动远端进程。
