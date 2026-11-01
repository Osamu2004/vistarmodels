# 待执行队列

2026-11-01用户规则：以后新训练实验统一20,000个optimizer updates，默认保存10k/20k，具体规则见AGENTS.md。旧训练入口用于新实验前先适配20k并核对实际预算；下方RSUniVLM是公开权重推理，不新增训练。

## RSUniVLM：LoveDA Val 分割与 LEVIR-CD test 变化检测

状态（2026-11-01最新WSL附件498838a7）：已生成模型文本，首张影像内某个裁块因旧解析器要求24行、收到23行而停止；没有完成整图评分。核对官方app.py后改为按实际矩形行列解码与OpenCV最近邻恢复，保留一致行宽、正计数、已知类别和token截断检查；实际尺寸与EOS/预算写入原文记录。补attention_mask、greedy中性参数与加载后meta参数核对。协议版本2，LoveDA改用下方v3独立输出目录；本机只静态检查，真实新协议结果待WSL回执。
协议与输出见 [baseline README](baselines/rsunivlm/README.md)。

工作目录假设为 `/root/code/vistarmodels`；如实际克隆目录不同，先改下面的 `cd`。
LoveDA 保存路径沿用 `/root/data/LoveDA`，LEVIR-CD 默认 `/root/data/LEVIR-CD`；
用户已解压的主模型目录为 `/root/data/weight/llava-qwen-gmoe`；本队列显式覆盖脚本旧默认路径。

本次用户授权改为512×512非重叠源图裁块，对齐论文Alters：1024原图→四块512→每块官方SigLIP384输入→每块七类RLE实际矩形（名义24×24）→OpenCV最近邻还原每块512→拼回1024评分。右/下不足处补黑，先恢复整块再裁掉补齐区。batch1、FP16/eager、pixel expert2、greedy/beam1、最多8192新token保持不变，LEVIR-CD整图双向并集保持不变。

权重路径：主模型使用`/root/data/weight/llava-qwen-gmoe`，视觉塔使用用户刚报告解压的`/root/data/weight/siglip-so400m-patch14-384`。checkpoint原配置中作者的`/data/liux/siglip-so400m-patch14-384`由RSUNIVLM_VISION_TOWER覆盖，无需改主checkpoint配置。主模型元数据通过，SigLIP在Mac的大小/哈希已校验，WSL解压内容与真实模型加载仍待启动回执。保留`BOOTSTRAP_RSUNIVLM=0`，不反复触发主权重下载。

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
export RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe
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

### 1.1 WSL下载主权重与SigLIP（2026-11-01补充）

用户请求可复制下载命令；本轮只核对官方下载源，不代替用户下载、不加载模型。用户随后报告主模型已解压到`/root/data/weight/llava-qwen-gmoe`；该目录文件完整时无需再执行主模型下载，仅准备尚缺的SigLIP。
在已有`vistar_rsunivlm`环境执行，主权重来自官方仓库所链Google Drive，视觉塔来自`google/siglip-so400m-patch14-384`。
`huggingface_hub`固定0.36.0以满足现有Transformers4.45.2的`<1.0`约束。

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m pip install 'gdown>=5.2,<6' 'huggingface_hub==0.36.0'

mkdir -p /root/data/weight/llava-qwen-gmoe
python -m gdown --folder --continue \
  'https://drive.google.com/drive/folders/1TtaoOPmh167gpgHHWRNBMCaA7t_XZ4Vg' \
  -O /root/data/weight/llava-qwen-gmoe
```

主模型下载成功后下载视觉塔：

```bash
python -m pip install 'huggingface_hub==0.36.0'
HF_HUB_DOWNLOAD_TIMEOUT=120 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/siglip-so400m-patch14-384',
    local_dir='/root/data/weight/siglip-so400m-patch14-384',
    max_workers=2,
)
PY
export RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe
export RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384
CUDA_VISIBLE_DEVICES=1 python tools/check_rsunivlm_deps.py
```

通过条件：两份下载无报错，两个目录均含完整配置/权重，最后依赖与元数据检查通过；此时仍未验证模型前向。
最新用户回执已通过主模型元数据/依赖/CUDA检查，当前只运行上面的视觉塔下载与检查；无需再次运行gdown主模型下载。
若Google Drive仍报Errno101，则停止推理，先恢复WSL网络/使用实际代理，或将本机已下载主模型完整复制到对应WSL目录。
来源：https://github.com/xuliu-cyber/RSUniVLM ，https://huggingface.co/google/siglip-so400m-patch14-384 。

### 1.2 按用户要求使用HF-Mirror下载SigLIP

用户改为索取第三方镜像下载命令，保存路径沿用其粘贴的WSL目录；未在本机启动下载。通过HF_ENDPOINT切换镜像，repo_id仍为Google原版SigLIP。公共模型显式token=False，不向镜像发送本机保存的HF令牌。本段尚无WSL下载完成回执。

```bash
python -m pip install 'huggingface_hub==0.36.0'

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 \
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/siglip-so400m-patch14-384',
    local_dir='/root/data/weight/siglip-so400m-patch14-384',
    max_workers=2,
    token=False,
)
PY
```

镜像使用说明：https://hf-mirror.com/ 。下载完成后仍按上段核对config.json与完整权重，再启动推理；不能仅因镜像命令已提供而记为下载完成。

### 1.3 用户授权本机下载SigLIP（已完成，用户报告已解压到WSL）

用户要求由本对话实际下载至`/Users/zhouziyu/Downloads/weight/siglip-so400m-patch14-384`。已核实磁盘可用约113GiB；官方与HF-Mirror快照均为`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`，model.safetensors为3,511,950,624字节，官方SHA256为`ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`。

SDK镜像下载因文件元数据兼容报错，随后切换HF-Mirror resolve URL的HTTP流式下载。PID99478、工具会话44487已正常退出（exit0），下载与校验耗时221.32秒，全部9个官方文件共3,515,156,215字节完整。LFS文件按官方SHA256、普通文件按官方Git blob SHA1核对，JSON及safetensors数据偏移检查通过。完成回执：`/Users/zhouziyu/Downloads/weight/siglip-so400m-patch14-384/download_receipt.json`（complete=true）；脚本：`/Users/zhouziyu/code/vistar/output/downloads/siglip_so400m_20261101/download_siglip.py`。这是本机下载与哈希校验完成；随后用户报告已解压到WSL的`/root/data/weight/siglip-so400m-patch14-384`，WSL实际层级/哈希与模型加载尚待核对，未产生GPU推理结果。

### 1.4 历史处理：创建本地源码副本（当前无需再执行）

用户现已明确取消已跟踪文件修改检查；更新本次代码后直接使用现有源码目录，本小节仅保留此前处理记录，不再作为启动条件。固定提交版本检查保留。

用户最新日志显示权重目录已补齐，当前是源码完整性检查失败。检查器读取固定commit与`git status --porcelain --untracked-files=no`；未跟踪的pycache不是该报错依据。尚无diff回执，不推断具体文件或更改原因。不关闭检查，不reset/clean/stash原目录；只从其现有Git对象创建新副本，保留全部原修改。用户WSL执行：

```bash
cd /root/code/vistarmodels
git -C third_party/RSUniVLM status --short --untracked-files=no
git -C third_party/RSUniVLM diff HEAD --stat

if [ ! -e /root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d ]; then
  git clone --no-hardlinks --no-checkout \
    /root/code/vistarmodels/third_party/RSUniVLM \
    /root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d
fi
git -C /root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
  checkout --detach 66dce1dc68b81a6c62f02fb8e5b306571092a864
export RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d
```

通过条件：新副本指向固定提交，原检查器通过；若新目录已有更改也保留并让检查器报错，不强制覆盖。保持BOOTSTRAP_RSUNIVLM=0，使用下方显式传入干净源码路径的命令。原源码改动有可能包含用户自定义适配，尚未核对；此路径用于原官方源码推理，不把旧改动默认为无效或删除它们。此方案已核对源码入口，未在本机执行克隆/模型或远端作业，WSL效果待回执。

### 1.5 适配器不认识512参数：导入离线更新

用户最新回执确认第三方源码和依赖检查已通过，但外层vistarmodels的evaluate.py仍是旧CLI。512功能已在本地e25159f实现，GitHub推送此前连接失败/超时，WSL当前未运行到该实现；具体WSL HEAD未核验。离线包包括该实现与最新运行记录，不包含权重、不改第三方源码或数据。将`rsunivlm_tile512_wsl_update_20261101.bundle`复制到WSL的`/root/code/vistarmodels`后：

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
git bundle verify ./rsunivlm_tile512_wsl_update_20261101.bundle
git pull --ff-only ./rsunivlm_tile512_wsl_update_20261101.bundle main
python baselines/rsunivlm/evaluate.py --help
```

通过条件：verify与快进导入成功，帮助中出现`--loveda_tile_size {0,512}`。随后按2重跑两原图。若有本地修改或非快进，保留Git报错，不reset/clean/强制覆盖。包基于已有RSUniVLM集成提交e5c58f8；正式WSL单元回归按1执行，本机仍只做静态和bundle完整性检查，未运行测试或模型。

### 1.6 本次RLE矩形协议修复（当前需更新）

先在WSL执行`git pull --ff-only`，可运行`python -m pytest tests/test_rsunivlm_protocols.py -q`完成协议回归。解码对齐官方app.py的实际矩形布局与cv2.INTER_NEAREST像素对齐；23×24等行宽一致且正常结束的描述符允许直接还原，未通过格式或预算检查仍停止，不补行/背景。保留24×24提示词、固定权重、512源图裁块/384视觉输入与既有评分数学实现。新输出目录v3与旧v2分开；旧失败文本保留。不要在旧v2目录中使用OVERWRITE混跑。

加载日志会额外显示`[RSUniVLM] Loaded model: no remaining meta parameters.`；官方被替换为Identity的SigLIP pooling head可能在初始加载时仍打印meta-copy警告，此日志不能替代物化检查。实际有meta参数时列名停止，不用空权重补齐。raw_text记录grid_shape及generation的sequence_tokens/ended_with_eos/hit_token_limit；metrics.json汇总descriptor_shape_counts，重启恢复也保留这些计数。达到MAX_NEW_TOKENS且无EOS的文本不作为完整矩形评分。当前修复只有静态回执，未在本机运行模型或测试。

### 2. 两样本冒烟，先检查模型输出（单 GPU，串行）

本次按用户要求取消工作区修改拦截：在WSL先`git pull --ff-only`更新。已有`RSUniVLM_clean_66dce1d`副本继续使用，无需重新克隆；固定提交不匹配仍会报错。可在WSL执行`python -m pytest tests/test_rsunivlm_protocols.py -q`检查修改后的行为，本机不运行测试。

需已导入`e25159f`（或后续版本）的512裁块代码。命令同时显式传入`--loveda_tile_size 512`，旧代码会报未识别参数并停止，避免把旧整图运行误当成512裁块。主权重使用用户已解压的`/root/data/weight/llava-qwen-gmoe`；SigLIP也须完整。

用户已报告SigLIP解压至相应WSL目录；先运行`ls -lh /root/data/weight/siglip-so400m-patch14-384/{config.json,model.safetensors}`核对没有多套一层文件夹，再执行下方原图两样本命令。推理使用GPU1，LongCat-Edit当前提供的训练命令使用GPU0；不据此假定远端进程已启动或资源一定空闲。

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 SAVE_IMAGES=1 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v3 \
bash run_bash/rsunivlm_loveda.bash --loveda_tile_size 512
# LoveDA通过后，需要LEVIR-CD时再串行执行：
GPU_IDS=1 BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_levircd.bash
```

LoveDA 输出：`/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v3`，与旧整图目录分开。
LEVIR-CD 输出：`/root/data/experiment/rsunivlm_levircd_test_native_rle_symmetric_union_v2`。
通过条件：两份 `metrics.json` 均 `complete=true`、`processed_samples=2`、
`is_full_benchmark=false`；检查 `pred_mask`、`pred_rgb` 和 `raw_text` 是否符合预期。
LoveDA两张1024原图应有8份裁块文本，逐图记录`source_tile_size=512,num_tiles=4`；完整预测与GT尺寸相同，metrics.json的descriptor_shape_counts总数应为8；具体形状不强制全部24×24。
RLE 不合法、显存不足或 CUDA 导入失败则停止，不继续全量。

### 3. 复用冒烟结果，分别跑全量

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=0 SAVE_IMAGES=1 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v3 \
bash run_bash/rsunivlm_loveda.bash --loveda_tile_size 512
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
OUTPUT_DIR=/root/data/experiment/rsunivlm_shared_checkpoint_both_tile512_rle_v3 \
bash run_bash/rsunivlm_eval.bash
```

该模式输出到单独根目录下 `loveda_val`、`levircd_test`，与步骤 2/3 的默认目录不同。
选择一种模式完成全量即可，无需重复跑。没有自动占用校园 GPU 或启动远端进程。
