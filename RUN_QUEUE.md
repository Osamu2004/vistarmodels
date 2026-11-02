# 待执行队列

2026-11-01用户规则：以后新训练实验统一20,000个optimizer updates，默认保存10k/20k，具体规则见AGENTS.md。旧训练入口用于新实验前先适配20k并核对实际预算；下方RSUniVLM是公开权重推理，不新增训练。

## RSUniVLM：LoveDA Val 分割与 LEVIR-CD test 变化检测

### 最新WSL进度：回退与格式恢复已运行到166/1669

用户回执时任务正在继续：首次实际batch3有1/3与单图不一致，已统一采用单图参考并固定batch1；Val_Rural_2685成功完成，越过旧163张断点。该图4裁块复用1份原文，另外2个缺计数water回答经过既有重试后恢复，随后Val_Rural_2686/2687均完成。后两图生成分别6.50s/8.43s，verify=0；首次27.33s含核验9.71s，不作为稳定吞吐。进度条剩余约3h20–30m，仅为当时动态估计。

这确认WSL恢复路径实际运行，不等于完整1669张或合批提速通过。cached=0表示未复用现成PNG，不能据此认定v4原文未复用；miou字段是单张图值，不是全量汇总。回执归档`baselines/rsunivlm/receipts/wsl_loveda_v5_serial_fallback_progress166_20261102.json`。状态为用户回执时运行中，本机没有WSL会话或PID，不冒充实时监控。保留运行中的代码/配置，无需为这份文档记录更新当前作业；最终仍需processed_samples=1669、complete=true和is_full_benchmark=true。

### 最新恢复：合批与单图不一致时，使用单图答案继续

用户WSL回执：v5完成163/1669后，在首次新合批的单图核验处停止，`Batched/single predictions differ`。已加载模型且无残留meta；本次触发终止的是核验分支。没有取得`batch_verification.json`原文，不能认定是浮点差异还是合批实现问题，也不知道当时实际batch是否为4（未完成子集可能更小）。失败回执在`baselines/rsunivlm/receipts/wsl_loveda_v5_batch_mismatch_20261102.json`。

当前修复：核验不一致时，这一组全部使用已经计算出的原单图答案，丢弃其合批候选用于正式评分的资格，后续effective batch固定1；不按GT、合法性或分数在两者中择优。核验原文/单图原文/裁块身份和回退原因保留于`batch_verification.json`，同协议重启恢复batch1，不重复失败的合批探测。原单图答案非法时仍走既有两次格式重试，预算耗尽仍停止，不能伪造掩码。最终metrics明确记录请求batch、effective batch、serial_fallback；回退后的结果不能称为成功batch4加速。

本机只静态检查，新增WSL合成回归尚未运行。旧科学包/评分源码未改；新增回退策略与代码哈希改变协议，沿用下方OUTPUT_DIR时会自动选择独立哈希目录，再导入已核验v4原文。前163张避免重复模型生成，但仍会重建掩码和评分。旧v5的核验文件保留供差异调查，不直接移植未知协议分数。

新离线包：`/Users/zhouziyu/code/vistar/output/offline/rsunivlm_serial_fallback_20261102.bundle`，含完整main历史，不含权重和缓存。在WSL复制到`/root/code/vistarmodels`后：

~~~bash
cd /root/code/vistarmodels
git bundle verify ./rsunivlm_serial_fallback_20261102.bundle
git pull --ff-only ./rsunivlm_serial_fallback_20261102.bundle main
git log -1 --oneline
~~~

然后运行下方完整batch4命令（保持VERIFY_BATCH=1，不关闭核验）。如果希望从头采用原单图路径，可将TILE_BATCH_SIZE改1并用独立`..._v5_bs1`输出，仍保留别名及v4导入。依赖/环境/GPU1单进程/只保存掩码与全1669完成条件不变。后续用户已回传166张运行中回执，见本节顶部；全量指标仍待回执，Git离线导入校验不代替模型验证。

### 历史离线更新：包含343db0f目录冲突修复的完整main包

用户要求提供离线包，产物为本机`/Users/zhouziyu/code/vistar/output/offline/rsunivlm_batch4_resume_20261102.bundle`。包含当前已提交main的完整历史及本段交接，不依赖GitHub或增量包前置提交；不包含未提交改动、预训练权重、Qwen缓存或数据集。实际包tip、字节数、SHA256及Git导入校验结果记录于旁边的`rsunivlm_batch4_resume_20261102.receipt.json`；这些只验证代码传输，不等于WSL模型或指标已通过。

将bundle复制至WSL的`/root/code/vistarmodels`，在WSL终端执行：

~~~bash
cd /root/code/vistarmodels
git bundle verify ./rsunivlm_batch4_resume_20261102.bundle
git pull --ff-only ./rsunivlm_batch4_resume_20261102.bundle main
git log -1 --oneline
~~~

通过条件：bundle校验与快进导入成功；实际HEAD与交付回执tip一致。若非快进或本地文件冲突，保留报错，不reset/clean/覆盖。随后继续下方batch4推理命令，无需在线git pull；旧v5冲突时会选独立协议目录，v4原文身份验证保持。状态：仅打包和Git对象完整性核验；无本机测试/模型、无远端作业回执。

### 已有恢复：v5目标目录协议冲突，自动保留并选择独立目录

用户WSL已更新批量入口，但在模型加载前停于`Batch output already belongs to a different protocol`。按调用次序，v4的源协议及输入清单检查已通过；失败在现有v5目标目录身份检查。未取得旧v5的run_config原文，因此具体差异字段尚不确定，不直接归因于模型/数据或某次代码更新。

新入口会打印具体不同的字段（例如扩展源码哈希、batch、dtype等），保留原目录，改用`OUTPUT_DIR`同级的`<原名>_p<当前协议前12位哈希>`。相同协议重启仍选同一路径；若该哈希目录也有身份冲突则继续拒绝覆盖。actual目录记录在终端`Actual OUTPUT_DIR=`及`output_resolution.json`。旧v4导入的所有身份/提示/几何检查仍保留，不关闭检查、不删除文件、不复制旧v5分数。

执行：WSL先`cd /root/code/vistarmodels`、`git pull --ff-only`，随后原样运行下方batch4命令即可；不要删除现有v5，也不用改`OVERWRITE`。依赖、GPU1单进程、512裁块、只保存掩码等约定不变。可选WSL回归仍是`python -m pytest tests/test_rsunivlm_batch_runtime.py -q`；新覆盖空目录、匹配身份、冲突目录保留、确定性路径、缺失配置、哈希目标再冲突及精确字段诊断。本机仅静态检查，未执行测试/模型，尚无修正后WSL回执。完成条件仍为全1669张完整评分。

### 当前：v4在163张后遇bareland中断，v5映射恢复＋512裁块batch4

状态：用户已提供v4与v5的163张中断回执；当前新回退代码只经本机静态检查，未运行本机测试或模型，修正后WSL执行待回执。不要把旧九图或163张进度写成完整基准成绩。详见`baselines/rsunivlm/BATCHING.md`和本节顶部。

依赖/环境：沿用已安装的`vistar_rsunivlm`，不重装torch。数据`/root/data/LoveDA`、主权重和本地SigLIP路径均沿用用户已有目录。终端为WSL，使用空闲物理GPU1、单进程；同一个输出根只开一个作业。先前v4需已停止，导入器会检查源运行锁；本机没有该WSL连接，不代为启动。

~~~bash
cd /root/code/vistarmodels
git pull --ff-only
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

# 可选：仅在WSL运行合成回归；本机未执行这些测试
python -m pytest tests/test_rsunivlm_batch_runtime.py -q

GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 TILE_BATCH_SIZE=4 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 FORMAT_RETRIES=2 MAX_SAMPLES=0 SAVE_IMAGES=0 \
VERIFY_BATCH=1 \
IMPORT_V4_ROOT=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v5_bs4 \
bash run_bash/rsunivlm_loveda.bash
~~~

`bareland`/`bare land`明确映射到`barren`，未知词和缺计数的`water`仍重试，不猜类别。batch4是四个独立512裁块，不是四图拼成一个问题；OOM自动降2/1，batch1仍OOM则停。原模型/384视觉输入/greedy/8192 token/两次固定格式重试保持。首次实际batch大小>1会与单图输出检查，首批额外耗时；不匹配时使用该组全部单图参考并固定batch1，核验仍保留在`batch_verification.json`。

先核验旧checkpoint/协议/输入元数据/裁块与提示身份，通过后只导入旧raw_text，按新增别名选择最早合法回答，重新构建掩码和评分，保留全部历史尝试。旧163张因此不需再次生成；重评分和读盘仍耗时。保留v4，不复制旧分数，不用OVERWRITE。源身份不匹配、无法选择安全目标目录、格式预算耗尽或batch1仍OOM则停，不能跳图。batch输出不一致按本节顶部回退单图。全量成绩待用户回执。

产物：v5的`pred_mask/`、`raw_text/`、`receipts/`、`metrics.json`、`per_class_metrics.csv`、`predictions.jsonl`、`cache_import.json`、`batch_verification.json`、`scoring_timing.jsonl`。SAVE_IMAGES=0不存其他图片。通过条件仍为processed_samples=1669、complete=true、is_full_benchmark=true，并保留协议与format_audit。真实新样本的速度看后续稳定图的generate/score耗时，不能用旧163张缓存重评分速度估算。

### 历史回执：v4九图通过及随后全量命令

用户WSL回执：9/9，complete=true、is_full_benchmark=false，36个512裁块全部为23×24合法原生矩形；5个查询各重试一次，共41次生成，均已恢复。合并mIoU=11.1732%、mF1=16.5885%、像素准确率52.9201%；仅9图，不能作为全量论文分数。进度条末尾0.0685是最后一张图，不能代替合并结果。格式链路通过，分割质量仍明显偏背景，原因尚未确定，不能将与旧两图不同的分数解释成精度提升。证据见baselines/rsunivlm/receipts/wsl_loveda_v4_smoke9_20261102.json。

保持原v4路径、相同prompt/FORMAT_RETRIES与代码，MAX_SAMPLES改0继续1669张。已完成9张在身份/掩码哈希通过后复用并重新计入汇总；没有删除缓存或强制覆盖。本轮只记回执与命令，不改推理源码/配置/权重、不启动GPU；无需为本次记录更新而改动WSL科学源码。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 FORMAT_RETRIES=2 MAX_SAMPLES=0 SAVE_IMAGES=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4 \
bash run_bash/rsunivlm_loveda.bash
~~~

完成条件：processed_samples=1669、complete=true、is_full_benchmark=true，并保留format_audit与源协议。仅保存预测掩码及必要RLE/评分回执。若某裁块三次仍非法则停止，回传该raw_text.attempts，不跳过样本、不填背景、不标记全量成功。后续全量已由用户启动并在163张后中断，当前处理见本节顶部；本机没有WSL连接。

### 当前入口：v4去除占位词，有限格式重试；先覆盖原失败的前9张

用户原文已确认Val_Rural_2530_tile512_y512_x0_seg最后一行是others *23, class *1：前575个网格位置为others，最后1个为未知类别。124个token、EOS=true、hit_token_limit=false，不是截断或额外尾文。不能删除第24行、替换未知类或填背景。回执保存在baselines/rsunivlm/receipts/wsl_invalid_class_20261102.json。

当前协议版本3：LoveDA提示不再包含class*count；每请求第一次输出无效时，以两条预定提示词依次重新生成同一RGB输入，最多共3次，首个合法即采用，不读取GT、不按分数挑选。全部原文和生成结束信息保留，metrics.format_audit报告查询/尝试/重试计数；全失败仍停止且不是全量成绩。原模型加载、视觉预处理、expert、greedy、类别映射、原生矩形恢复与指标数学不改。

因为prompt与恢复策略改变，使用新v4输出目录；旧v3的8张/原文保留，不复制或改身份混入v4。继续只保存pred_mask与文本/评分回执，不保存输入/GT/彩色/叠加图。没有在本机或WSL代执行测试/推理，当前仅静态核验。

~~~bash
cd /root/code/vistarmodels
git pull --ff-only
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

GPU_IDS=1 NPROC_PER_NODE=1 TILE_SIZE=512 \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 FORMAT_RETRIES=2 MAX_SAMPLES=9 SAVE_IMAGES=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4 \
bash run_bash/rsunivlm_loveda.bash
~~~

通过条件：9/9、metrics.complete=true、is_full_benchmark=false，确认原失败Val_Rural_2530的掩码与全部裁块合法；raw_text保留每次attempt，format_audit包含36个查询（标准1024原图，每图4块）及实际重试次数。通过后同命令MAX_SAMPLES改0，保留v4输出根复用9张，完成全部1669后才is_full_benchmark=true。再次失败则看v4对应raw_text的attempts，不修改未知标签或跳过样本。

WSL回归（精简文件/假模型，无预训练权重；本机未执行）：

~~~bash
python -m pytest tests/test_rsunivlm_protocols.py tests/test_rsunivlm_format_retries.py -q
~~~

以下为v3失败的历史诊断记录；当前运行以本节v4命令为准。

### 2026-11-02追加：第9张图输出未知class，先读取失败原文

用户全量回执显示8/1669已完成，第9张某个512裁块的第24个RLE行出现Unknown loveda label 'class '，整轮停止。加载后无残留meta参数，不能归因为当前日志中的head warning。前8张预测/回执与已成功裁块保留；不删除输出目录，不将未知class映射背景、不补行，也不跳过样本后声称全量完成。

源码中的提示词含class*count占位写法，可能诱发模仿，但还没有失败response原文，无法区分额外格式示例与真实掩码未知类别。下一步在WSL只读打印最新失败裁块的完整原文与结束信息；不加载权重、不占GPU，不重新推理。原模型/解析器/指标代码未修改，缓存身份保持。

~~~bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v3')
bad = []
for path in (root/'raw_text').glob('*.json'):
    data = json.loads(path.read_text())
    if data.get('decode_error'):
        bad.append((path.stat().st_mtime_ns, path, data))
print('已保存预测掩码数：', sum(1 for _ in (root/'pred_mask').glob('*.png')))
if not bad:
    raise SystemExit('没有找到带decode_error的原文，请核对OUTPUT_DIR')
_, path, data = max(bad, key=lambda item: item[0])
print(json.dumps({'file': str(path), 'error': data.get('decode_error'),
                 'generation': data.get('generation'), 'prompt': data.get('prompt'),
                 'response': data.get('response')}, ensure_ascii=False, indent=2))
PY
~~~

状态：待用户回传原文后确定处理方式；当前未宣称修复、未重跑GPU或得到全量分数。greedy原提示词不变时直接重跑可能再次输出同一非法标签，先不要反复启动全量。

状态（2026-11-02用户WSL附件af748b60）：v3两张原图完整跑通，processed_samples=2、complete=true、is_full_benchmark=false；8个裁块中7个23×24、1个24×24；加载后没有残留meta参数。用户回执汇总mIoU=7.5737%、mF1=11.5266%、像素准确率41.6254%，有效像素预测背景占98.7080%。这只是两样本诊断，未完成全量，低分原因仍待原始输出/输入排查。进度条0.3073为最后一张图分数，不是合并分数。回执与统计见baselines/rsunivlm/receipts/wsl_loveda_smoke2_20261102.json。

按用户最新节省空间要求：运行脚本默认SAVE_IMAGES=0，只保存pred_mask预测掩码；RLE文本、逐图回执、评分日志保留，不再保存输入/GT/彩色/叠加/错误图。旧v3预测身份保持，可复用；不主动删除已有图片。本次没有重新推理或启动全量，代码修改仅存图默认开关。下方已有命令均改为SAVE_IMAGES=0，现有WSL代码也支持此参数。
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
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 SAVE_IMAGES=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4 \
bash run_bash/rsunivlm_loveda.bash --loveda_tile_size 512
# LoveDA通过后，需要LEVIR-CD时再串行执行：
GPU_IDS=1 BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=2 bash run_bash/rsunivlm_levircd.bash
```

LoveDA 输出：`/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4`，与旧整图目录分开。
LEVIR-CD 输出：`/root/data/experiment/rsunivlm_levircd_test_native_rle_symmetric_union_v3`。
通过条件：两份 `metrics.json` 均 `complete=true`、`processed_samples=2`、
`is_full_benchmark=false`；检查 `pred_mask` 和 `raw_text` 是否符合预期。
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
BOOTSTRAP_RSUNIVLM=0 MAX_SAMPLES=0 SAVE_IMAGES=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v4 \
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
