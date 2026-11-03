# 待执行队列

2026-11-01用户规则：以后新训练实验统一20,000个optimizer updates，默认保存10k/20k，具体规则见AGENTS.md。旧训练入口用于新实验前先适配20k并核对实际预算；下方RSUniVLM是公开权重推理，不新增训练。

## RSUniVLM：论文七数据集分割与变化检测

### 2026-11-03 DataLoader离线更新包

用户要求制作离线包。固定已提交的DataLoader实现（代码提交f29b995）并附完整main Git历史、路径保护导入器、SHA256校验和安装说明；不需要旧增量包前置提交，不包含权重/数据/已有预测或其他未提交基线改动。外层工作区产物：output/offline/rsunivlm_dataloader_20261103.zip；实际bundle tip与归档SHA记录在同名receipt。只做归档/Git传输校验，不运行本机pytest、DataLoader worker、模型或CUDA。

先等原队列退出并回到shell，将ZIP复制到WSL的/root/code/vistarmodels。安装器校验包内文件SHA并调用标准库导入器：仅临时恢复已知路径JSON的HEAD字节、ff-only导入bundle，然后无论成功失败均恢复用户JSON并保留临时备份；其他本地修改受Git正常保护，不reset/stash/强行覆盖。该步骤不联网、不自动测试或启动模型：

~~~bash
cd /root/code/vistarmodels
python -m zipfile -e ./rsunivlm_dataloader_20261103.zip ./rsunivlm_dataloader_update_20261103
bash ./rsunivlm_dataloader_update_20261103/install_update.bash /root/code/vistarmodels
~~~

通过条件：SHA校验与Git bundle verify/快进导入成功，并显示DataLoader发布记录；若Git拒绝其他本地修改，停止并保留错误，不覆盖。归档安装不代表推理已通过。

随后沿用下一节的WSL合成测试，但跳过其中联网git pull那一行。测试通过且两卡恢复后，运行该节原OUTPUT_DIR的完整七集命令。包内README也附相同测试与运行命令，默认每卡2个CPU worker、prefetch2、模型batch16、CUDA_LAUNCH_BLOCKING=0；既有有效掩码/原文按原规则续跑，旧格式失败预算不增加。宿主缺卡/实际吞吐/真实恢复仍以WSL回执为准。

### 2026-11-03 当前：DataLoader并行RGB读取，保持原目录与原组批续跑

按用户要求参考相邻Vistar验证加载器，新增每卡2个CPU worker、每worker预取2个原图/时相对；GPU batch仍每卡16条裁块查询。spawn、有序返回、独立CPU generator，无随机变换或GT预读；原15个数值源码不变，原缓存判定/查询批次/分片保持。详见baselines/rsunivlm/DATALOADER.md。用户曾报告两平台缺第二张卡，当前尚无恢复回执；缺卡保护保留，本次不自动启动实验。

更新后在WSL先跑合成回归（无模型权重/无CUDA前向），包含真实spawn进程测试；本机不运行：

~~~bash
cd /root/code/vistarmodels
git pull --ff-only
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m pytest tests/test_rsunivlm_guarded.py tests/test_rsunivlm_data_loading.py -q
~~~

通过条件：合成测试全部通过。失败则停止并保留输出，不改旧结果、不强制跳过协议校验。DataLoader=2的合成spawn检查不是实际模型吞吐或GPU稳定性验证。

两卡重新被系统识别、CUDA恢复且回归通过后，以相同原七集/原目录续跑。CUDA_LAUNCH_BLOCKING显式恢复0以退出此前同步诊断模式，监督和faulthandler继续保留；数值参数不变。NUM_WORKERS/PREFETCH_FACTOR是I/O配置，不改变原模型batch或生成预算。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

CUDA_LAUNCH_BLOCKING=0 PYTHONFAULTHANDLER=1 \
GPU_IDS=0,1 BATCH_SIZE=16 NUM_WORKERS=2 PREFETCH_FACTOR=2 \
MAX_SAMPLES=0 PLAN_ONLY=0 SEMANTIC_MODE=native_classwise \
FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=/root/code/vistarmodels/configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1 \
bash /root/code/vistarmodels/run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

运行时应打印[RGB DataLoader] workers/GPU=2、prefetch/worker=2、ordered=True。原有效掩码/原文复用，旧格式预算耗尽仍跳过；进度重新扫描不表示重新生图。每集/卡shards/rankNNN/data_loading.json记录实际等待时间/解码时间/消费数；guarded_runtime_finished.json记录I/O选项与代码SHA。若出现_p后缀、数据身份异常、CUDA致命错或设备缺失，保留首次日志并停止，不把局部结果当全量评分。

真实提速以WSL时间统计为准，不能从使用DataLoader推定生成算子更快。可用NUM_WORKERS=0隔离I/O差异，保持同一原输出/模型参数；不要同时跑两个写同一目录的队列。独立格式补跑入口维持原策略，本次改动适用于七集主队列。

### 2026-11-03 发布故障保护后更新与续跑

用户要求提交推送并提供运行命令。本轮发布仅含RSUniVLM监督入口、Bash入口、回归测试及相关审查/诊断记录，其他基线的本地修改不纳入。Git发布以实际push回执为准；本机仍未执行测试、模型或CUDA检查。WSL七集尚未恢复完成，宿主缺卡原因尚未确定。

先更新代码并在WSL运行只用合成对象的监督回归（无权重/无CUDA前向）：

~~~bash
cd /root/code/vistarmodels
git pull --ff-only
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m pytest tests/test_rsunivlm_guarded.py -q
~~~

通过条件：pytest全部通过；失败则保留输出并停止，不启动完整队列。依赖沿用原环境，无需为此更新torch/驱动。

Windows和WSL重新识别两张卡、CUDA初始化恢复且上述合成回归通过后，使用原batch16/原两卡/原输出目录续跑。GPU清单缺失/不明确时新入口会在模型worker启动前拒绝；worker段错误或已知CUDA致命错使本次队列停止，不再继续另一个数据集。CUDA_LAUNCH_BLOCKING=1保留用于初始非法地址定位，会牺牲吞吐，不是修复或恢复成功的证据。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 \
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
SEMANTIC_MODE=native_classwise FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=/root/code/vistarmodels/configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1 \
bash /root/code/vistarmodels/run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

同身份已完成掩码/有效原文会复用，进度条重新扫描和评分不等于重新生成；旧零重试格式失败仍按原策略保留，需等中断人口补齐后使用独立格式恢复入口。监督记录为guarded_runtime_finished.json，每轮新日志为logs/guard_<启动ID>_gpu_0.log和gpu_1.log，故障dispatch记录为guard_inflight_rank000/001.json。原15个数值源码未变，若实际OUTPUT_DIR产生协议后缀则先核对身份，不能当旧目录续跑成功。

### 2026-11-03 当前：Windows与WSL都缺第二张卡，先采集宿主日志

用户明确两平台nvidia-smi均只显示一张卡。当前不继续七集/纯CUDA负载诊断，也不把缺卡只当WSL配置错误。首次非法地址与宿主设备异常先后关系未知；保护修改及详细源码证据见baselines/rsunivlm/GPU_FAILURE_AUDIT_20261103.md。

在Windows PowerShell（不是WSL）执行以下只读命令，将输出保存后贴回；它不改驱动、设备、电源设置或TDR注册表：

~~~powershell
Get-Date
Get-PnpDevice -Class Display |
  Select-Object Status, FriendlyName, InstanceId | Format-List

Get-CimInstance Win32_VideoController |
  Select-Object Name, Status, ConfigManagerErrorCode, DriverVersion, PNPDeviceID | Format-List

Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-24)} -ErrorAction SilentlyContinue |
  Where-Object { $_.ProviderName -match 'nvlddmkm|^Display$|WHEA-Logger|Kernel-PnP' } |
  Select-Object -First 30 TimeCreated, Id, ProviderName, Message | Format-List
~~~

依据：PnP能否仍看见第二张卡、错误码、驱动恢复事件与WHEA/PCIe事件可缩小驱动/设备路径；无事件输出也不能排除故障。采集后、其他工作保存且任务停止时，可以正常重启Windows再核对两张卡；不是只重启WSL即可保证恢复。没有收到宿主日志前不宣称供电不足、显卡损坏或特定驱动bug。

本地新增独立guarded controller并让Bash默认走它：所选index/UUID缺失时不启动；致命CUDA错立即退出；任一worker崩溃即停止本队列peer；记录启动边界/故障批次，不在致命退出后合并旧指标。本轮未推送/未安装到WSL，15个数值身份源码不改，原输出身份保留。仅静态检查，以下测试只在两平台设备恢复后由用户WSL执行；现在先不运行。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m pytest tests/test_rsunivlm_guarded.py -q
~~~

该测试使用合成对象，不加载权重、不做CUDA前向；无回执不记通过。真正续跑还需两卡识别和CUDA初始化恢复。保护代码不是已经证明能消除宿主掉卡；不静默降batch、减token或改两卡分片混入旧结果。

### 2026-11-03 当前：启动崩溃位于torch.cuda.is_available

用户WSL faulthandler回执：队列依赖、worker与torch导入均通过；在第3阶段torch.cuda.is_available内SIGSEGV，栈为环境torch/cuda/__init__.py:180、stdin:9，尚未加载权重、输入或执行模型forward。当前未验证已安装torch的具体调用源码；官方实现默认经cudaGetDeviceCount触发驱动初始化。故障定位到设备可用性检测路径，尚不能区分先导入项目依赖造成的原生冲突、CUDA运行库问题或WSL/宿主GPU状态，也不能声称已排除最初推理非法地址与batch的关系。回执见baselines/rsunivlm/receipts/wsl_cuda_is_available_sigsegv_20261103.json。

先在同一WSL环境获取nvidia-smi，并单独仅导入torch检查设备与一个元素的CUDA运算；不导入项目、不加载权重、不写原实验结果。该检查只由用户在WSL运行，本机不执行：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

timeout 10s nvidia-smi

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u -X faulthandler - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
print('Checking CUDA availability...', flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)
print('Testing one CUDA tensor...', flush=True)
x = torch.ones(1, device='cuda')
torch.cuda.synchronize()
print('CUDA test passed:', x.item(), flush=True)
PY
~~~

判断：纯torch仍在设备检查崩溃，说明不需要项目导入即可触发，优先处理运行时/WSL/宿主驱动路径；纯torch通过则保留环境，继续比较导入顺序，不能据此重装或判定设备已稳定。nvidia-smi能显示GPU也不等于CUDA tensor/模型forward通过。若超时退出124记录为超时，不伪称驱动损坏。

仅当纯torch也失败且确认没有其他WSL任务需要保留时，可在Windows PowerShell尝试重启WSL状态。wsl --shutdown会终止所有运行中的WSL发行版/进程；用户先保存工作。它不删除磁盘上的实验输出，但不是已验证修复。此处不自动执行、不重置GPU、不重装依赖、不修改数值源码：

~~~powershell
wsl --shutdown
wsl -d Ubuntu-22.04
~~~

重新进入后激活原环境并重复上面的纯torch检查。只有它成功且原项目启动检查也通过，才继续原七集命令；保持两卡、batch16、原输出身份复用已有结果。若仍失败，保留本次版本/完整崩溃栈与nvidia-smi输出后再诊断，不连续盲重启。本轮未取得上述纯torch/重启回执，未宣称恢复。

### 2026-11-03 当前：两worker退出码均为-11，定位启动段错误

用户附件95d92530回执：runtime_finished.json的owned_pids为15324/15326，exit_codes为[-11,-11]，表示两个子进程被SIGSEGV终止；不是普通Python异常退出。两卡日志尾部仍是此前FLAIR非法地址及后续集合连锁失败，未提供日志时间，不能当作本轮新推理。新一轮没有worker start日志，结合源码只能将范围缩至进入数据集循环前；具体导入库/CUDA初始化位置及首次非法地址的关系尚未知，不能据此判定OOM、batch16或驱动因果。原轮rank0快照5696/7850、格式失败1337；rank1快照6688/7850、成功5069/失败1619。回执见baselines/rsunivlm/receipts/wsl_resume_sigsegv_20261103.json。

先在WSL原环境单独检查GPU0的启动阶段，保持与worker一致的先队列依赖、再worker/torch导入顺序；开启faulthandler以在段错误时打印当时Python栈。此命令只导入模块和初始化CUDA，不读取权重、不推理、不写原实验目录；无需更新代码。不要与旧队列并行运行。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
CUDA_LAUNCH_BLOCKING=1 python -u -X faulthandler - <<'PY'
print('[1/4] import queue dependencies', flush=True)
import tools.eval_rsunivlm_paper_queue
print('[2/4] import worker / torch', flush=True)
from tools.rsunivlm_paper_worker import Engine
import torch
print('[3/4] CUDA availability:', torch.cuda.is_available(), flush=True)
print('[4/4] initialize CUDA device', flush=True)
print(torch.cuda.get_device_name(0), flush=True)
print('Startup checks passed; no weights loaded or inference performed.', flush=True)
PY
~~~

失败条件：任何段错误或异常，保留从最后一个阶段标记开始的完整堆栈，不继续盲跑七集。通过条件：打印Startup checks passed；它仅排除这一次进程中的导入/设备初始化失败，不证明推理恢复或解释原双卡段错误。若通过，再给原七集命令增加PYTHONFAULTHANDLER=1，保留CUDA_LAUNCH_BLOCKING=1、原batch/卡数/输出，定位完整worker实际路径；无需删除缓存、升级依赖或重建环境。本机未执行该检查、模型或测试，当前仅核对源码/命令语法与回执。

### 2026-11-03 当前：同步续跑预检后立即退出，先读worker退出码

用户重新执行原双卡batch16命令并启用CUDA_LAUNCH_BLOCKING=1，七集预检全部通过，实际OUTPUT_DIR保持原目录；随后直接汇总complete=false，无worker start/模型加载/新推理进度。当前只能确认预检与汇总，不能确认新一轮CUDA非法访问或OOM。源码的merge会读取同身份已有分片metrics，因此该complete=false可能包含上一轮不完整记录。最新worker退出码、日志时间与尾部缺失，不能把这一现象直接说成已完成续跑或再次模型错误。回执见baselines/rsunivlm/receipts/wsl_sync_resume_immediate_exit_20261103.json。

先用精简命令读取最新退出码与两卡日志尾部（只读，不启动模型）：

~~~bash
RSUNIVLM_RUN=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1
cat "$RSUNIVLM_RUN/runtime_finished.json"
tail -n 60 "$RSUNIVLM_RUN/logs/gpu_0.log"
tail -n 60 "$RSUNIVLM_RUN/logs/gpu_1.log"
~~~

需要核对日志是否为旧内容时，用下面完整检查，包含记录修改时间与worker回执。只读文件，不加载模型或启动GPU；日志使用deque流式读取尾部，不把完整长日志读进内存：

~~~bash
cd /root/code/vistarmodels
python - <<'PY'
import json
import sys
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path('/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1')
zone = timezone(timedelta(hours=8))
print('Python:', sys.executable)
for name in ('runtime.json', 'runtime_finished.json', 'worker_0.json', 'worker_1.json'):
    path = root / name
    print('\nFILE:', path)
    if not path.is_file():
        print('MISSING')
        continue
    print('Modified:', datetime.fromtimestamp(path.stat().st_mtime, zone).isoformat())
    print(json.dumps(json.loads(path.read_text()), ensure_ascii=False, indent=2))
for gpu in (0, 1):
    path = root / 'logs' / f'gpu_{gpu}.log'
    print('\nLOG:', path)
    if not path.is_file():
        print('MISSING')
        continue
    print('Modified:', datetime.fromtimestamp(path.stat().st_mtime, zone).isoformat())
    with path.open(errors='replace') as stream:
        print(''.join(deque(stream, maxlen=45)))
PY
~~~

本轮没有修改模型/数值源码、删除缓存或启动WSL任务。暂不盲目变更batch、卡数、模型或依赖，先依据退出码区分Python初始化异常、原生进程崩溃与外部终止；若得到被强制终止的证据，再检查系统OOM/驱动记录。

### 2026-11-03 当前：FLAIR遇CUDA非法访问，先新进程续跑未处理部分

用户最新终端回执：GPU1/rank1的FLAIR在6688/7850处报`CUDA error: an illegal memory access was encountered`；此前5069张成功、1619张格式失败，尚余1162张未遍历。随后同一进程中的UAVid/xBD/CHN6/LEVIR/SECOND均0张报相同CUDA错误。最终队列返回shell，complete=false；GPU0状态未在片段中出现。回执：`baselines/rsunivlm/receipts/wsl_flair_cuda_illegal_access_20261103.json`。

该问题与RLE格式失败分开。现有worker捕获所有Exception后继续使用同一个Engine，是导致后续CUDA错误连锁的控制流缺陷；尚未修改该数值源码，以保护已有缓存身份。首次出错的具体算子/驱动原因未知，不能把illegal address当作普通OOM或直接归因batch16。NVIDIA说明非法地址错误后进程必须重启，PyTorch的CUDA_LAUNCH_BLOCKING=1用于同步定位异常，会降低吞吐，不是已确认修复。

先打印已保存完整堆栈（只读，不加载模型；终端没有完整Python栈，需此内容定位首次失败）：

~~~bash
cd /root/code/vistarmodels
python - <<'PY'
import json
from pathlib import Path
root = Path('/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1')
path = root / 'errors/rank1_flair.json'
data = json.loads(path.read_text())
print(path)
print(data.get('traceback') or data.get('error'))
PY
~~~

然后使用原七集/原输出目录/原两卡与batch16重新启动；保留原源码/权重/输入元数据，不先升级依赖。CUDA_LAUNCH_BLOCKING仅增加同步诊断。该命令尚未执行，无恢复效果或新分数回执：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

CUDA_LAUNCH_BLOCKING=1 \
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
SEMANTIC_MODE=native_classwise FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=/root/code/vistarmodels/configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

同协议成功掩码和有效原文会复用并重新评分，进度可能从0重新扫描；这不是全部重新生成。当前格式错误的1619张不会因该命令自动重新尝试。先补齐CUDA中断人口，待所有分片`exhausted_population=true`后，再使用下方格式恢复入口；它会拒绝仍因CUDA/IO中断的分片。若非法访问再次出现，保留首次异常的同步堆栈并停止重复盲跑。通过条件是未处理样本继续前进且不再发生CUDA连锁；全量complete仍需另行补齐格式失败，不能把局部成功子集当论文分数。

### 2026-11-03 当前：失败样本独立补跑，等待原七集队列结束

状态：用户日志显示FLAIR约54%/60%，有格式失败但队列仍继续；原七集完成状态未知。新补跑入口仅静态检查，WSL合成回归和真实恢复均未运行。本次只新增恢复工具/测试/文档，没有修改当前队列的数值源码。先等待整个旧队列退出，再更新；不要并行运行旧队列和恢复入口。原因/恢复范围见`baselines/rsunivlm/FAILED_RECOVERY.md`。

下载`rsunivlm_failed_recovery_20261103.zip`到WSL项目目录后，使用本地离线包安装，不依赖GitHub连接；安装器保留用户手改的数据路径JSON。不包含权重/数据。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m zipfile -e ./rsunivlm_failed_recovery_20261103.zip ./rsunivlm_failed_recovery_update
python ./rsunivlm_failed_recovery_update/install_offline.py --repo . \
  --bundle ./rsunivlm_failed_recovery_update/rsunivlm_failed_recovery_20261103.bundle

python -m pytest tests/test_rsunivlm_failed_recovery.py \
  tests/test_rsunivlm_native_semantic.py tests/test_rsunivlm_paper_queue.py \
  tests/test_rsunivlm_second_undirected.py tests/test_rsunivlm_levircd_forward.py -q
~~~

测试通过才执行；测试失败保留输出并停止补跑，不把未运行记为通过。以下`SOURCE_RUN`填原来终端的Actual OUTPUT_DIR，若有`_p...`请完整保留。预检自动选择所有有格式失败且已处理完整名单的数据集，不启动GPU：

~~~bash
SOURCE_RUN=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1
python tools/retry_rsunivlm_failed.py --source_run "$SOURCE_RUN" --plan_only
~~~

原队列已退出、预检通过且两卡空闲后，沿用源运行的双卡和每卡batch16上限（失败查询不足时实际batch会更小），执行同一补跑：

~~~bash
python tools/retry_rsunivlm_failed.py --source_run "$SOURCE_RUN"
~~~

可选`--datasets flair`仅补FLAIR；默认覆盖原七集中所有符合条件的失败集合。可选`--gpu_ids 0,1`改变卡号但数量必须与源分片数一致；不新增训练。默认输出为源目录加`_failed_retry_r2`，工具打印实际目录。成功整图掩码硬链接/跨盘复制，失败整图中有效原文复用；只对坏/缺失查询执行模型。输入/源码/权重身份改变时停止，不能跨协议硬导入；预算原已为2时不重复无效模板。

产物：`recovery_import.json`、逐卡原文/掩码/失败记录、每集`metrics.json`、`paper_summary.json/csv`。通过条件：源目录保持，已成功推理不重复计算；所有需要的集合完整时`complete=true`、`num_failed=0`，成功数等于该集原名单。若仍有失败，队列会处理完其余样本，最后返回非零；保留partial结果和错误原文，不当作全量论文分数。新的首个有效固定模板策略/来源单独记录，不替换论文分数。

### 2026-11-03 当前：格式失败后继续，不中止数据集

用户32张FLAIR回执确认两卡已加载模型、实际batch16，分别因`building:1`等损坏RLE而在0/16停止。新代码将格式预算耗尽改成记录失败整图并继续剩余样本/数据集。`FORMAT_RETRIES=0`仍是仅一次原始模板；不是无限重试或放宽解码，不用背景补齐。最终成功子集分数注明partial，失败未补齐时complete=false；全队列处理完才返回非零状态。详情见`baselines/rsunivlm/CONTINUE_FORMAT_FAILURES.md`。

将新包`rsunivlm_continue_format_errors_20261103.zip`放到WSL项目目录；待旧队列退出后更新。安装器保留用户手改路径JSON。以下测试/模型命令只在WSL运行，本机没有执行：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m zipfile -e ./rsunivlm_continue_format_errors_20261103.zip ./rsunivlm_continue_format_update
python ./rsunivlm_continue_format_update/install_offline.py --repo . \
  --bundle ./rsunivlm_continue_format_update/rsunivlm_continue_format_errors_20261103.bundle

python -m pytest tests/test_rsunivlm_native_semantic.py \
  tests/test_rsunivlm_paper_queue.py tests/test_rsunivlm_second_undirected.py \
  tests/test_rsunivlm_levircd_forward.py -q
~~~

沿用用户全量命令，无需新增开关：

~~~bash
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
SEMANTIC_MODE=native_classwise FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

已有旧协议目录时会打印新的哈希目录，旧预测和原文保留，不自动跨版本导入；新版相同身份可续跑。每集根目录`failed_samples.json`和`metrics.json.num_failed`列出缺失，`attempted_samples`包含成功与失败，`processed_samples`只含完整预测已评分样本。`exhausted_population=true`表示已处理名单末尾，有失败时仍complete=false、分数仅为成功子集。仅格式错误不再截断其余图；真实数据错误/模型异常仍报错。当前仅静态验证，真实继续执行和吞吐待用户WSL回执。

### 2026-11-03 当前：按原论文逐类别查询

用户要求对齐表7（Vaihingen54.46、UDD5 65.41、VDD49.60）的类别处理。当前默认已从一次枚举全部类别改为论文附录逐类提示，官方演示二值还原，others为查询背景；先按固定类别累计全数据TP/FP/FN，再求类别mIoU。表7具体多类合并/评分脚本未公开，结果明确`metric_scope=independent_class_queries`、`table7_protocol_verified=false`，不据三个汇报值猜测作者隐藏规则。详见`baselines/rsunivlm/NATIVE_SEMANTIC.md`。本次不改变评测集为那三个集合，仍处理我们的五个分割集及两类变化任务。

新包 `rsunivlm_native_classwise_20261103.zip` 含完整代码历史、路径保护安装器，不含模型/数据。当前代码尚无WSL测试或真实推理回执；本机仅静态和Git传输检查。先等待现有队列退出，再更新，不在运行中改源码：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m zipfile -e ./rsunivlm_native_classwise_20261103.zip ./rsunivlm_native_classwise_update
python ./rsunivlm_native_classwise_update/install_offline.py --repo . \
  --bundle ./rsunivlm_native_classwise_update/rsunivlm_native_classwise_20261103.bundle

python -m pytest tests/test_rsunivlm_native_semantic.py \
  tests/test_rsunivlm_paper_queue.py tests/test_rsunivlm_second_undirected.py \
  tests/test_rsunivlm_levircd_forward.py tests/test_rsunivlm_flair_background.py -q
~~~

先用FLAIR32张开发核对，双卡各16张/类；这个32只供检查，不写正式分数。通过后再跑下面独立正式目录。

~~~bash
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=32 PLAN_ONLY=0 \
SEMANTIC_MODE=native_classwise FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
DATASETS=flair DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_flair_tile512_bs16_smoke32_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

通过条件：flair/metrics.json complete=true、processed_samples=32；每图有12张0/1预测掩码，raw_text中的query_class与论文首轮提示一致，无一次列出12类的请求；未读取GT选择查询。默认只允许一次原始模板。若RLE失败，保留原始回答和失败记录，不用背景填充，不继续正式队列。真实吞吐和显存需回执；逐类调用次数会增加。

正式七集队列：

~~~bash
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
SEMANTIC_MODE=native_classwise FORMAT_RETRIES=0 MAX_NEW_TOKENS=32000 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

只保存必要类别预测掩码和原文/评分回执；有效同协议预测/原文可续跑，旧一次多类RLE不能导入。LoveDA1669/FLAIR15700/UAVid270的多类全群体完成看complete=true、is_full_query_population=true；标准互斥多类协议仍未核验，standard_metrics_available=false、is_full_benchmark=false，不是缺图。二值集合xBD933、CHN6 903及现有变化集合按其既有指标/全群体核对。根complete表示所有请求已完成；根“全部标准基准完成”标志仍受多类协议待核对限制，不自动更新论文数值。

本次STEP不改变SECOND用户指定的六类无方向定义，不把原论文WHU二值变化指标冒充SECOND六类指标。LEVIR仍每对一次，不反向OR。下面v2总队列及FLAIR背景ID12小节为历史接口；只有显式`SEMANTIC_MODE=legacy_categorical`时才是那套一次多类输出，不能套入本次原生逐类目录。

### 2026-11-03：FLAIR 的 others 按背景处理

用户明确要求将 `others` 视为背景。FLAIR 新增预测背景 ID12，原十二类 0..11 和 GT 映射保持；有效 GT 上预测背景计入该类漏分，不当 ignore 丢弃。混淆矩阵 12×13，十二类均值不增加背景；边界评分保留背景/前景交界。文档：`baselines/rsunivlm/FLAIR_BACKGROUND.md`。当前只完成本机静态核查，WSL 测试和真实模型效果待回执。

离线包 `rsunivlm_flair_background_20261103.zip` 含完整 main 历史和路径保护导入器，不含模型/数据。先把 ZIP 放到 WSL 项目目录，再执行；不修改正在运行的队列源码，待当前进程退出后导入。其他六集已有输出保留，此命令只运行 FLAIR：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m zipfile -e ./rsunivlm_flair_background_20261103.zip ./rsunivlm_flair_background_update
python ./rsunivlm_flair_background_update/install_offline.py --repo . \
  --bundle ./rsunivlm_flair_background_update/rsunivlm_flair_background_20261103.bundle

python -m pytest tests/test_rsunivlm_flair_background.py \
  tests/test_rsunivlm_paper_queue.py tests/test_rsunivlm_second_undirected.py \
  tests/test_rsunivlm_levircd_forward.py -q
~~~

回归通过后运行 FLAIR 全量，沿用已预检的路径和固定权重；每卡请求16，实际尾批/OOM折半可较小：

~~~bash
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 FORMAT_RETRIES=2 \
DATASETS=flair DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_flair_background_tile512_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

输出为预测 ID 掩码、原始回答、逐图记录及汇总，不保存 RGB/GT/叠加图。旧七集失败目录不覆盖，也不自动跨协议导入原文；新版同目录可续跑。通过条件：`flair/metrics.json` 的 processed_samples=15700、complete=true、is_full_benchmark=true，confusion_matrix 为12×13，末列标为 background；valid_pixels 包括预测背景落在有效 GT 上的像素，并报告 background_pixels_on_valid_gt。只请求一集时根汇总“七集全量”字段为 false 属预期。无效 RLE/其他未知词仍失败，不能把 incomplete 结果写入论文。

### v2总队列：七集依次推理，双GPU分片

用户要求总队列运行命令。前提：已导入v2包，建议先完成下方四对SECOND类别查询与LEVIR核对；当前尚未收到新版真实前向完成回执。以下沿用总队列最多两次固定格式重试，SECOND保留六类无方向掩码及mIoU_c/mF1_c，LEVIR无方向二值建筑变化、每块查询一次。所有模型输入、裁块和评分以新协议记录，不导入旧七类/双向结果；新建输出目录，原结果保留。此次只补充命令，现有v2包无需重下。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
FORMAT_RETRIES=2 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_paper_undirected6_tile512_bs16_2gpu_v2 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

每个GPU工作进程按上述列表依次处理自己的互斥样本分片，首次前向加载模型后复用；两卡之间不要求逐集同步。请求每卡16，实际尾批/格式重试/显存不足时可更小。最终合并各集统计，输出paper_summary.json/csv；完整七集需complete=true、is_full_benchmark=true，任何失败仍保留incomplete，不能跳过后发布为完整结果。

若只排队重跑本次改过的变化检测两集，使用相同入口，将DATASETS设为`levircd,second`，同时OUTPUT_DIR改为`/root/data/experiment/rsunivlm_cd_undirected6_tile512_bs16_2gpu_v2`。这是两集子队列，不称作完整七集；各单集仍分别核对128和593的完整群体。

### 当前优先：SECOND保留六类的无方向语义变化

用户明确“不是二值变化，类别也要，只是无方向”。当前SECOND已改为water/bare land/low vegetation/tree/building/playground六类分别查询，每类每块只输入一次(T1,T2)，不交换时相、不对预测做OR；保留六类mIoU_c/mF1_c。类别可以重叠，例如树木变建筑，两类变化都计入。GT仍是两份原始0..6语义标注，不做整体二值化。详见`baselines/rsunivlm/SECOND_UNDIRECTED.md`。状态：本机静态核对，WSL测试/真实前向尚待执行。

重新打包的完整离线包`output/offline/rsunivlm_second_undirected6_20261102_v2.zip`含完整Git bundle、标准库导入脚本及校验文件，不含权重/数据；此次未改模型、推理或评分，只更新交付文件名和导入说明。用户手动修改过路径JSON，直接git pull会因该本地改动停止；使用随包脚本仅临时恢复该配置的HEAD内容、做ff-only导入，并在成功或失败时原样恢复用户JSON，同时保留临时备份。其余本地修改仍受Git保护，不stash/reset/stage或强制覆盖。先导入，再在WSL回归和检查4对：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python -m zipfile -e ./rsunivlm_second_undirected6_20261102_v2.zip ./rsunivlm_second_update_v2
python ./rsunivlm_second_update_v2/install_offline.py --repo . \
  --bundle ./rsunivlm_second_update_v2/rsunivlm_second_undirected6_20261102_v2.bundle

python -m pytest tests/test_rsunivlm_second_undirected.py \
  tests/test_rsunivlm_levircd_forward.py tests/test_rsunivlm_paper_queue.py -q

GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=4 PLAN_ONLY=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_second_undirected6_tile512_bs16_smoke4_v1 \
bash run_bash/rsunivlm_second_undirected_queue.bash --tile_size 512
~~~

核对入口默认FORMAT_RETRIES=0，只检查类别原始提示；所有类别固定查询，不用GT判断哪些类需要跑。每类独立跨图填批，请求16而实际尾批可较小；OOM自动折半。通过条件：所选4对complete=true，每对六张类别掩码、raw_text有query_class与temporal_mode=undirected、无反向预测，逐类指标仍存在。测试与模型前向本机均未执行。

4对真实核对通过后再用新正式目录跑593对，避免混入开发尾批预测：

~~~bash
cd /root/code/vistarmodels
conda activate vistar_rsunivlm
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_second_undirected6_tile512_bs16_full_v1 \
bash run_bash/rsunivlm_second_undirected_queue.bash --tile_size 512
~~~

正式单集完成看second/metrics.json：processed_samples=593、complete=true、is_full_benchmark=true，scores保留六类miou_c/mf1_c。根汇总只请求SECOND，其“完整七集”字段为false属预期。此次不自动导入旧七类/双向预测，也不重跑已存其他集合；新方案需独立模型有效性回执后才能入论文。

### 当前优先：LEVIR无方向二值建筑变化，每对查询一次

用户要求先只修改LEVIR。新入口保留512源图裁块、双GPU和每卡batch16；只查询(A,B)，不做reverse/OR，首轮使用官方CD_PROMPT。核对入口默认FORMAT_RETRIES=0，仅使用原始提示词，不把格式重试当作官方流程。SECOND不在本次入口中。详见`baselines/rsunivlm/LEVIR_FORWARD.md`。本机只静态检查，不跑模型/测试；WSL实际核对尚待执行。

用户进一步明确任务不要方向性：LEVIR原生GT和现有评分已是无方向二值建筑变化，新增/拆除统一为前景，未按新增/消失分别预测或评分。代码和目录里的forward仅标识固定(A,B)输入一次，现统一称“每对查询一次”，不声称输入交换的模型输出必然一致。此次仅澄清术语，不改提示词/数值源码/缓存身份，无需重下fbe00fd离线包；下方命令保持可用。

离线包`output/offline/rsunivlm_levircd_forward_20261102.bundle`包含完整main历史，不含权重/数据；复制到WSL项目目录后更新：

~~~bash
cd /root/code/vistarmodels
git bundle verify ./rsunivlm_levircd_forward_20261102.bundle
git pull --ff-only ./rsunivlm_levircd_forward_20261102.bundle main
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

# WSL合成回归，不加载预训练权重；本机未执行
python -m pytest tests/test_rsunivlm_levircd_forward.py -q

# 四对真实开发核对，先验证原生单向接口
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=4 PLAN_ONLY=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_levircd_forward_tile512_bs16_smoke4_v1 \
bash run_bash/rsunivlm_levircd_forward_queue.bash --tile_size 512
~~~

通过条件：所选4对complete=true；每个raw_text.direction为forward，每条序列含两个图像；pred_mask仅有forward二值掩码，不能有reverse/union。实际尾批可小于16。GT在预测完成并保存后才评分；无效RLE或无EOS截断仍应失败，不给背景填充。状态：本机静态核对，WSL测试/前向/速度/显存未取得回执。

4对核对通过后再跑完整128对；不同目录避免将开发尾批结果混入正式批量运行：

~~~bash
cd /root/code/vistarmodels
conda activate vistar_rsunivlm
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
OUTPUT_DIR=/root/data/experiment/rsunivlm_levircd_forward_tile512_bs16_full_v1 \
bash run_bash/rsunivlm_levircd_forward_queue.bash --tile_size 512
~~~

正式完成看单集`levircd/metrics.json`：processed_samples=128、complete=true、is_full_benchmark=true。根paper_summary只请求LEVIR，因此其“七集完整”的is_full_benchmark=false属预期。旧六集和SECOND输出保留，本轮不导入或重算；原论文指标未改。相同新协议可恢复，不把旧双向OR掩码当作单向预测。

### 当前交付：论文七集队列，任意正整数batch，双GPU分片

**最新纠正：旧七集试跑最终complete=False，GPU0 SECOND因未知类别词和格式预算耗尽而未完成。用户已指出SECOND不应套“未变化＋六类”一次输出，也不应把双向OR当作模型原生流程。下面保留的旧七集正式命令是历史适配版本，当前不要照抄重跑；数据预检通过仍有效，但不能代替任务接口验证。** 先保留全部原文、预测和汇总；其他六集及GPU1完成状态尚待完整paper_summary.json。原生接口核对与下一步见`baselines/rsunivlm/NATIVE_CHANGE_AUDIT.md`，失败回执见`baselines/rsunivlm/receipts/wsl_paper_second_format_failure_20261102.json`。本轮未发布替代数值实现、未改旧代码/缓存身份、未启动新推理。

2026-11-02最新状态：用户WSL七集共同预检已通过，顺序/数量为LoveDA1669、FLAIR15700、UAVid270、xBD-pre933、CHN6-CUG903、LEVIR-CD128、SECOND593。有效新路径为`/root/data/flair#1-test`、`/root/data/uavid`、`/root/data/xview2`；其余根目录保持配置JSON记录。脚本保留旧单集预检目录，七集完整计划正常写到`/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1_p9c14c4b854db`。最终明确未加载模型、未启动GPU进程；本回执不是推理/吞吐/显存或成绩验证。见`baselines/rsunivlm/receipts/wsl_paper_all7_preflight_20261102.json`。

正式启动命令（已预检；仍需用户在WSL执行，本机不代启动）。不必为本轮文档回执更新再次下载代码包；保持输入、源码及设置不变时，原请求输出根会解析到上面的p9c14c4b854db目录：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 PLAN_ONLY=0 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

实际每卡batch以日志actual_batch为准，尾批/重试可能小于16，CUDA OOM按既有策略折半。每卡加载一次模型并复用七集；只保存预测掩码及必要原文/评分回执。各集评分后总览paper_summary.json/csv；七集全量且complete=true、is_full_benchmark=true及无错误方为正式完成。没有上述回执时不写作运行中/已完成。

以下保留已通过的xBD根目录同步及七集预检命令供重查；无需重新下载代码包，不修改已通过的FLAIR/UAVid配置：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python - <<'PY'
import json
from pathlib import Path
p = Path('configs/rsunivlm/paper_data_roots.json')
d = json.loads(p.read_text())
d['xbd_pre'] = '/root/data/xview2'
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
PY

GPU_IDS=0,1 BATCH_SIZE=16 PLAN_ONLY=1 \
DATASETS=loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

通过条件：七集分别发现1669、15700、270、933、903、128、593个样本且最后Preflight passed。不加载模型或启动GPU作业。xBD评估器支持根目录直接images/labels等既有布局，或根目录下test/images和test/labels等布局；若两者都没有或数量不符，应保留失败并核对目录，不跳图或改群体。

下面保留已通过的FLAIR路径同步及单集预检命令供重查；仅覆盖FLAIR键，保留其余配置：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
python - <<'PY'
import json
from pathlib import Path
p = Path('configs/rsunivlm/paper_data_roots.json')
d = json.loads(p.read_text())
d['flair'] = '/root/data/flair#1-test'
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
PY

GPU_IDS=0,1 BATCH_SIZE=16 DATASETS=flair PLAN_ONLY=1 \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

通过条件：发现FLAIR15,700对且预检通过；不加载模型、不启动GPU作业。该预检可能读取标签作编码审计，不能当作实际模型前向。下面保留已通过的UAVid命令供重复检查，不使用非标准群体开关：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

# 兼容尚未拉到新JSON的WSL；仅覆盖UAVid路径，保留其他字段。
python - <<'PY'
import json
from pathlib import Path
p = Path('configs/rsunivlm/paper_data_roots.json')
d = json.loads(p.read_text())
d['uavid'] = '/root/data/uavid'
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
PY

GPU_IDS=0,1 BATCH_SIZE=16 DATASETS=uavid PLAN_ONLY=1 \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
bash run_bash/rsunivlm_paper_queue.bash --tile_size 512
~~~

通过条件：输出UAVid270个配对且预检成功；本命令不启动模型或GPU作业。评估器支持`/root/data/uavid/{Images,Labels}`，以及其中再嵌套一层`uavid/{Images,Labels}`等既有布局；目录存在但内容不匹配仍应失败，不用猜测式重命名或复制数据。以下七集正式命令须等全部路径预检通过后执行。

512分块已与Alters源码核对，见`baselines/rsunivlm/ALTERS_TILING.md`；本次没有修改运行代码或协议，不需要为这份核对记录重跑已有预测。新增合成回归仅在WSL运行，不加载权重：

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
ALTERS_SOURCE_ROOT=/root/code/vistar \
python -m pytest tests/test_rsunivlm_alters_tiling.py -q
~~~

状态：用例已编写，本机仅AST/源码哈希检查，尚未执行。预期10项通过；缺失Alters参考源码或哈希不符应停止并核对，不跳过。正式推理命令及缓存恢复方式保持如下。

范围/数据路径/精确口径见`baselines/rsunivlm/PAPER_QUEUE.md`：LoveDA Val1669、FLAIR test15700、UAVid all270、xBD-pre test933、CHN6-CUG val903、LEVIR-CD test128、SECOND DynamicEarth val593。后两者是变化检测，不使用SECOND生图1694 test；不新增训练/生图任务。每卡加载一次模型复用七集，对每个集合分配互斥样本，最后合并统计而非平均分数。

用户WSL执行前先Ctrl+C结束旧推理并等回到提示符，确认0/1两卡供本轮使用；本机没有该WSL的连接，不代启动。新离线包为外层工作区`output/offline/rsunivlm_paper_queue_20261102.bundle`，复制到`/root/code/vistarmodels`后：

~~~bash
cd /root/code/vistarmodels
git bundle verify ./rsunivlm_paper_queue_20261102.bundle
git pull --ff-only ./rsunivlm_paper_queue_20261102.bundle main
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

# TIFF读写依赖；不改CUDA版torch或其他模型环境
python -m pip install 'rasterio>=1.3,<1.5' 'tifffile>=2023.7,<2025'

# 路径/群体预检，无模型加载或GPU作业
GPU_IDS=0,1 BATCH_SIZE=16 PLAN_ONLY=1 \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
bash run_bash/rsunivlm_paper_queue.bash

# 可选WSL合成回归；本机没有运行这些测试
python -m pytest tests/test_rsunivlm_paper_queue.py -q

# 正式完整七集；每张卡最多16个tile序列，改变16为任意正整数即可
GPU_IDS=0,1 BATCH_SIZE=16 MAX_SAMPLES=0 \
DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json \
OUTPUT_DIR=/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1 \
bash run_bash/rsunivlm_paper_queue.bash
~~~

若实际数据路径不同，编辑`configs/rsunivlm/paper_data_roots.json`并在预检和正式命令中都设置`DATA_ROOTS_JSON=configs/rsunivlm/paper_data_roots.json`。可用`DATASETS=loveda,levircd`选择部分队列，但该运行不等于论文七集全部完成。任一所选集合缺失时预检一次列出并停止，不下载数据、不用子集替代。

`BATCH_SIZE`为每GPU序列数（16×2卡最多32），每个序列是单块RGB或变化任务的一对RGB；原生512裁块跨图填批。格式重试/尾批可以更小，OOM自动折半。直接批量模式不做单图等价核验，已知差异不隐藏；从独立新目录启动，不混用旧单图缓存。实际大小看`actual_batch=`和raw generation；速度/32GB显存适用性待WSL测量。

产物包括各集`metrics.json`、`shards/rank*/pred_mask`、RLE/回执/进度/错误、GPU日志、环境和PID/退出码记录，以及总`paper_summary.json/csv`。只有预测图片，不保存输入/GT/叠加图。格式预算耗尽的数据集保持incomplete并继续剩余队列；只有全部所选完整人口完成才complete，完整官方七集另需is_full_benchmark=true。状态：实现及本机静态检查，尚无WSL合成测试/前向/完整指标回执。训练暴露差异见TRAINING_DATA_AUDIT.md，不能标成所有数据集零样本。

### 用户当前要求：增大实际batch，独立batch4完整推理

用户明确要求增大实际batch。当前ec6929f已有`TILE_BATCH_SIZE=4 VERIFY_BATCH=0`入口，无需修改或更新推理代码：关闭单图一致性比较，从而不触发已知的比较失败→永久batch1。保留原单图作业输出；先在其WSL终端按一次Ctrl+C并等回到提示符，再运行下列独立新目录命令。不要同时在同一GPU上启动两份。没有本机WSL连接，不代中断或启动。

已知batch3和单图存在1/3差异，原因未定。新运行作为明确的批量协议单独记录，不宣称与单图逐像素等价。此次`IMPORT_V4_ROOT=`不导入单图原文，完整1669张从头按同一批量路径推理，避免把已知非等价的单图缓存混成“全量batch4”。原有输入、权重、512源图裁块、384视觉处理、fp16/eager、greedy、8192新token、至多两次格式重试与全部评分函数不变。

~~~bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

GPU_IDS=1 NPROC_PER_NODE=1 \
TILE_SIZE=512 TILE_BATCH_SIZE=4 VERIFY_BATCH=0 \
DTYPE=float16 ATTENTION=eager \
DATA_ROOT=/root/data/LoveDA SPLIT=val \
RSUNIVLM_ROOT=/root/code/vistarmodels/third_party/RSUniVLM_clean_66dce1d \
RSUNIVLM_MODEL_PATH=/root/data/weight/llava-qwen-gmoe \
RSUNIVLM_VISION_TOWER=/root/data/weight/siglip-so400m-patch14-384 \
BOOTSTRAP_RSUNIVLM=0 FORMAT_RETRIES=2 \
MAX_SAMPLES=0 SAVE_IMAGES=0 IMPORT_V4_ROOT= \
OUTPUT_DIR=/root/data/experiment/rsunivlm_loveda_val_native_tile512_rle_v5_bs4_direct \
bash run_bash/rsunivlm_loveda.bash
~~~

一张1024原图有4个512裁块，首次推理可4条独立序列合批。格式重试仅对失败子集，实际batch可能1–3；CUDA OOM仍保留拆批到2/1，不伪造batch或跳图。只保存预测掩码及必要文本/评分回执，run_config.protocol.batch_verify=false，raw_text的generation.inference_batch_size记录实际批量；最终metrics记录requested/effective batch，不能仅凭目录名断言实际值。产物路径以终端输出及run_config为准。完成条件仍为1669、complete=true、is_full_benchmark=true。

状态：命令已按当前源码核对，未在本机执行测试/模型，也未代在WSL启动；新的实际batch、速度、显存和最终指标待用户回执。单图回退路径仍保留给VERIFY_BATCH=1，现有离线代码包已经支持本命令，无需重新下载包。

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
