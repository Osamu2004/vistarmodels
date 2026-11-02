# RSUniVLM 接入记录

## 2026-11-02：v4九图WSL成功回执，准备固定协议全量

用户回传v4结果：9/9 complete=true但is_full_benchmark=false，identity=767f4ad67e7e48c50ec58f8797038238bef83bb13e6104b792557eaf47f88bc8。36个原图裁块均23×24，format_queries=36、format_attempts=41、format_retry_queries/attempts均5；出现water缺少计数及remote-sensing未知标签的查询经固定格式重试恢复，未用GT补造掩码。原第9张已包含在完成集合，证明本次有限恢复链路在该小集合可运行，不等于全量保证或CPU回归已执行。

合并mIoU=11.1732%、mF1=16.5885%、像素准确率52.9201%；混淆矩阵显示有效像素的背景预测占97.9723%，不把旧单裁块统计套用到本集合。本次完整混淆矩阵与指标归档于receipts/wsl_loveda_v4_smoke9_20261102.json。进度条0.0685是最后单图指标。预测偏背景，尚无充分依据认定模型能力或修复精度改善；不同prompt/样本数不直接比较旧两图分数。

按既定任务继续提供MAX_SAMPLES=0、相同v4输出根、512源图裁块、FORMAT_RETRIES=2、SAVE_IMAGES=0的全量命令，原9张匹配回执可复用。只增加运行记录/JSON证据，不改变科学源码或代启动WSL。完整1669结果、格式重试比例和质量验证仍待用户回执，不写正式论文成绩。

## 2026-11-02：原文确认掩码内未知词，新增v4提示与有限格式恢复

用户回传Val_Rural_2530_tile512_y512_x0_seg原文：23行others *24，最后一行others *23, class *1；共575个已知背景格和1个未知格。sequence_tokens=124、ended_with_eos=true、hit_token_limit=false，排除了token预算截断和“多余格式示例尾行”的解释。未知词与提示词class*count占位写法一致，模仿占位符是待新运行验证的解释；不将该未知格解释成真实背景或按GT补类。

移除LoveDA提示中的占位模式，保留原任务与七类名，不添加示例掩码；默认至多两次预定格式纠正提示，重用同一RGB输入、保持greedy与expert，不给GT或错误标签的拟合结果。只选首个格式合法输出，全部尝试的prompt/原文/结束信息/错误入raw_text.attempts；中断继续未完成预算，耗尽后仍停止。未知类映射、行/计数/截断验证、nearest恢复、原数据与评分函数保持严格；新增format_audit在图片恢复后仍可汇总实际查询和重试次数。后端预算与记录协议需一致。

科学协议版本3，新LoveDA输出v4，旧v3八张及失败原文保留、不混入新prompt结果。默认只保存预测掩码与必要文本/评分；未启动训练/GPU。新增WSL回归覆盖本次真实坏格式、首个合法/同输入/不选分数、首次成功不重试、预算耗尽、中断恢复、截断拒绝、原文保留、统计恢复及身份变化拒绝。按用户要求本机不跑测试或模型，只AST/bash及原加载/解析/指标等数值函数保护核验；实际格式恢复率/速度/精度待WSL前9张和全量回执。命令统一RUN_QUEUE当前v4节。

## 2026-11-02：全量第9张出现未知class，等待原文定位

用户WSL回执：加载后no remaining meta parameters，LoveDA全量8/1669后在第9张某个512裁块第24行因Unknown loveda label 'class '退出；约15.35s/image来自该回执，非全量时间保证。前8张已有掩码/身份回执和成功裁块原文应按现有恢复逻辑保留，当前不能将其登记成完整成绩。

检查现有query_grid确实在解析失败前保存完整response/generation和decode_error。提示词包含class*count，但当前用户只贴了异常，无法确认class是格式示例尾文还是掩码本身非法标签；不根据猜测删除第24行、替换背景或放宽类别映射。向用户请求原文并在唯一RUN_QUEUE提供标准库只读提取命令，不占GPU、不修改输出。模型、prompt、解析器、评分和科学缓存身份均未改；待原文后选择具体处理，不将文档记录称作修复。

## 2026-11-02：两样本推理完成，按用户要求只保留预测掩码

用户附件af748b60回执确认v3的两张LoveDA原图完成，complete=true但is_full_benchmark=false；8个512源图裁块描述符为23×24共7个、24×24共1个。加载后检查无残留meta参数，38秒两张是用户日志的本次观测，不代表全量速度或权重逐张量校验已完成。合并mIoU=7.5737%、mF1=11.5266%、像素准确率41.6254%；由混淆矩阵列和计算背景预测2,070,057/2,097,152=98.7080%，仅building/water另有预测。进度条0.3073对应单张指标，不取代合并分数。存档标记user_pasted_wsl_log，不登记成全量或官方基准结果，低分原因尚未确认。

随后用户明确要求减少存图、只保存预测掩码。将共享RSUniVLM bash的SAVE_IMAGES默认改0，LoveDA/LEVIR-CD均默认传--no-save_images，队列显式命令也改0；预测掩码原本独立于此开关保存，因此保留原分割结果和逐图恢复。保留评分JSON/RLE文本，不保存额外输入、GT、彩色、overlay或error图。仅改bash默认与说明，不修改模型、解析器、指标源码，保留已有v3缓存身份，也不自动删除WSL已有图片。直接Python入口沿用兼容默认，命令需显式--no-save_images。本机只bash语法与保护源码哈希核对，无测试、模型或新增GPU任务。

## 2026-11-01：对齐官方实际矩形RLE解码，补齐生成与加载审计

用户附件498838a7的WSL回执已出现Model Class并返回文本，但首张影像内某个裁块被旧decode_rle强制24行拒绝（收到23行），整图未完成。原始文本未随附件提供，不能断言该23行的宽度一定一致。核对固定官方提交app.py::parse_output_string，官方按返回的实际矩形行列构建掩码再nearest还原，并未硬性检查24行；先前本适配的该约束比官方更严。

评分查询改用native_rectangular解码：正计数、已知类别、一致行宽、每边1…64；合法23×24等直接按官方OpenCV INTER_NEAREST像素对齐还原裁块，不补造第24行、不填背景。保留decode_rle(grid_size=24)严格工具模式。记录每份原文的实际尺寸/生成token数/EOS/预算，达到预算且无EOS的文本仍拒绝；逐图、恢复receipt与汇总保留descriptor_shape_counts。双向CD在各自恢复原图后取并集，在同尺寸网格情形，先还原再并集与先并集再按同一最近邻还原等价；v3的OpenCV像素对齐与旧v2的PIL版本独立记录，不将旧预测混入新协议。

补显式attention_mask及greedy中性采样参数。固定官方SigLIP源码会将pooling head替换Identity，本机仅只读主checkpoint safetensors头确认包含421个视觉张量且无head张量；不据此声称WSL权重全部已物化。新增完整加载后剩余meta参数检查，实际结果待WSL回执。协议版本2，默认LoveDA新目录native_tile512_rle_v3；保留v2失败原文，旧缓存不混用。新增WSL回归，本机只AST/Bash/差异核查，未运行测试、模型或GPU，无新评分。

## 2026-11-01：按用户要求取消已跟踪文件修改检查

最新WSL回执确认512 CLI已更新，但新副本仍报已跟踪文件有修改。核对官方固定提交的Git树发现其中实际跟踪30个cpython-310.pyc文件；Python3.10导入可能改写这些已跟踪缓存，先前新建副本无法持久避免此类误报。具体WSL差异列表尚未收到，不能将所有实际修改都断言为pyc。

用户随后明确要求“取消已跟踪文件有修改的检查”。据此完整移除protocols.source_revision中的git status与拒绝逻辑，以及bootstrap的重复拦截；Git提交必须为固定66dce1d的版本检查保留，不再要求工作区干净或继续克隆。模型权重加载、推理参数、512裁块、提示词与评分代码不改。新增WSL回归覆盖已跟踪源码/字节码修改可通过与错误提交仍拒绝；按用户要求本机仅AST/Bash/保护源码差异核查，不运行测试或模型，真实WSL推理结果仍待回执。

## 2026-11-01：干净官方源码检查通过，WSL适配器仍缺512更新

用户日志确认本地克隆RSUniVLM_clean_66dce1d成功并切到66dce1d；重新运行已通过官方源码/GMoE导入、依赖、主模型元数据与CUDA检查。随后evaluate.py的argparse报不认识--loveda_tile_size 512，帮助列表也缺该参数，因此当前WSL适配器缺本地e25159f的512裁块更新；具体WSL HEAD未知，不将第三方源码checkout误当成外层适配器已更新。模型尚未加载和前向，无任何新分数。

提供基于e5c58f8的离线bundle，包含已静态核对的e25159f实现和当前运行记录；先verify/快进导入，再用--help确认512参数，之后沿用新源码根目录与本地双权重路径跑两原图。保留512参数作为旧代码守卫，不让删除参数后整图运行被误报为512对齐。GitHub此前推送连接失败/超时，交付以本地离线包验证为准；未在本机执行测试/模型或远端作业。

## 2026-11-01：WSL权重路径齐备，官方源码被修改而退出

用户ls回执确认SigLIP配置576字节与model.safetensors约3.3GiB直接位于指定WSL目录，主模型元数据/所列依赖/GPU1 CUDA检查通过；本次源码检查报`Official RSUniVLM tracked source has local modifications`，未执行模型加载或推理。检查逻辑先验证固定commit，再检查已跟踪文件状态，不能把该错误归因于未跟踪pycache；未收到具体diff，因此不臆测变更原因、不删除或覆盖原目录。

提供无需联网的本地git clone --no-hardlinks --no-checkout方案，从已有Git对象创建RSUniVLM_clean_66dce1d副本并切到固定官方提交，RSUNIVLM_ROOT显式指向副本后再跑两原图/512裁块；主模型与视觉塔路径及指标协议不变。保留原源码修改与完整性检查。未修改源码校验规则，未在本机或远端执行模型/测试/克隆；新副本和推理成功仍待用户WSL回执。

## 2026-11-01：用户报告SigLIP已解压至WSL，继续LoveDA冒烟

用户报告`\\wsl.localhost\Ubuntu-22.04\root\data\weight\siglip-so400m-patch14-384`已解压，对应Linux路径`/root/data/weight/siglip-so400m-patch14-384`。将当前状态从WSL缺视觉塔更新为用户报告已解压、待核对目录层级与真实模型加载；不把用户解压陈述当作WSL哈希验证或推理通过。继续提供GPU1、512源图裁块、Val两原图、保存图片、BOOTSTRAP_RSUNIVLM=0命令，主模型路径仍为`/root/data/weight/llava-qwen-gmoe`。本机未连接WSL、未启动推理，未改数值配置。

## 2026-11-01：本机SigLIP下载与校验完成

按用户明确要求使用HF-Mirror下载至`/Users/zhouziyu/Downloads/weight/siglip-so400m-patch14-384`。SDK镜像元数据不兼容后改用镜像resolve URL的HTTP流式下载；固定官方/镜像一致的9fdffc58快照，9个官方文件共3,515,156,215字节完成，模型权重3,511,950,624字节。全部官方文件大小/哈希核对通过，JSON与safetensors偏移检查通过；权重SHA256为ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a。目标目录download_receipt.json已complete=true。进程正常退出，未加载模型、未运行测试或GPU推理；WSL仍需用户复制该完整目录，不能将本机下载成功当作WSL推理通过。

## 2026-11-01：WSL依赖与主模型元数据通过，缺SigLIP

用户回执列出主模型层级中的config.json、tokenizer.json、model.safetensors，并报告Torch2.11.0+cu130、Torchvision0.26.0+cu130、Transformers4.45.2等依赖导入通过；上游固定提交66dce1d及GMoE导入通过，GPU1 RTX5090D可见，主checkpoint元数据通过。脚本因`/root/data/weight/siglip-so400m-patch14-384/config.json`不存在而退出；未到模型加载与前向，不记推理成功或精度结果。官方源码216行的SyntaxWarning并非本次退出原因。先前gdown SSL错误不影响已解压主模型的路径选择，当前仅提供官方SigLIP下载命令，保留BOOTSTRAP_RSUNIVLM=0。记录属于用户WSL回执，本机未连接该机器、未运行模型或测试。

## 2026-11-01：用户报告WSL解压路径

用户将主模型解压至`\\wsl.localhost\Ubuntu-22.04\root\data\weight\llava-qwen-gmoe`，对应Linux目录`/root/data/weight/llava-qwen-gmoe`。更新RUN_QUEUE的所有当前主权重命令与README路径说明，显式通过RSUNIVLM_MODEL_PATH覆盖脚本旧默认值。主模型无需因目录差异重新下载；文件是否直接位于该层、SigLIP完整性和真实模型加载仍待WSL检查。未连接用户WSL、未启动推理、未修改模型数值配置。

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
