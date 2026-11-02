# RSUniVLM 接入记录

## 2026-11-02：UAVid预检通过，用户提供FLAIR新路径

用户贴出WSL回执：`[preflight] uavid: 270 samples; OVSISBench/all 270 paired assets`，随后`Preflight passed; no model loaded and no GPU process started.`，输出目录`/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1`，请求GPU0/1、每卡batch16。仅证明路径/配对/标签预检通过，不是推理或显存/精度验证；归档小回执，不复制数据或预测。

随后用户提供`\\wsl.localhost\Ubuntu-22.04\root\data\flair#1-test`，对应`/root/data/flair#1-test`。只更新配置JSON的FLAIR字段，保留UAVid新路径与其余字段；RUN_QUEUE加入不依赖重新下载包的单键更新和FLAIR单集预检命令，PAPER_QUEUE同步当前状态。FLAIR15,700对待WSL核验，xBD-pre实际路径仍未解决；本机未启动模型/测试，未改数值源码或论文分数。

## 2026-11-02：用户提供UAVid新WSL路径

用户七集预检回执：LoveDA1669、CHN6-CUG903、LEVIR-CD128、SECOND593发现成功；FLAIR目录缺失、UAVid旧目录布局不匹配、xBD-pre布局不匹配，build_plan在模型加载前停止。随后用户提供`\\wsl.localhost\Ubuntu-22.04\root\data\uavid`，对应`/root/data/uavid`。仅将配置JSON的UAVid字段及当前命令/说明更新；Python数值源码、旧默认值、模型、512裁块、标签规则和270群体要求不变。运行必须显式传DATA_ROOTS_JSON；命令在RUN_QUEUE，附只覆盖UAVid键的WSL兼容更新方式。

本机无法连接该WSL，不将路径声明写成目录/数量/内容核验通过。新UAVid单集预检待用户执行；FLAIR和xBD-pre路径仍未解决，未启动新推理或测试。

## 2026-11-02：按Alters实际源码核对512源图分块

检查LoveDA/UAVid/CHN6/SECOND四套裁块函数、xBD对CHN6的导入、LEVIR对SECOND的复用，以及FLAIR原生512检查。四套ceil/pad/coords的AST一致；RGB为[0,1]补0，等同队列PIL补黑；左上原点、512步长、右下边缘补齐、原生拼接裁回都一致。LoveDA已填结果沿用论文实验来源核对文档，不把旧256脚本默认值当主表运行配置。未发现需要更改队列裁块数值路径的差异；保留384 SigLIP/RLE网格与Alters内部预测分辨率的区别。

新增ALTERS_TILING.md、源文件/函数哈希及行号审计、供WSL执行的10项实际参考函数合成回归，命令写入唯一RUN_QUEUE。本机只AST解析与哈希比较，不执行测试或模型；没有WSL连接、没有新成绩。运行源码及其协议身份保持不变，既有同协议缓存不受此次文档/用例更新影响。

## 2026-11-02：实现论文七集队列、任意batch和双GPU分片

按当前CVPR实验节与补充核对七集：LoveDA Val1669、FLAIR test15700、UAVid all270、xBD-pre test933、CHN6 val903、LEVIR-CD test128、SECOND val593。新增paper_protocol/worker/queue与bash入口、路径JSON和593名单（原Vistar名单字节复制，hash见PAPER_QUEUE）；没有把SECOND生图1694或VDD替换进来。分割沿用已有标签/GT/指标数学；SECOND六类方向并集及全局计数按既有DynamicEarth协议实现，RGB标签仅精确已知颜色，不做GT最近色猜测。

batch接受任意正整数，单卡流式跨图积累tile序列；变化任务每序列保留两张配准图。两卡独立副本，各处理模world_size的互斥样本，每卡首次需前向时加载并跨七集复用。父进程校验覆盖/重复后合并全局混淆矩阵、SECOND各类TP/TN/FP/FN和WFm逐图统计，不平均mIoU。OOM折半，尾批/重试可更小；直接批量不做单图等价声明，不导入旧单图结果。

完整预测落盘后才读取GT评分；有效PNG形状/ID/哈希恢复并重评分，损坏/半写入回退同协议原文。只存预测掩码与必要JSON，LEVIR存并集一张，SECOND存双方向两张。某集失败继续其他集但保留incomplete，正式全量完成需七集全部严格覆盖。预检不启动GPU，原始模型、GMoE expert2、384视觉处理、512原生裁块、greedy、KV cache和8192新token上限保持，未修改任何原baseline包Python或共享数值评分模块。新增rasterio/tifffile依赖供FLAIR。

WSL用例已编写：任意奇数/17 batch、单/双图对应、尾批、互斥分片、PNG恢复、全局混淆矩阵合并、六类变化并集、名单/RGB拒绝、OOM不掉样本；本机遵循要求不执行测试/模型，只做AST/bash/JSON/名单及旧源码保护检查。代码与完整main离线包将交付；实际速度、显存、格式有效率、CUDA兼容和全量精度仍待WSL回执。所有命令集中RUN_QUEUE，不修改论文数值或训练。

## 2026-11-02：用户要求先核对训练数据，任意batch/双卡仍在开发

核对当前CVPR主稿与补充：分割为LoveDA、FLAIR、UAVid、xBD-pre、CHN6-CUG；变化检测为LEVIR-CD和SECOND（593对DynamicEarth验证集、六类变化双向合并），不是SECOND生图1694测试对。随后用户要求先解释RSUniVLM训练数据，先完成一手来源核查，暂不把开发草稿打包交付。

原论文arXiv v2第4.1节、附录表10/12明确包含LoveDA与UAVid分割训练以及LEVIR-MCI变化训练，两阶段均涉及上述像素级来源。不能称RSUniVLM在LoveDA/UAVid上是数据集未见的零样本，也不能仅凭用了数据集断言测试泄漏。公开checkpoint逐样本manifest、具体划分和LEVIR-MCI/LEVIR-CD重叠仍未审计。其他未列名的评测集也不直接等同无重叠。完整来源和表列数据见TRAINING_DATA_AUDIT.md；稿件未修改。

跨图batch16草稿已写入工具及WSL合成回归文件，用户进一步要求任意正整数batch和双卡；这些要求尚未完成。草稿未测试、未提交、未推送、未生成新离线包，不改变已交付的85a44ce以前代码能力。后续应继续实现任意batch和双卡，再做静态核查及WSL交付。仍遵从本机不执行测试/模型的要求。

## 2026-11-02：用户要求增大batch，提供现有直跑batch4配置

用户明确需要加大实际batch；核查当前wrapper与BatchBackend后确认无需代码改动：TILE_BATCH_SIZE=4、VERIFY_BATCH=0时不比较单图，因此不会触发比较失败的serial_fallback；新协议身份也不会继承原VERIFY_BATCH=1目录的回退记录。保留OOM拆批和严格格式重试。配置及完整启动命令写入唯一RUN_QUEUE。

批量与单图已经出现差异，因此用新`..._v5_bs4_direct`目录，清空IMPORT_V4_ROOT，从完整1669张首次请求开始按批量路径推理，不把单图缓存混称batch4结果；原单图文件保留。每张原图4裁块，首次请求batch4，失败子集重试可小于4，实际大小以raw生成元数据为准。权重/空间处理/精度/attention/greedy/格式预算/评分函数保持，命令显式指定原fp16/eager。

仅更改运行记录，不改正在运行的数值源码，不代中断或启动WSL、不在本机测试。用户需先Ctrl+C原单图进程回到提示符，再用同一GPU启动新命令；当前已交付代码包即支持，无需为本次命令更新包。尚无这次直跑的GPU速度/显存/精度回执，不称为已加速成功。

## 2026-11-02：WSL单图回退生效，旧断点后继续到166张

用户回传ec6929f路径的运行日志：模型加载无残留meta，首次实际batch3中1例与serial不同，按既定策略使用该组全部serial并将后续固定batch1。Val_Rural_2685的4裁块raw_reused=1，两个初始water缺计数经过重试后整图完成；随后2686与2687也完成，最新进度166/1669。该回执证明实际恢复已越过旧163断点，不能再记为“只有静态、WSL尚未运行”；合成回归是否在WSL执行仍未提供，完整基准尚未完成。

首图generate=27.33s、verify=9.71s，后两图generate=6.50/8.43s、verify=0，均为日志墙钟阶段时间。进度条约8s/image，显示余时3h20–30m，仅为动态估计。cached=0是PNG未复用，不能抵消raw_reused；miou=0.0547/0.0511/0.2134对应三张图，不写为全量分数。核验两路完整原文仍未取得，差异原因未定，不直接归为浮点误差。回执归档receipts/wsl_loveda_v5_serial_fallback_progress166_20261102.json。

本次仅保存用户证据、更新队列和状态说明；运行代码、权重、配置、已有离线包不变，不代启动/中断WSL、不在本机运行测试或模型。以用户回执时运行中表述，无SSH/PID实时核验。最终1669张结果待继续回传。

## 2026-11-02：首个新合批与单图不一致，改为复用参考并持久回退batch1

用户从离线包更新后回传：模型加载完成且无残留meta，v5处理163/1669后在BatchBackend.generate_batch的核验分支报Batched/single predictions differ。此处不是meta warning或CUDA OOM导致中断。没有取得batch_verification.json两份原文，不能确定具体差异或实际子集batch大小，更不能直接归为FP16误差。回执归档receipts/wsl_loveda_v5_batch_mismatch_20261102.json。

原先“检查不一致就终止”的恢复方式改为：当前整组全部采用已计算出的原单图参考，后续固定effective batch1，不重复计算当前参考，不按GT、分数或合法性择优。单图无效仍走原有限格式重试。batch_verification保留两路原文、case对应raw路径/identity和serial_fallback，重启同协议恢复回退；匹配时保留原合批路径。最终metrics显式标记requested/effective batch与回退原因。新增策略和源码哈希进入协议身份，旧v5保持，v4严格核验及原文导入不变。

补充WSL合成回归：不一致整组采用serial、各裁块不串位、无重复参考调用、原文记录、重启不再探测、不同协议不继承回退、匹配时继续batch。遵照用户要求，本机只做AST/bash/旧数值源码保护检查，不执行测试或模型。生成完整main离线包rsunivlm_serial_fallback_20261102.bundle并做Git导入校验；WSL实际恢复、速度与全1669成绩仍待回执，不能承诺此回退能加速。

## 2026-11-02：按用户要求生成完整main离线包

交付`output/offline/rsunivlm_batch4_resume_20261102.bundle`（相对外层Vistar工作区），覆盖343db0f的目录冲突修复、7b3d54e的batch4/别名/原文导入及交接记录。采用完整main历史，避免增量包依赖某个WSL前置提交；不收录任何未提交工作区内容，不包含预训练权重或缓存/数据。包旁回执记录tip、SHA256和Git对象验证，WSL导入步骤在唯一RUN_QUEUE。本轮不运行模型或实验测试，打包不能当作推理成功。

## 2026-11-02：v5目标协议冲突，保留已有目录并自动选协议路径

用户WSL已更新并运行batch4命令，依赖/元数据通过，import_raw_answers在目标目录协议比较处停止，还未加载推理模型。源码次序表明旧v4 run_config和输入清单已验证，失败是当前期望与既有v5 run_config不同；用户未提供该文件，不能断定具体哪个字段不同。归档于receipts/wsl_loveda_v5_output_conflict_20261102.json。

新增纯输出管理：递归报告差异字段，既有输出协议不符或缺身份时保留原目录，选同级带当前协议哈希的确定性路径；hash目录如再冲突则报错，绝不覆盖。主入口同步args.output_dir、后端、导入器和评分写入同一actual路径，记录output_resolution.json。旧v4严格核验、独立原文导入、别名、模型、裁块/推理/指标全部保持；不自动采用不明v5内容，也不让更换目录成为绕过v4源身份检查的手段。

RUN_QUEUE仍使用原batch4命令，先拉取更新，actual路径在终端显示。补充WSL合成回归（匹配/不匹配、保留原文件、缺配置、确定性重启和hash目标冲突）。遵循用户要求，本机只做AST/bash/旧源码哈希保护检查，不运行测试或模型；修正后实际WSL运行待回执。

## 2026-11-02：v4在bareland中断，准备v5裁块batch4与原文恢复

用户全量回执停在163/1669，失败原图Val_Rural_2685、裁块y0_x0，最后错误是第22行`bareland`；三次既有格式预算耗尽。新增回执`receipts/wsl_loveda_v4_bareland_failure_20261102.json`只记录已提供的错误和进度，不捏造原文或完整指标。`bareland`明确对应LoveDA barren，新增两种显式别名，其余未知词及无计数`water`仍严格拒绝。

新增tools下的独立v5入口，原RSUniVLM包Python字节不变。每序列一图、固定提示/expert2/greedy/384视觉输入，原512非重叠拼接和评分函数保留。请求batch4，CUDA OOM拆分2/1，首批各大小对照原单图并记录一致性（不读取GT）；格式失败子集按既有固定提示重试。不同协议使用独立v5目录，身份包含新增映射、batch和扩展源码哈希。

通过旧协议/输入/裁块/提示核验后只导入v4原文，选择扩展别名下最早合法的存储回答，重建预测并重新评分，保留原始尝试和导入来源。能避免已完成163张再次生成，但仍需CPU评分；旧分数/PNG不复制，旧目录不覆盖。新增逐图生成与评分耗时，SAVE_IMAGES=0仍只保存预测掩码及必要文本/评分。操作命令见RUN_QUEUE，设计细节见BATCHING.md。

遵照用户“这里不测试，先push、WSL测试”，只做AST、bash语法、旧数值源码保护检查；WSL合成测试仅编写未运行，无本机模型或GPU任务。实际吞吐/显存/数值一致性/全量精度均待WSL回执，不能写为已通过。meta警告也不能仅凭无残留meta就宣称所有视觉权重数值已核对。

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
