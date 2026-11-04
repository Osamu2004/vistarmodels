# RSUniVLM 接入记录

## 2026-11-04：按要求制作最新约束RLE离线包

以约束代码9e677ef和本段安装文档的已提交main固定快照制作完整Git bundle/ZIP，附SHA校验、保留用户数据路径JSON的标准库导入器、中文README和静态回执。不打包未提交的其他基线、权重/数据/预测。RUN_QUEUE提供解压/安装与仓库内正式RLE恢复脚本命令；安装本身不测试或启动GPU。

本机仅归档/Git传输校验：空bare仓库导入、对象fsck、旧提交快进、用户路径JSON原字节保留和本地冲突拒绝后的恢复、ZIP CRC/成员SHA及Bash/AST/JSON静态验证。不运行pytest、tokenizer、原解码器、模型或CUDA，不实际接入WSL。包tip/SHA/大小与传输回执记录外层output/offline；实际推理完整性仍待用户运行。

## 2026-11-04：用户授权添加约束RLE生成，独立兜底策略

在说明自由输出无法保证合法格式后，用户要求添加RLE。实现finite token-trie约束，固定网格默认24×24、只允许others/查询前景、正计数行和固定、完整后EOS；模型仍选择预测内容，原decoder后追加网格/类别/EOS校验。通过官方HF4.45.2 logits_processor接入原GMoE generate；核对上游固定提交inputs_embeds路径转发kwargs/初始空生成前缀，避开该版内置PrefixConstrainedLogitsProcessor的零长度view问题，不添加BOS或猜测提示词切片；验证tokenizer片段往返和最坏token预算。

前两次自由输出/机械修复不变，最后retry启用约束；旧三次已耗尽的查询在CPU失败名单中分组补一条真实约束回答，不重跑成功查询。缓存导入支持明确tagged第四条历史，模型条件/人口/分片验证保留；新策略/helper SHA/网格入独立_formatfix_rle_v1身份。旧mask/原文/重试次数复用，不强改原15个数值源码或GT/时相顺序/评分。仍异常保留失败，不声称预测准确或无硬件故障。

新增受版本记录的生成hook、runtime、tokenizer CPU预检与run_bash/rsunivlm_recover_rle.bash，WSL回归覆盖有限语法/HF logits屏蔽/实际hook恢复/旧第四次兜底及最后retry/失败不补掩码/不重放。当前无WSL连接，本机仅AST/Bash/JSON与原15源码字节检查，不运行任何测试/解码/tokenizer/模型/CUDA；真实可运行性和恢复率待WSL回执。命令、来源和新旧协议区别见RUN_QUEUE及CONSTRAINED_RLE.md，旧队列结束后再更新。

## 2026-11-04：用户选择保留协议，一次调用自动修复/补跑/终检/汇总

用户希望不再手工多轮调用，明确选择保留当前推理协议，不增加约束RLE解码。新增recovery_runtime包装器，对每个真实新回答先用原两条分隔符规则及原解码器规范；不改前向、提示词、网格或三次总预算。GPU正常结束后自动末尾检查，只有新增修复的集才做CPU重计数，输出逐集原失败整图恢复数量；GPU致命/未封口回执不伪装成功。

新增--reuse_run，严格验证并先导入旧派生运行，再由原source填空；当前已恢复LoveDA64、FLAIR1152等工作及真实重试历史不会丢弃，也不清零已用次数。活动队列锁拒绝迁移；默认新_formatfix_once_v2保存新执行策略身份，原目录不改。已完整或所有剩余历史耗尽原预算时不做无效GPU扫描；仍坏保持partial，不能承诺全部图片成功。

补充WSL合成回归与RECOVERY_ONCE.md、RUN_QUEUE。旧队列未结束前不更新正在使用的源码；本对话没有WSL连接，没有启动本地/其他远端实验。只AST/Bash/JSON与原15源码字节检查；新增回归、一次调用恢复效果/分数待用户WSL回执。用户贴的是运行中快照，不能当作最终恢复统计。

## 2026-11-04：授权七集失败恢复，统一先修复后GPU补推理命令

用户要求开始处理失败样本：可修复的机械修复，其余重跑。核对当前6ea3329入口已支持--run_remaining；RUN_QUEUE新增七集完整命令，先vistar_positive审计，再离线导入/规范/CPU重建，余下查询进入原两卡分片、每卡batch16的guarded补推理，最多两个既有备用提示。成功原文/掩码复用，不重做成功模型前向；LEVIR未计分但已保存的有效预测也可复用。GT不重写，原15个数值源码与查询语义不改。

本对话没有用户这台WSL的可连接入口，未启动任何本机/其他远端任务；状态为待用户执行，既有63项旧版通过回执不充当新规则/本轮恢复验证。只对新命令做Bash语法和差异静态检查，不运行本机测试或模型。相同命令可续跑；新GPU回答的机械格式错误需下次同命令规范，当前不自动无限循环或重置模板预算。实际修复量/补跑量/七集完整性以新输出报告为准。

## 2026-11-04：WSL确认5张L标签含156/254，显式对齐Vistar正值规则

用户完成旧版63项回归（7.36s）；严格GT审计128张，123通过、5拒绝：test_18值0/254/255，test_68/70/73/74值0/156/255，均L/uint8/1024×1024。因此并非RGB问题；原&&链在审计停止，尚无新重算分数。完整回执归档receipts/wsl_levir_grayscale_audit_20261104.json。

再次核对现有Vistar compute_saved_levircd_opendpr_metrics._load_binary与模型仓库levircd_common.load_binary，二者都PIL convert(L)→uint8→>0。增加显式vistar_positive模式，审计--gt_mode、修复--levircd_gt_mode统一指定；严格模式默认不变，不将5张GT文件重写或逐张特判。mode/policy/helper SHA进新identity，正值审计另存不覆盖严格审计，记录非标准正灰度值/像素数；保存预测照原哈希复用并CPU重算。

补充与现有参考读法逐像素一致、严格默认仍拒绝、正值分片CPU恢复/续跑、模式身份分离及审计保留的WSL回归。本机不执行测试/解码/模型/CUDA，仅静态校验。旧63 passed不是新模式已验证；新正值审计结果、恢复数量/分数待用户运行。用户现已Git更新，RUN_QUEUE新节使用Python入口，不要求离线包脚本。

## 2026-11-04：LEVIR GT格式校验中断，保留预测CPU恢复

记录用户真实WSL回执：LEVIR已评分64、格式失败1、63张未计入；两分片分别在test_18.png/test_68.png的load_targets校验失败，实际mode/像素仍未知。原finish先保存掩码/receipt再读GT，所以未评分不等于未推理。

新增独立rsunivlm_levir_labels兼容器与纯CPU审计CLI，原二维0/1/255逐像素不改，只追加精确同通道二值RGB、不透明RGBA/LA及旧索引不合法时的精确二值调色板读取；中间灰度/彩色/透明或尺寸异常仍拒绝。不改GT文件、原15个数值源码、A/B查询/裁块/batch。政策/helper SHA进入派生plan，接入格式修复CPU重算与显式guarded GPU补推理，普通旧主队列仍保留旧策略。

补充WSL回归代码、来源错误receipt与RUN_QUEUE审计/恢复命令，并制作新的levirgt离线包。仅本机AST/Bash/源码字节及Git/ZIP传输检查，未运行测试、解码/模型/CUDA或真实GT审计，尚无恢复数量/新评分。细节见LEVIR_LABELS.md。

## 2026-11-04：显式RLE分隔符离线修复与CPU重建

用户明确要求实现road**2/building:1→road*2/building*1并用原解码器完整校验。新增rsunivlm_format_repair.py，只替换完整类别/整数间的ASCII冒号或重复星号；只有所有原尝试非法时才取首个规范后合法候选。任何已有合法尝试、类别词/计数/换行/EOS均不改；ragged/缺计数/显式未结束/未知原映射等不放宽，不填背景。每个接受修复的attempt保存精确原文、前后SHA和编辑列表。

新增repair_rsunivlm_format.py默认纯CPU导入来源缓存、机械修复、原Engine重建/评分；OfflineEngine禁止模型core，缺失查询只记待生成不伪造模型attempt。来源可以是已退出的CUDA中断分片；源锁、plan/config/15个数值源码与原文/输入身份必须一致。派生_formatfix_v1协议纳入后处理/工具/导入器SHA，原目录不改；完整旧掩码硬链接/跨盘复制，CPU汇总仍沿用原计分，不平均子集mIoU。未推理查询使exhausted_population=False，保留partial，不当作格式失败已跑完整人口。

显式--run_remaining才允许在_formatfix_v1_retry2派生策略下用两个既有固定模板补推理，并经最新guarded/DataLoader监督。旧retry_rsunivlm_failed.py也改路由至guarded，并仅接受身份匹配的CPU结束回执；未完成查询人口仍拒绝旧格式补跑入口。新的CPU结束记录单独存offline_runtime_finished.json，不覆盖GPU故障回执。未自动执行任何GPU任务或更新论文。

新增WSL合成回归覆盖机械规则及原解码、失败候选不保存、历史合法优先、原计数/EOS、来源不变、CPU重建与重复续跑、缺失/CUDA中断、输出/活动锁/数值身份保护及显式监督补推理。按用户要求，本机仅AST/Bash/源码哈希静态检查，不运行测试/解码/模型/CUDA；实际修复率和WSL评分待回执。用户命令统一RUN_QUEUE；另制作20261104格式修复离线包，已提交main完整历史，路径保护导入器保留用户JSON，包SHA/tip/传输验证以外层output/offline的manifest和receipt为准。20261103离线包不包含新修复工具。

## 2026-11-03：按用户要求制作DataLoader离线交付

用户要求离线包。以已提交DataLoader代码f29b995及本段安装文档为固定Git快照，制作含完整main历史的bundle和ZIP，随附原路径保护导入器、SHA校验安装脚本、中文README与静态验证记录；不包含任何未提交基线改动或模型/数据/预测。RUN_QUEUE新增离线解压/安装命令，安装不启动测试/GPU；WSL测试与原七集续跑单独由用户执行。

归档校验只验证传输：bundle可导入空bare仓库、对象fsck、ZIP CRC/文件SHA，以及合成Git旧版本的快进导入和用户路径JSON字节保留。不运行本机pytest、读取DataLoader样本或模型/CUDA。实际tip、SHA和验证回执存外层output/offline，不能把离线导入校验记为WSL分割/速度测试通过。

## 2026-11-03：参考Vistar验证DataLoader，保留原七集缓存续跑

用户指出同步读图过慢，要求使用DataLoader且与当前一样续跑，随后明确参考自己的Vistar代码。只读核对相邻vistar的LoveDA/SECOND验证shuffle=False、drop_last=False及基类prefetch=2/pin_memory=False；未修改Vistar文件或引入训练随机增强。

新增rsunivlm_data_loading.py，guarded worker安装有序RGB读取包装，真实protocol解码函数不改。每卡默认2个spawn CPU worker、每worker预取2个源样本；DataLoader关闭自动组批，原评估逻辑继续按类别/时相和512源裁块组装每卡batch16。CPU uint8共享内存传输、独立CPU generator、FIFO、原global-index-mod-world分片、元数据检查、失败后恢复模块与关闭workers；不读取GT像素，不做CUDA/权重加载。每集/卡I/O统计和运行时代码SHA独立记录，不修改原15个数值源码或缓存identity，原掩码/有效原文/坏原文预算仍由旧evaluate判定。

新增WSL合成回归覆盖像素/时相/FLAIR、spawn顺序与CPU RNG、七集同批输入/掩码/计数、全缓存和部分原文续跑、零重试格式失败、双分片、坏未来图片/异常清理/空分片。未在本机执行测试、模型或CUDA，只做AST/Bash/JSON与保护源码逐字静态检查。真实速度、WSL回归和显卡恢复均待用户回执，不宣称已提速。

RUN_QUEUE记录更新/测试与同原输出完整命令，NUM_WORKERS=2/PREFETCH_FACTOR=2可调；正常吞吐命令将旧CUDA_LAUNCH_BLOCKING诊断从1恢复0，监督与faulthandler保留，生成数值参数不改。所有其他基线修改保留；发布状态以Git回执为准，未启动新实验。

## 2026-11-03：按用户要求发布监督保护与原七集续跑命令

用户要求push并给运行代码。仅暂存本任务的监督入口/Bash/合成回归、故障审查/回执与RUN_QUEUE/本记录，保留其他基线本地修改。再次做AST/Bash/JSON/差异及15个数值源码相对fccfd94的逐字核对，不在本机运行测试、CUDA或模型。发布状态以随后真实Git回执为准，不把本地提交当已推送或WSL验证。

RUN_QUEUE新增先更新并在WSL执行test_rsunivlm_guarded.py，再按原GPU0,1/batch16/native_classwise/七集/512源裁块/32k/零格式重试/原输出续跑的完整命令。显式保留原官方副本和双权重路径，CUDA_LAUNCH_BLOCKING为诊断同步；无改驱动、降batch、换卡数或删缓存。缺卡时监督入口应拒绝启动，宿主设备恢复仍未取得回执，格式失败仍需独立补跑，不声称全量评分完成。

## 2026-11-03：Windows侧同样缺卡，完成静态审查并补致命退出保护

用户澄清Windows PowerShell和WSL的nvidia-smi都只显示一张卡。当前必须将宿主设备/驱动异常与Python控制流分开；没有PnP/事件记录，不能把供电、PCIe、显卡故障、TDR或负载触发的驱动bug写成根因。暂停继续负载建议，RUN_QUEUE新增Windows只读设备/事件采集，未执行重启或驱动修改。

逐项核对队列/模型/批处理/原生类别模式/生成参数、官方66dce1d的GMoE/多模态拼接/示例及本机已下载配置。确认旧worker在illegal address后继续复用Engine，是后续集合连锁失败缺陷；父进程对worker信号退出没有fail-stop且追加日志无启动边界。B16与32k/use_cache是长尾显存风险而非已验证掉卡原因；官方示例亦用32k，不能仅因该值归因。未发现两模型错误共用物理GPU0、跨批持续保存GPU输出或重置/超频/禁用设备指令。实际WSL依赖/源码修改与初始故障算子仍未取证。完整证据与边界见GPU_FAILURE_AUDIT_20261103.md。

新增eval_rsunivlm_guarded.py监督入口，Bash转入该入口，原15个数值身份源码保持。前置只读nvidia-smi选卡存在性/UUID校验；缺卡阻止启动；包装原evaluate仅在CUDA致命异常后以SystemExit跳出旧捕获，父进程停止自己启动的peer；faulthandler、启动独立日志、批次CPU元数据及明确不完整运行回执保留，致命错误不再合并旧summary误作新结果。提示/权重/分块/batch/token预算/评分数学不变，保护代码单独记SHA。未声称修复宿主异常。

新增WSL合成回归覆盖缺卡、UUID重复、致命错退出、普通格式/OOM、对象透传及同队列peer终止。遵守用户要求本机仅AST/Bash/源码哈希静态检查；没有运行测试/模型、连接WSL或启动GPU，当前未提交/推送，也未在用户机器安装。

## 2026-11-03：faulthandler定位设备检测崩溃，纯torch隔离待运行

用户直接粘贴诊断回执：队列依赖与Engine/torch导入通过；第3阶段torch.cuda.is_available在torch/cuda/__init__.py:180触发SIGSEGV，尚未到get_device_name或权重加载。故障确认在CUDA可用性检测路径，不能只据此断定驱动损坏、排除项目依赖冲突或解释最初batch16推理非法地址。官方torch源码的默认检查经cudaGetDeviceCount/cuInit，WSL CUDA依赖宿主Windows驱动；实际用户环境版本/库路径本轮尚未核对。

RUN_QUEUE新增同环境nvidia-smi及仅导入torch的一元素CUDA检查，隔离项目导入影响，保持原模型/两卡batch16/输出身份。仅纯torch也失败且其他WSL任务已保存时，提供Windows PowerShell的wsl --shutdown、wsl -d Ubuntu-22.04作为状态恢复尝试，明确停止所有WSL进程且不是已验证修复；重开后需重复纯torch及项目启动检查才续跑。没有自动关闭WSL、连接其他SSH、修改15个数值源码或启动本机测试/模型；尚无纯torch检查或恢复回执。新JSON仅记录已收到的栈和未验证假设。

## 2026-11-03：新回执确认双worker SIGSEGV，先隔离启动阶段

用户附件95d92530给出runtime_finished.json：owned_pids=[15324,15326]，exit_codes=[-11,-11]。按POSIX/Python子进程约定为SIGSEGV，说明本轮两个worker原生崩溃；不能归为普通OOM或RLE失败。日志尾部仍为旧FLAIR非法地址及后续集合连锁，未有日志时间，不将其计为新增推理。补齐旧rank0快照：5696/7850、格式失败1337；rank1仍6688/7850、成功5069/失败1619。原错误与新段错误是否同根因未知。

核对worker在打印数据集start前依次导入worker/torch、核对身份、CUDA可用性及get_device_name，Engine构造不加载权重。RUN_QUEUE新增与原导入顺序一致的GPU0启动诊断，python -X faulthandler配合阶段标记，仅导入/初始化设备、不加载权重或写原输出；通过也不声称双卡/完整worker已恢复。保留batch16、两卡分片与缓存身份；未改15个数值源码、依赖或原输出，未连接WSL或本机执行测试/模型。需要实际段错误栈后再定修复，不盲目降batch或重装。

## 2026-11-03：同步续跑只显示预检与汇总，尚无worker退出原因

用户按原目录/原batch16/原七集启用CUDA_LAUNCH_BLOCKING=1重启，预检人口均正确，但没有worker启动或推理日志即汇总complete=false并返回shell。只据此不能说再次发生CUDA非法访问或OOM。核对launch会将每卡stdout追加到logs/gpu_N.log，并将最新退出码写入runtime_finished.json；worker进入数据集前还有源码/上游/权重身份与CUDA可用性等初始化检查。merge会复用同身份旧分片metrics，旧complete=false不代表本轮产生新预测。

为用户准备只读、流式日志尾部诊断命令至RUN_QUEUE，含解释器、runtime退出码、worker记录和日志修改时间/尾部，以区分旧记录、初始化异常、信号终止或原生崩溃。没有得到这些WSL输出前不归因具体内核/批量/驱动，不修改数值源码或缓存。没有连接WSL、运行模型或本机测试。

## 2026-11-03：GPU1 FLAIR非法显存访问导致同进程后续集合连锁失败

读取用户附件5fe6955d的完整终端片段：rank1在FLAIR6688/7850处，成功5069/格式失败1619，触发CUDA illegal memory access；随后UAVid/xBD/CHN6/LEVIR/SECOND各自0张即报同样错误，最终complete=false并回到shell。未看到rank0状态，不推断全部数据集结果。该新片段包含先前格式日志没有的CUDA致命错误，两者分别记录。

核对当前worker源码，所有Exception均写出错误后继续复用同一Engine；NVIDIA官方文档明确illegal address后CUDA进程处于不一致状态，必须退出/重启。可确认连锁传播的控制流问题，首次出错算子仍需errors/rank1_flair.json完整栈及CUDA_LAUNCH_BLOCKING=1同步定位，不能据此认定OOM、batch因果或特定attention/驱动缺陷。现存mask/hash与raw_text恢复逻辑可用于新进程同协议续跑，未处理人口应先补齐，再做格式恢复。

本轮仅记录证据和WSL只读/续跑命令至RUN_QUEUE，不修改队列/模型/提示/解码/指标源码或依赖，保护原输出身份。没有连接WSL、启动GPU或运行本机实验/测试，未取得重启后恢复回执。控制流的致命错误退出/进程隔离修正仍待单独实施与兼容性验证，不把同步诊断开关称作修复。

## 2026-11-03：独立补跑格式失败，保护正在执行的数值源码

用户日志FLAIR两卡分别4240/7850、failed966及4704/7850、failed1131，仍在继续；主要是缺计数、损坏RLE项和不等宽行，未显示OOM或数据损坏，不能推定batch因果。failed按整图计算，12类任一查询出错便排除整图。相同原命令只跳过已耗尽零重试预算的坏原文，不能自动重新生成。

新增独立retry_rsunivlm_failed.py，不触碰现有queue/native/worker/模型/提示/解码/指标源码。原整个队列结束且锁释放后，核对源码/权重/输入/分片身份，独立目录复用已完整掩码和部分有效原文；只为坏查询继续现有两个固定模板，记录旧尝试来源及新预算。成功掩码优先硬链接以节省磁盘，GT仍在预测落盘后按原实现计数，全局重算分数，不平均子集mIoU、不背景填补、不按GT选输出。保留两卡分片和batch上限，稀疏失败查询实际batch可更小，不声称逐字等价。非格式中断集合单独拒绝/报告；剩余失败仍partial。

新增WSL合成回归覆盖FLAIR和SECOND复用/恢复、只执行坏查询、两卡计数汇总、自身续跑、剩余模板失败不死循环、原文件不变、活动锁/输入/源码/回答身份保护、跨盘复制与plan-only。按用户要求本机不运行测试/模型，只做AST、数值源码逐字比较和Git归档完整性校验。真实恢复率/吞吐和WSL测试待回执。命令统一RUN_QUEUE，离线包使用保留用户路径的安装器；不修改论文分数。

## 2026-11-03：格式失败记录整图并继续队列

用户要求不要格式重试耗尽后退出，并提供native FLAIR32张真实回执。两卡模型已加载且无剩余meta参数，实际batch16；rank1建筑回答出现`building:1`和引号分隔，rank0针叶树出现无计数/破损文本，各自0/16后终止。当前FORMAT_RETRIES=0，本质是首次无效RLE就中止，非CUDA/OOM错误；没有取得成功整图或新版评分。

修改native与通用worker的失败控制流：只将格式预算耗尽的整图记入失败名单，进度继续，同批其他图、后续样本与后续数据集继续处理。任何缺失类别/裁块的图均不保存不完整掩码、不读取GT评分、不背景填补；原文与失败路径保留。逐卡/全局记录attempted、processed、num_failed、failed_samples、exhausted_population、scores_scope，失败未补齐仍incomplete，所有数据集走完后才返回最终失败码。模型/提示/解码/采样/几何/计分数学保持；权重身份或数据错误仍按原规则报错。

新增WSL回归覆盖首次/尾批失败、0/2重试预算、全失败、同批继续、完整掩码才计分、已知失败恢复不阻塞、LEVIR/SECOND及worker后续数据集。仅本机AST/Bash/数值区域与保护源码核对，未运行测试/模型。更新失败策略改变协议身份，旧目录保留，新版同身份可恢复；不跨版本静默导入原文或分数。离线包包含路径保护导入器，命令归于RUN_QUEUE。新版真实效果尚待用户执行。

## 2026-11-03：按原论文逐类别语义查询，并核对表7口径

用户要求按原论文推理，进一步明确要对齐表7的Vaihingen54.46/UDD5 65.41/VDD49.60类别处理，而不是沿用一次要求输出全类别的接口。重新读取arXiv v2附录、固定66dce1d app.py、完整155文件Git树及三个现有eval文件；原生可核实部分是单类提示、greedy/像素专家、others背景及其余文本标签前景、patch20+OpenCV最近邻还原。没有找到表7多类融合和评分脚本；公开期刊页403、HF演示不可访问，未据此推断未公开细节。

新增隔离的native_classwise语义执行模块并设队列默认：LoveDA7、FLAIR12、UAVid8类全部固定查询，xBD建筑/CHN6道路各一次。支持同类跨图任意batch、双卡分片、OOM折半、部分原文与预测恢复；保存全部查询掩码后才读取GT。按每类全数据TP/FP/FN求mIoU/mF1，保留query别名和详细计数/重叠诊断，不使用类别先后覆盖或GT选类。来源和评分推定部分单独记录：table7_protocol_verified=false，多类不冒充已核实的互斥分割标准基准，暂不产生虚构WFm或填论文原表。二值集合标准IoU/WFm维持。SECOND仍用户指定六类无方向适配，LEVIR仍一次双图无反向OR。

默认首轮使用论文第一模板、FORMAT_RETRIES=0、MAX_NEW_TOKENS=32000，沿用用户512源图裁块和批量要求；不宣称这些工程适配与官方单条演示逐字等价。legacy_categorical保留显式复现旧接口。新目录保护旧输出，WSL先测试/FLAIR32张核对，再独立正式七集。新增WSL契约覆盖官方函数解码/恢复对照、固定类别、分片与非平均指标、重叠保留、预测先落盘、恢复及binary指标；本机不运行测试/模型，只AST/Bash/源码保护/离线Git完整性核对，真实效果待回执。

## 2026-11-03：按用户要求将 FLAIR 的 others 解析为背景

用户提供双卡队列日志：FLAIR 输出 others，固定重试预算耗尽，rank1 为0/7850后进入UAVid；没有完整七集完成回执。用户随后明确“others直接认为是背景类”。据此仅为 FLAIR 新增预测背景ID12，接受 others/other/background；原十二类GT、building=0、提示词、模型和裁块不变。不能把模型背景预测从有效GT像素中移除：复用现有额外预测列的混淆数学保留FN，12×13矩阵全局累加，mIoU仍只平均十二类。边界函数把12作为正常预测标签，原GT-ignore支持保持。

更新值域校验/恢复、逐卡与根合并、显式背景协议元数据和背景像素统计。未知词、格式错误、无EOS截断仍拒绝。原源码身份改变，新FLAIR独立目录；不覆盖旧失败原文，不为本次修改重跑其他六集，不填论文分数。新增WSL待运行回归，覆盖背景编码、漏分分母、边界、掩码保存/恢复、合并以及其他数据集原映射。仅本机AST/源码保护/离线Git完整性核查，未运行测试或模型，真实WSL效果待用户回执。完整ZIP附路径保护导入器，保留用户手改JSON及其他本地修改。

## 2026-11-02：按用户要求重新生成v2离线更新包

当前数值实现保持744ef46版本不变：SECOND保留六类、无方向类别查询，不交换输入或OR预测，继续原六类mIoU_c/mF1_c；LEVIR保持一次双图查询。仅更新交付文件名为rsunivlm_second_undirected6_20261102_v2，使用独立解压目录并通过Python标准库解压，避免依赖额外unzip安装或混用旧包。新包包含最新完整main历史、路径保护导入器、说明和校验文件；重新进行离线Git导入/归档完整性验证，不执行实验测试或模型。

## 2026-11-02：离线导入保留用户手动数据路径

Git传输兼容检查发现：从用户已导入的d4a27d0出发、保留其手改FLAIR/xBD路径时，普通ff-only pull会因配置本地改动而停止。新增标准库导入工具只暂存该已知JSON的原始字节到临时备份、临时恢复HEAD文件并执行本地bundle的ff-only导入，finally无论成功失败均原样恢复用户路径；不reset/stash/stage、不强制覆盖其他修改。发布zip随附该脚本，用户无需先更新仓库才能调用。此属于Git交付验证，不运行实验测试或模型。

## 2026-11-02：SECOND保留六类，改为无方向类别查询

用户明确“不是二值变化，类别也要，只是无方向”，撤销先前口头提出的整体二值化解释（未执行该错误转换）。当前保留原始两份0..6 GT与六类mIoU_c/mF1_c数学；类别改为六个独立查询输出，每类每块输入(T1,T2)一次，不交换输入、不合并两个预测方向。每类GT继续由两份语义标注的该类成员集合确定，类别允许重叠；不是一个总变化前景，也不压成互斥类别图。

新增SECOND_QUERIES固定六类、类别专属无方向提示/严格解码、同类跨图缓冲填满任意batch、六类预测掩码保存/恢复与元数据，前向/GT读取顺序保持预测全部落盘后评分。默认核对不做格式重试，未知land-cover等泛词或其他类别仍拒绝。source tile512、双GPU分片、模型加载、权重和LEViR行为保持。新second_query身份隔离旧七类/双向缓存；专用入口不重跑其余集合。

新增WSL回归覆盖六类保留、类别重叠、每类一次固定输入、GT不用于选类、保存后评分、恢复、分片计数合并、跨图同类batch、未知词/EOS拒绝；更新原几何与通用测试以对应六类查询。命令在RUN_QUEUE，说明SECOND_UNDIRECTED；本机只AST/Bash/源码保护静态检查，不运行测试/模型，不填论文分数。先4对真实核对，再独立目录593对正式评分，实际有效性/性能待回执。

## 2026-11-02：澄清LEVIR任务无方向，forward仅是一次输入调用标识

用户要求不要有方向的任务。重新读取LEVIR官方页面确认其building growth/decline统一标为change=1；只读核对当前load_targets直接将原始0/1/255标签转为(raw>0)，没有按某期建筑类别分割或筛选。现有输出/IoU/F1因此已经针对无方向二值建筑变化。

将当前文档名称和说明改为“无方向二值建筑变化，每对查询一次”。保留(A,B)固定输入、官方原始提示词、无reverse/OR、512裁块和全部评分代码；未声称交换图像后模型输出必然相同。本次只有文档澄清，不改变数值源码/协议身份，不需重新下载fbe00fd离线包；未运行测试、模型或远程任务。

## 2026-11-02：按用户要求先修正LEVIR为单向二值查询

队列LEVIR首轮prompt直接使用官方CD_PROMPT，方向固定forward，缓存/输出键改为forward，移除LEVIR reverse生成与OR合并；GT二值解码与IoU/F1、512源图几何、模型/权重、双卡调度、任意batch及OOM策略保持。协议新增levircd_query元数据，旧OR运行身份不兼容，新目录隔离且不导入旧掩码/原文。独立旧Python/启动器默认方向也改为forward，显式legacy symmetric_union仍可用于历史复现。

新增只跑LEVIR的forward_queue入口，默认FORMAT_RETRIES=0，专门核对官方首轮原生输出；先4对smoke，再独立目录全量128。第二集及其他分割分支未改、不重跑。新增WSL合成回归覆盖一次有序双图查询、二值预测与正确评分、保存后读GT、恢复、旧OR拒绝、两分片合并和格式/EOS拒绝。本机按既有要求只AST/Bash/哈希静态检查，不执行测试或模型；WSL运行回执待补。命令在唯一RUN_QUEUE，说明LEVIR_FORWARD；不改论文指标。

## 2026-11-02：SECOND旧适配失败；纠正原生变化检测接口

用户完整队列尾部回执最终complete=False，GPU0 SECOND仅处理1/297便因00031_forward格式预算耗尽结束；重试子集实际batch13/12，出现land-cover等未知类别及一份未EOS截断。本轮尚未取得完整汇总、原始回答或GPU1结果，不猜测其他六集状态。

用户连续纠正“未变化＋六类方向变化”、要求按LEVIR处理，并指出原本没有双向OR。重新读官方app.py/main与论文v2确认：公开例子为指定类别的单次有序双图二值变化查询；未实现反向OR，公开CD结果在WHU-CD，未提供SECOND六类适配的验证。此前把Alters/DynamicEarth外部协议说成RSUniVLM原生流程不准确；记录NATIVE_CHANGE_AUDIT并标记旧队列历史适配说明。临时逐类双向OR helper与测试未运行，按用户纠正撤回，未发布。旧数值源码、缓存、预测和论文保持；后续应先核对官方单向二值接口，不自动将SECOND改成二值后冒充六类指标。

## 2026-11-02：七集共同预检全部通过，准备正式双卡推理

用户WSL回执在同一次计划中发现LoveDA1669、FLAIR15700、UAVid270、xBD-pre933（test/pre-disaster）、CHN6-CUG903、LEVIR-CD128、SECOND593，最终`Preflight passed; no model loaded and no GPU process started.`。七集计划与原UAVid单集配置不同，脚本保留旧目录并解析到`/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1_p9c14c4b854db`，属正常保护，不是失败。

归档用户回执，RUN_QUEUE置顶明确PLAN_ONLY=0的双GPU/每卡16完整七集启动命令；原源图512、完整群体、只存预测掩码及必要记录、同协议恢复和OOM折半策略保持。用户尚未提供真实模型启动/前向/显存/吞吐或全量评分回执；本机没有WSL会话，本轮未修改数值源码、配置或论文，也未运行测试/模型。文档更新不需要用户重下离线包或改变已通过的运行设置。

## 2026-11-02：用户提供xBD根目录，准备七集共同预检

用户回复xBD数据路径为`\\wsl.localhost\Ubuntu-22.04\root\data\xview2`，对应`/root/data/xview2`。配置JSON仅修改xbd_pre键，保留已通过的FLAIR/UAVid新路径。只读核对`baselines/segearth_ov/protocols.py:_resolve_xbd_dirs`：支持根目录直接图像/标签布局，或继续尝试test子目录；严格数量仍为933张pre-disaster配对，不因根路径不同改变数据群体或指标。

RUN_QUEUE记录无需再次下载离线包的WSL单键配置修改及七集预检命令。当前六集各自已有通过回执，xBD和七集共同预检仍待用户执行；本机没有WSL会话、不宣称路径存在或模型已运行，未修改Python数值源码、标签或论文分数。

## 2026-11-02：FLAIR 15,700对预检通过

用户贴出WSL回执：`[preflight] flair: 15700 samples; flair#1-test`，最终`Preflight passed; no model loaded and no GPU process started.`。此前UAVid单集预检占用原请求目录；当前FLAIR单集的datasets/populations/prompts等身份不同，解析器正常保留原目录并另用`/root/data/experiment/rsunivlm_paper_queue_bs16_2gpu_v1_pfa2f270f4177`，不是推理错误，不需清理或覆盖。请求GPU0/1、batch16仅为计划配置，未进行CUDA前向。

归档用户原文关键字段并更新PAPER_QUEUE/RUN_QUEUE状态。六集各自预检通过，仍缺xBD-pre实际路径及七集共同完整预检；没有新模型成绩，本轮未改数值源码、数据根配置、预测或论文。

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
