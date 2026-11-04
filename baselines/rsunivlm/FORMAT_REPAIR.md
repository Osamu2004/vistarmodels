# 已保存RLE的离线格式修复（2026-11-04）

## 固定规则与完整校验

新增tools/rsunivlm_format_repair.py与tools/repair_rsunivlm_format.py。默认只处理已有回答，不加载模型、不查询CUDA、不增加生成尝试：

- 类别和整数计数之间的ASCII冒号改成一个星号：building:1 → building*1。
- 类别和整数计数之间的连续多个ASCII星号缩成一个：road**2 → road*2。
- 不改类别文字、整数原文、逗号、空白、换行、行数、时相顺序或generation/EOS标志。不补计数/行/背景，不裁剪或重新平衡行宽，不猜测破损词汇；Unicode标点、引号、Markdown词内修饰等不在v1规则内。

规则仅在完整单项显式写出“类别+分隔符+整数”时替换分隔符。候选必须由原native.decode_binary或protocol.decode完整解码成功才保存；多行中另一项仍坏、行宽不一致、计数不合法或类别不符合原数据集映射，整份候选拒绝。原解码器允许的native矩形1..64边长保持，不擅自改成必须24×24；例如building*1按原规则是1×1，工具不凭空补另外23行。

类别解释也完全沿用原解码器：native_classwise的官方demo规则是others为0，其他语法合法的标签均为1；修复器不新增类别白名单，也不声称能发现这种原解码器本就接受的错误词汇。SECOND等有显式映射的分支仍由原映射拒绝未知类别。

明确ended_with_eos=False的坏回答不修复；不移除截断标志。原本已合法的记录保持原样。若一次查询有多个旧尝试，只要其中已有合法回答，就不再修复更早的无效回答，保留原来的选择。只有所有原尝试均无效时，才按原顺序选第一个可机械规范并通过原解码器的候选，不读取GT来选择。

## 默认流程：纯CPU修复、重建与重计数

原队列必须已退出，source_run的plan/config、源码身份和结束回执必须一致；活动队列锁、尚存活的未结束调用记录、输入元数据变化和原文身份冲突均拒绝。与旧模型补跑工具不同，此阶段允许来源因CUDA中断而尚未遍历完整名单，已有回答仍可修复；尚未生成的回答单独保留为待推理。

默认输出为源目录同级的`<源目录>_formatfix_v1`。新解码后处理策略、源运行/plan SHA及工具/导入器源码SHA纳入新identity；不冒充旧零后处理结果，也不覆写原raw_text、mask、receipt、metrics或论文。

1. 复用原来校验通过的完整掩码，优先硬链接，跨文件系统复制；原成功查询不重新生成。失败图中的原文和真实尝试历史导入独立目录。
2. 只修复没有任何原合法回答的查询；其余原文不变。每个接受修复的attempt保留original_response、前后SHA、分隔符编辑清单、原解码错误及修复后网格尺寸。未通过整份校验的候选不写回response。
3. 在CPU上用原Engine/解码/还原/评分函数重建可完成的整图并重新汇总计数。OfflineEngine.core被禁止，forward不生成或伪造回答；GT仍在整图预测落盘后读取。
4. 缺失的旧回答不被虚构成一次模型尝试；不能修复的坏回答也不补背景。整图仍缺任一查询则不评分，输出partial。缺失/待模板查询分别记录，存在这类待生成查询时exhausted_population=False，旧“仅格式补跑”入口不会将未推理人口当作已跑完。
5. 重复相同命令可续跑修复目录，复用新旧成功结果。始终传原始source_run；不要把派生目录当作该工具的source，或直接用主队列Bash去覆盖派生协议。若参数/工具变化导致身份不同，按原规则保护旧输出并选_p后缀。

CPU阶段返回0表示离线阶段完成，不表示全部原始样本均已修好；以format_repair_report.offline_complete与paper_summary.complete/num_failed/deferred_queries判定覆盖。

2026-11-04追加：用户报告LEVIR-CD在读取test_18/test_68的GT时中断。新版只要选择了LEVIR，就在派生策略中记录并安装精确二值颜色兼容器；先审计所有标签，通过后复用预测CPU重算。原15个数值源码与旧运行不改，具体规则见LEVIR_LABELS.md。旧_formatfix目录不混用新政策，实际可能选_p后缀。其他数据集GT规则不改。

后续严格审计确认5张是L标签且含156/254，默认exact_binary因此仍拒绝。现追加显式--levircd_gt_mode vistar_positive，与既有Vistar的灰度>0评分统一；对应审计参数为--gt_mode vistar_positive。该策略及helper SHA单独记入派生身份，原GT不改，审计/GT读取记录异常正灰度值和像素数量。用户此前63项测试通过属于旧版，新选项的WSL测试及实际重算尚待验证。

## 可选模型补跑

显式增加--run_remaining才允许调用模型：先做同样的离线修复/重建，再对仍缺失或不可修复的查询启用既有最多两个固定备用提示。保持原卡数/分片/batch16上限，新目录默认`<源目录>_formatfix_v1_retry2`，新identity明确记录重试策略。已耗尽原三次模板的回答不无限重试。

这条分支已连接最新guarded监督入口和有序DataLoader。缺卡不启动；CUDA致命错停止队列。若CPU阶段已经完整，跳过GPU。新生成回答仍由原严格解码器处理；再次执行同命令时，可继续机械规范其保存下来的坏格式。不会自动不断调用模型或按GT挑结果。

旧tools/retry_rsunivlm_failed.py也改为guarded入口，并支持身份匹配的offline_runtime_finished.json作为CPU修复结束回执；其“来源各分片必须已完成查询人口”的限制保留。仍有缺失回答/未执行备用模板的CPU分片不满足该条件。

## 产物与验证边界

- format_repair_report.json：导入/修复/无法修复/既有修复计数、CPU完整性、待生成查询、是否请求/调用GPU续跑，以及实际worker启动回执。
- format_repair_scan.jsonl：本轮修复和拒绝记录；原文精确备份在对应attempt.format_repair中。
- raw_text、pred_mask、receipts：独立派生协议下的原文、规范文本和重建掩码；没有新增RGB/GT/叠加图。
- offline_runtime_finished.json：CPU阶段身份/结束/0模型调用证据；不覆盖guarded的GPU运行回执。
- paper_summary.json、逐集metrics：沿用原评分并明确成功子集/待生成状态。多类native仍为独立类查询，不因此声称原论文未公开的互斥融合已复现。

WSL回归：tests/test_rsunivlm_format_repair.py、tests/test_rsunivlm_format_repair_pipeline.py及旧补跑回归。覆盖分隔符、原文本/计数/换行不变、原解码完整矩形/类别拒绝、EOS、合法旧尝试优先、部分CUDA中断与缺查询、源文件不变、CPU重计数和相同命令续跑、显式GPU分支监督路由。

本机遵守用户约定，仅AST/源码哈希/Bash静态校验，不运行这些测试、解码、模型或CUDA。实际能修复多少、WSL回归和评分变化需要用户执行后才有证据；未修改原15个数值源码，未直接更新论文分数。完整命令统一RUN_QUEUE.md。
