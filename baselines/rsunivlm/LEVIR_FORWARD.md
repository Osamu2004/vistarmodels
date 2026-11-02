# LEVIR无方向二值建筑变化：每对图像查询一次

2026-11-02，按用户明确要求只先修正LEVIR。状态：实现完成，本机只做AST、Bash语法及源码范围检查；WSL合成回归、真实模型前向和正式128对结果尚待用户执行。未修改论文数值或运行模型。

## 当前行为

- **任务与标签无方向。** 建筑新增和拆除都属于同一个变化前景，0=未变化、1=建筑变化；直接使用官方二值GT，不按某一期是否存在建筑筛选GT，也不分新增/消失两类。官方数据说明：<https://justchenhao.github.io/LEVIR/>。
- 下文代码/文件名中的`forward`仅是固定输入顺序(A,B)、每对调用一次的历史标识，不表示方向性语义变化目标。单次输入顺序固定用于复现；没有据此声称模型交换输入后必然产生逐像素相同的结果。
- 按官方test的A、B顺序，每个源图块只输入一次(A,B)，不交换顺序，不生成reverse，不做方向OR。
- 首轮直接使用官方demo的`CD_PROMPT`，查询建筑变化区域；原生输出是others/building二值RLE，不要求一次输出多个地物类别。
- 专用核对入口`run_bash/rsunivlm_levircd_forward_queue.bash`只允许LEVIR。默认`FORMAT_RETRIES=0`，只检查原始提示词能否产生有效结果；可显式设置2使用既有固定格式重试，但这属于适配，应与0的结果分开记录。
- 保留用户指定的原生512源图裁块、stride512、右下补黑、完整512预测恢复后裁回有效区域、原图拼接评分。官方SigLIP内部仍处理为384。裁块、双GPU分片和任意正整数batch是VISTAR适配，不称作完整官方benchmark配置复现；论文WHU-CD还提到CoT，这里没有新增CoT。
- 双卡各持有一份模型，按互斥样本分片；每张卡请求batch16，跨源图填批，尾批可以更小，OOM仍按既有策略折半。只保存一份`<sample>_forward.png`二值ID掩码及原文/回执，不保存反向或OR预测。
- GT仍是原始LEVIR二值建筑变化标签；全量累积混淆矩阵后报告building IoU/F1，无GT输入、填背景或选答案步骤。

本次按用户“不要做有方向”的澄清，只修正文档术语，模型提示词、预测、GT、评分、源码哈希和缓存身份均未更改，已交付的fbe00fd离线包仍适用，无需为了这项澄清再次下载。

## 协议与历史结果

根run_config新增`protocol.levircd_query`，明确官方提示词、A/B顺序、forward方向、二值建筑变化、reverse_query=false和direction_union=false。源码和提示词哈希会改变协议身份；专用新目录隔离旧OR运行，不导入旧PNG或旧原文。现有其他数据集输出保留，不为了这个修改重跑它们。

原独立`rsunivlm_levircd.bash`、通用旧启动器和独立Python CLI也将默认CD方向改为forward，防止默认继续OR。独立旧入口仍是整图处理；需要用户此前的512源图裁块和双卡batch时使用本页专用队列入口。旧`symmetric_union`选项只为显式历史复现保留，并非当前核对配置。

LEVIR首次修改时未改SECOND。随后用户另要求SECOND保留六类、采用无方向类别查询，现实现见`SECOND_UNDIRECTED.md`，LEVIR专用入口仍不会运行SECOND。分割集提示词、模型、权重及LEVIR GT/评分保持。更改公用源文件意味着新整个队列代码身份变化，不能据此自动导入旧六集；原结果先保留，待明确各集状态再逐项核对。

## WSL验证与正式完成条件

所有用户命令在唯一`RUN_QUEUE.md`。先运行新增`tests/test_rsunivlm_levircd_forward.py`：检查官方首轮提示、每图块一次(A,B)查询、原生二值掩码、GT在预测之后评分、同协议恢复、旧OR掩码拒绝、两分片合并及未知类别/EOS截断拒绝。本机未执行这些测试。

再跑4对开发核对，使用独立smoke目录。通过条件为所选4对完成、每条raw_text.direction均为forward、images_per_sequence=2，pred_mask只有forward文件。开发子集不是128对正式结果；四对分到两卡时，实际尾批可能小于16，属于预期。

全量命令使用另一个正式目录，MAX_SAMPLES=0。LEVIR的`levircd/metrics.json`必须processed_samples=128、complete=true、is_full_benchmark=true；父队列只请求一个集合，根paper_summary的is_full_benchmark仍为false（该字段保留“七集全部完整”的含义），不能据此把有效LEVIR单集结果误判为缺图。

来源：<https://github.com/xuliu-cyber/RSUniVLM/blob/main/app.py>、<https://arxiv.org/html/2412.05679v2>、当前`baselines/rsunivlm/protocols.py:CD_PROMPT`。本轮没有更改固定上游checkout或公开权重。
