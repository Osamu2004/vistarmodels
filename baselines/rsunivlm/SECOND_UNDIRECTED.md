# SECOND六类无方向语义变化

2026-11-02，按用户纠正实现：**保留类别，不把整个任务二值化；不交换图像做第二遍推理。** 当前仅完成代码与本机静态核对，WSL合成回归、4对真实开发核对、593对正式分数待用户运行。未改论文数值。

## 类别与表示

六类为water、bare land、low vegetation、tree、building、playground，对应原始标注ID1..6。原始0表示未变化，GT读取和RGB映射保持。模型不再被要求一次输出“未变化＋六类”互斥类别图。

按照RSUniVLM原生的类别查询接口，每类各问一次“这两张图之间该类哪些区域发生了变化”，输出该类变化掩码。**每类掩码以0/1编码，但完整任务保留六个类别及六张独立掩码。** 原始提示形式经过无方向语义适配，不冒充官方已发布的SECOND评测流程。

例如某像素由树木变成建筑，其树木变化掩码与建筑变化掩码都应为前景；类别允许重叠，不做argmax、优先级选类或强制压回一张单类别图。掩码键为water、bare_land、low_vegetation、tree、building、playground。

## 预测与评分

- 每个512源图块固定输入(T1,T2)，对六类分别查询，总计6条类别请求；不会输入(T2,T1)，没有reverse预测，也不对两个预测方向OR。
- GT的第c类变化区域为`(label1==c)|(label2==c)`，延续原DynamicEarth六类评价口径。这是从两份已有语义标注确定无方向类别集合，不是再运行一个模型方向。原始两份GT不会先全部合成一个“变化”前景。
- 每类累加全样本TP/TN/FP/FN，维持既有epsilon=1e-7和六类mIoU_c/mF1_c数学。未变化像素在各类评分中作为负例；不把整体二值变化IoU填入六类指标。
- 六类固定全部查询，不根据GT类别是否出现跳过请求或选择回答。未知类别词、其他查询类别、无EOS截断和格式预算耗尽继续拒绝，不把land-cover映射成背景/目标类。
- GT只在该样本全部六张预测已形成并保存之后进入评分；类别集合、提示词和模型输入均不从GT像素构建。

## 调度、裁块与保存

沿用593对Canonical Bi-SRNet/DynamicEarth名单、512无重叠源图块、右下补黑、完整512预测恢复后裁回有效区域、原图拼接。官方SigLIP内部384及模型/权重不改。

每张GPU持有一份模型，样本分片互斥；每类独立维护跨原图的请求缓冲区，支持任意正整数batch，默认请求16。不同类别的提示词不会混成同一文本batch，满批、尾批和OOM折半记录实际值。模型每卡只加载一次供六类复用。

只保存每样本六张必要类别预测ID掩码，以及原始文本、哈希和评分记录；不保存RGB输入、GT或叠加图。原文记录query_key、query_class_id、query_class、temporal_mode=undirected、input_order=[T1,T2]，没有forward/reverse结果。PNG值仅0/1。每样本receipt核验六张掩码完整性、形状、值域与哈希后方可恢复。

新协议元数据`protocol.second_query`明确六类查询、类别允许重叠、每类每块1次、reverse_query=false、prediction_union=false。新源码/提示词改变身份，独立目录不导入旧七类/双向PNG或原文。旧其他集合和失败现场保留，不为本次修改重新跑其余六集。LEVIR行为保持上一版：无方向二值标签、每块一次(A,B)、不做OR。

## 执行与验证

专用入口`run_bash/rsunivlm_second_undirected_queue.bash`只评估SECOND，默认FORMAT_RETRIES=0，先核对原始类别查询；需要既有两次固定格式重试时可显式设2，协议身份随之改变。

命令集中在RUN_QUEUE。先在WSL运行新增`tests/test_rsunivlm_second_undirected.py`及LEVIR/通用队列回归，再用独立smoke4目录检查4对真实样本；确认每对有六类掩码、输入顺序固定、类别未丢失、有效RLE及完整结果后，另起正式目录跑593对。开发结果不是正式基准分数。

正式SECOND单集的metrics必须processed_samples=593、complete=true、is_full_benchmark=true，scores保留miou_c、mf1_c及六类per_class计数/指标。单集父汇总的“七集全量”is_full_benchmark=false仍属预期，不把它误判为缺图，也不声称已完成七集重评。

此实现没有声明模型交换输入后一定给出逐像素相同预测；无方向指任务目标和类别标注，固定输入顺序用于复现。真实有效性、吞吐、显存和精度尚待WSL回执。
