# RSUniVLM原生变化检测接口纠正

2026-11-02。依据用户SECOND失败回执、用户“按照LEVIR处理”和“原本不支持双向OR”的纠正，以及本轮重新读取的官方论文和demo源码。

## Purpose

区分模型原生二值变化查询、Alters/DynamicEarth评分协议和本仓库新增适配。不能把数据预检或任意类别文本输入当作原生支持SECOND六类语义变化的证据。

## Setting

旧七集队列的SECOND分支使用有序图像对、七类一次输出提示（unchanged加六类）、两个交换方向，再按类做并集。这是本仓库新增的评估扩展，并非RSUniVLM官方流程。源图512分块对齐并不能解决任务接口不匹配。

一手来源：

- 官方demo：<https://github.com/xuliu-cyber/RSUniVLM/blob/main/app.py>。示例对一次有序双图输入询问指定类别的变化区域；其可视化解析器将others作为背景，其余名称作为前景。没有交换时相后再OR的步骤。
- 官方arXiv v2：<https://arxiv.org/html/2412.05679v2>。附录10的Change Detection给出建筑变化查询；4.2的公开变化检测结果在WHU-CD上，正文提及CoT辅助评测。该版本没有提供SECOND六类语义变化的验证流程。
- 当前源码：`tools/rsunivlm_paper_protocol.py:prompt/decode`与`tools/rsunivlm_paper_worker.py:Engine.evaluate/finish`是新增多类/双向适配；`baselines/rsunivlm/protocols.py:CD_PROMPT`则保留官方建筑查询。旧独立LEVIR入口已有`--cd_direction forward`可选项，但默认仍为本仓库扩展的symmetric_union；不能称其默认设置为官方单向流程。

本轮网页读取的是官方main和论文v2，未重新读取WSL固定revision的app.py，不伪造固定版本源码哈希。当前已检查的官方公开材料不支持把SECOND适配标为官方原生实现；这也不等于证明模型在任何新提示下都不可能泛化。

## Results

用户WSL七集计划最终`complete=False`。GPU0 SECOND处理1/297后因`00031_forward_y0_x0.json`的格式预算耗尽停止该分片，报告Unknown second category land-cover等；另有一份描述符未EOS截断。实际batch13/12为重试子集，不是OOM证据；日志没有显示本次显存不足。其余六集、GPU1及最终分数仍须读取paper_summary.json，不据该尾部日志猜测。

旧SECOND输出仅为未完成的开发结果，不能填入论文或当作原生六类能力结论。原始文件保留，不删除、填背景、改未知类别或按GT选择答案。

## Analysis

先前把“未变化加六类方向输出”当作RSUniVLM应当支持的接口不成立；把“双向OR”说成遵循官方LEVIR流程也不准确。双向OR在技术上属于外部评估策略，本仓库可以实现，但它不是此次核对到的官方demo步骤。

同样不能直接将SECOND全部类别并成二值前景，再把所得IoU填入论文六类mIoU_c列。两者定义不同。逐类查询加双向OR的临时方案尚未运行，已按用户纠正撤回；没有将其发布为新实现或启动实验。

## Next Steps

1. 不重跑旧七类/双向SECOND全量任务；先保留旧输出并读取完整汇总、两张卡错误及失败原文。
2. LEVIR按官方单向、指定类别、二值输出重新核对，明确空间裁块仍是本仓库适配，不冒充完整官方benchmark复现。
3. SECOND六类语义变化在正式表格中暂不填数；若另做二值变化或新单类查询实验，单独声明任务/GT/指标和协议身份，先小样本验证，不自动转换成六类结果。
4. 其他分割集的类别清单一次输出也属于适配，需按各自运行回执确认有效性；不能因SECOND失败就断言它们全失败或全成功。

本轮只读取公开代码和本地适配源码、保存回执/记录，未连接WSL、未运行模型或测试、未改论文数值，也未修改旧数值源码或其协议哈希。使用experiment-craft流程收集失败、查原生接口并区分已证实问题与待验证原因；该技能未要求用户批准或停止工作，当前不继续全量推理源于用户对接口的纠正和缺少有效原生任务对应。
