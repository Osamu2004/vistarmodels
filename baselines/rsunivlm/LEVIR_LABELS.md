# LEVIR-CD真实标签读取中断（2026-11-04）

## 已确认的证据

用户WSL回执：rank000在test_18.png、rank001在test_68.png处抛出Expected binary native LEVIR-CD GT。来源为rsunivlm_native_classwise_tile512_bs16_2gpu_v1，LEVIR总128张，已评分64、格式失败1、exhausted_population=False，另63张尚未计入成功/失败；这63张不等于没有生成过预测。

源码tools/rsunivlm_paper_protocol.py的原条件只接受二维数组且每个值属于0、1、255。仅凭异常无法区分RGB/alpha/调色板编码与实际非二值像素；还没有这两份真实文件的像素回执，不能断言数据损坏或断言是RGB。worker.finish在load_targets前已写入完整预测及receipt，已保存且哈希有效的掩码可直接复用重算，无需再次模型推理。

相邻Vistar的gencore/data_provider/levir_cd.py与tools/compute_saved_levircd_opendpr_metrics.py使用convert(L)后>0；模型仓库baselines/levircd_common.py也采用该规则。新增兼容器对精确二值灰度复制编码有相同像素结果，不把任意颜色、半透明或中间灰度猜成二值。

## 最小兼容规则

新增tools/rsunivlm_levir_labels.py：

- 原先合法的二维0/1/255（含模式1和原先已接受的P索引）继续用原>0规则，不重新解释已评分标签。
- 新增RGB必须三通道逐像素完全相等，且只包含0/1/255。
- 新增RGBA、LA必须alpha全255，颜色/亮度仍满足上述二值限制。
- 原P索引不满足旧条件时，允许将调色板显式渲染为RGBA后做相同的精确黑白/不透明校验。旧条件已合法的P不切换到这个分支。
- 不缩放、不改原GT文件，不对127等中间值、红色等彩色类别或透明像素做阈值猜测。原图尺寸必须严格一致。其余数据集仍走原load_targets。

政策和helper源码SHA纳入派生plan的levircd_gt_reader字段。原15个数值源码不改；原目录和缓存身份保留。CPU格式修复入口只要选择了LEVIR就自动声明/启用该政策；若既有_formatfix_v1属于旧工具，按原_p后缀规则选新输出，不混合新旧评分策略。

普通旧主队列命令保持原协议；本次恢复使用更新后的repair_rsunivlm_format.py。它会导入有效掩码/原文、执行原RLE完整校验、在完整预测落盘后读取GT并重新汇总。选--datasets levircd可先只处理这个数据集。完整人口没补齐仍partial；缺少生成回答时CPU工具不调用模型。

显式--run_remaining的GPU派生分支也会通过guarded worker安装同一GT兼容器；source/helper政策不匹配就拒绝执行。GT政策不会改变提示词、A/B顺序、裁块、batch或按GT选择RLE回答。

## 用户在WSL运行

完整命令统一RUN_QUEUE.md。先执行CPU标签审计，读取plan中的全部128份标签，输出每份mode/raw_shape/dtype/值预览/旧条件能否接受/新兼容编码和mask SHA；默认报告写到原目录同级的_levir_label_audit.json，不写入原目录。

只有rejected=0、complete=True才继续CPU重建；否则保留具体失败原文，不能直接阈值修改GT。每个重评分分片额外写gt_reading.jsonl；格式修复与评分报告仍为format_repair_report.json和paper_summary.json。

WSL合成回归覆盖原L/1不变、RGB/RGBA/LA精确解码、P新旧路径、非二值/颜色/透明拒绝、原尺寸、两个分片因RGB GT中断后复用掩码CPU重评分、无模型/CUDA调用、源文件不改、同命令续跑和政策不匹配保护。本机只做AST/Bash/字节哈希静态与Git/ZIP传输校验；不运行测试、GT解码、模型或CUDA。实际文件类型、审计结果和恢复评分数量等待用户WSL回执。
