# FLAIR 的 others 按背景处理

2026-11-03，按用户明确要求：FLAIR 输出的 `others`、`other`、`background` 统一解析为预测背景 ID **12**。原有十二类仍为 0..11，building 保持 0；不会把 building 当背景。名称按现有规则去除首尾空格、忽略大小写。

这是 VISTAR 适配器的用户指定解释，不声明为官方 FLAIR 原生类别或官方 RSUniVLM 评测结果。提示词、固定格式重试、权重、512 源图裁块、每卡 batch16 及双卡分片均未改变。其他数据集已有的 others/background 映射不变；其他未知词、损坏 RLE 和无 EOS 截断仍按原规则失败。

## 保存和计分

- FLAIR 预测 PNG 值域扩为 0..12，12 专门表示背景；255 仍只用于原有 GT 忽略标记。
- GT 继续 raw1..12→0..11，其他原有忽略标签不变。mIoU/mF1 仍对十二个评分类别求均值，不另加第十三个背景类别的指标。
- 混淆矩阵为 **12 行 × 13 列**：前十二列为原类别，末列为预测背景。有效 GT 像素上预测背景保留在该 GT 类的 FN 中，不丢弃这些像素，也不根据 GT 改选背景所属类别。
- 指标新增 `background_pixels_on_valid_gt` 和 `background_rate_on_valid_gt`；`valid_pixels` 包含这些像素。背景不使用通用指标中的 rejected/ignore 命名。
- 边界 WFm 把 12 作为实际预测标签；背景和前景之间的边界参与评分，仍只根据原 GT 的 ignore 区域排除支持。
- 缓存值域检查、逐卡累加和根目录合并均支持该背景列。协议中显式记录 `prediction_background`，旧源码身份自动隔离。

## 更新与执行

当前只有本机 AST、源代码差异和离线传输完整性核查；不在本机运行测试或模型。WSL 回归与真实预测待用户执行，尚无新的 FLAIR 分数。

只重跑 FLAIR，保留其他六集现有输出。使用独立目录 `rsunivlm_flair_background_tile512_bs16_2gpu_v1`，不要覆盖旧失败目录。旧目录的 `others` 失败记录保留，此次不自动跨协议导入原文；同一新版目录中断后支持原有的掩码/原文续跑。运行命令统一放在仓库 `RUN_QUEUE.md` 最新小节。
