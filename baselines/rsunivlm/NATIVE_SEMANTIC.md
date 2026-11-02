# 按原论文公开接口逐类别推理

2026-11-03。替代一次列出全部类别的多类 RLE 适配。默认入口仍为 `run_bash/rsunivlm_paper_queue.bash`，现在 `SEMANTIC_MODE=native_classwise`。模型加载和权重保持；本机不运行测试或模型，WSL 回归与真实前向待执行。

## 公开依据和复现边界

- 原论文附录第10节：https://arxiv.org/html/2412.05679v2#S10 。原文提供按一个类别名称查询的提示词。
- 官方公开代码固定提交：https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/app.py 。`echo` 使用 Qwen1.5 对话模板、像素专家2、FP16、greedy 和 max_new_tokens=32000；`parse_output_string` 将 others 解码为0，其他标签解码为前景，先扩展20×20小块再用 OpenCV 最近邻恢复。
- 对固定提交完整递归树及 `llava/eval/model_vqa.py`、`evaluate_interleave.py`、`llava/train/llava_trainer_eval.py` 做了只读核对。这些公开文件中未找到表7多类分割的融合/评分流程；不能把未找到的多类冲突处理自行写成官方规则。审计回执和源码哈希单独保存。
- 按用户既有要求保留源图512非重叠裁块、右下补黑和双卡每卡16批量。它们是本项目比较用的适配，官方交互演示是单条请求，不声称批量数值逐字等价。
- 继续保留已有像素专家逐步传递修复、meta参数检查、格式边界/EOS审计和协议身份保护；不是未经适配的原样执行 app.py。

## 实际执行

首轮提示固定为 `[SEG] Can you segment the {class name} in this image?`。默认 FORMAT_RETRIES=0，不追加类别清单、强迫24行或自行加格式说明。若用户显式设1/2，失败时使用原文另外两种固定提示，记录实际模板；这是额外的失败恢复策略，不能当作原文已说明的模板投票。

| 数据集 | 每个源图块的查询 |
|---|---|
| LoveDA | 原七类各一次，包含该数据集的background |
| FLAIR | 原十二类各一次 |
| UAVid | 原八类各一次 |
| xBD-pre | building一次，others为其余区域 |
| CHN6-CUG | road一次，others为其余区域 |

每类单独跨样本填满正整数batch，同一个batch使用同一类别提示；双GPU样本互斥分片、OOM折半和模型每卡加载一次保持。所有固定类别均查询，不用GT判断出现哪些类。每条回答保存原文、类别、实际标签计数、批成员、网格尺寸及生成信息。

解码按官方二值规则：只有 `others` 为0，其余合法RLE标签为1。因此即使模型用了其他类别词，也记录该词并按官方前景解释；不是将它语义纠正成请求类别。单独输出 `water` 而没有计数、行宽不一致、非正计数、超界或无EOS截断仍失败；不猜测、补行或把失败掩码置零。常见23×24完整矩形允许按其实际尺寸恢复。

只保存各查询的0/1预测PNG、原文和必要回执。类别之间可以重叠；同一像素可以被多个查询标为前景，也可以都不是前景。不argmax、不按类别顺序覆盖、不用GT选类。新命名为`<sample>_class_00_<name>.png`等，缓存需要全部类别的完整尺寸/值域/哈希一致。旧多类预测和旧失败原文不自动转成本协议；新版同目录可恢复已保存预测或有效类别回答。

## 指标和完成状态

- 多类集合（LoveDA、FLAIR、UAVid）先跨全数据累加每个查询的TP/TN/FP/FN，再按标准类别IoU公式计算 `miou`、`mf1` 和 `per_query`；保留 `query_miou`、`query_mf1` 同值别名便于核对。忽略范围仅来自原GT；有效GT像素预测为others时仍计入该类FN。不平均逐图或逐卡mIoU，不把各类的二值前景/背景均值当作类别IoU。全数据中并集为0的类别记为null并从均值排除，沿用现有空类口径。
- 用户进一步指定要对齐表7的54.46/65.41/49.60。本实现的逐类提示和二值还原有公开依据，但表7实际类别集合、重叠/未覆盖处理和累积脚本未公开，不能仅凭这三个数倒推实现。结果显式标 `metric_scope=independent_class_queries`、`table7_protocol_verified=false`，不称为已精确复现表7的评分协议，也不自动写入论文原表。
- 同时记录 `zero_positive_query_pixels`、`overlapping_query_pixels`，检查未覆盖和重复覆盖。多类结果不输出虚构混淆矩阵或单张语义图WFm，不填入论文原有mIoU/WFm列。
- 二值集合（xBD、CHN6）只有一个前景查询，因此保留原标准building_iou/road_iou、混淆矩阵和边界WFm，附加查询指标便于核查。
- LEVIR维持一次(A,B)建筑变化查询、无reverse/OR；SECOND继续用户指定的六类无方向查询。SECOND是项目适配，不声称为原论文公开验证的六类变化流程。本次未将它改回有方向标签。
- `complete=true`表示所请求样本的推理已完成。多类集合全群体查询完成后为`is_full_query_population=true`，但`standard_metrics_available=false`、`is_full_benchmark=false`；这不是缺图，而是标准多类融合协议尚无公开依据。其他有标准指标的集合仍按原全群体条件判断。根汇总也明确标准指标可用性，避免把查询均值写进原表。

## 执行

更新包和唯一RUN_QUEUE含完整命令。先在WSL运行`tests/test_rsunivlm_native_semantic.py`及现有队列/变化检测回归，再用32张FLAIR开发核对（双卡各16条/类），成功后另起正式全量目录。默认MAX_NEW_TOKENS=32000；实际通常遇EOS提前结束。完整调用次数因逐类查询而增加，不承诺与旧一次多类生成相同耗时。

原多类接口保留为`SEMANTIC_MODE=legacy_categorical`。若复现上一轮FLAIR背景ID12方案，还必须显式设置原MAX_NEW_TOKENS=8192、FORMAT_RETRIES=2，并另用独立目录；旧身份不自动兼容本次代码。不得在仍运行的队列中直接热更新数值源码。
