# 约束RLE生成恢复策略（2026-11-04）

用户在了解自由输出/有限重试不能保证合法格式后，明确要求添加RLE解码。本次是“生成阶段约束RLE”，不是把缺数、缺行或破损词汇事后猜成掩码。策略显式启用--constrained_rle，独立保存为原目录同级_formatfix_rle_v1；保留原始预测、原文、真实历史和失败记录。

## 固定约束与适用范围

- 当前七集的native_classwise语义、LEVIR单向建筑变化、SECOND六个无方向类别查询均采用二值类别查询。每个RLE只允许others和该固定查询的前景类，GT不参与类别/计数选择。
- 新兜底声明固定24×24网格，可显式设置RLE_ROWS/RLE_COLS（1..64）。每行正整数计数之和必须等于列数，行数固定；完整网格后才允许EOS。不补背景、计数、行或伪造EOS。原先合法的23×24等自由输出仍按原协议复用，不被重绘。
- 这是解码方式/兜底网格的新策略，不声称与原论文自由输出完全相同；model weights、512源裁块、384内部视觉处理、原query文本、A/B顺序、掩码还原和指标函数保留。新policy/源码SHA/网格与复用来源写入identity，不能混用原基线身份。
- 前两次仍自由输出/机械修复；最后一次既有重试使用约束生成。旧自由历史已经用完三次的查询，再允许一次明确标记的约束回答；最多四条真实回答。已经做过一次约束但仍异常的历史不无限重复。

## 实现依据与验收

采用Transformers 4.45.2官方logits_processor接口。核对固定RSUniVLM上游GMoE generate：images转换为inputs_embeds后，其他kwargs传给super().generate；该版本Transformers在inputs_embeds路径使用长度0的生成input_ids。内置PrefixConstrainedLogitsProcessor对长度0执行view(-1, beams, 0)不兼容，因此使用独立的逐行RLELogitsProcessor等价地屏蔽非法token，不添加BOS或盲目按提示词长度裁剪。任何不兼容入口立即报错。

参考：

- https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/text_generation
- https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/generation/utils.py
- https://raw.githubusercontent.com/xuliu-cyber/RSUniVLM/66dce1dc68b81a6c62f02fb8e5b306571092a864/llava/model/language_model/llava_qwen_gmoe.py

有限token trie只允许完整“类*正整数+合法分隔符”片段，任何选择都减少本行剩余格数或推进有限前缀；所以总run数不超过rows×cols。逐片段验证本地tokenizer精确往返，计算完整网格最坏token上界并检查原max_new_tokens预算足够。每批callback状态独立，支持已完成成员的EOS padding。模型仍依据自身logits选择合法类别与计数；没有预先指定预测内容。

生成前只临时向同一core.model.generate注入prefix约束、EOS和与该语法兼容的生成限制，结束/异常后恢复原method。单图与多图仍调用原generate_requests/GMoE像素专家路径。所有约束回答保留真实generation标志，并经原decoder后追加固定网格、允许类别及真实EOS检查；任何异常都保留失败，不能把可解析但尺寸不符的输出当成功。

## 旧失败查询与续跑

从CPU预检分片的失败名单中找出已耗尽自由预算、没有做过约束的查询，按相同类别prompt组批，最多原batch/GPU=16；只读取这些样本的RGB/图对，原坐标/黑色padding。保存新增真实约束回答后再进入原完整预测/评分循环，避免将已合法查询重跑。未耗尽的查询继续原循环，最后retry才启用约束。

用--reuse_run导入旧_formatfix_once_v2或_formatfix_v1_retry2，先复用旧成功掩码/回答，再由原source填空；原始源和派生源都必须已退出。含约束结果的来源只能在显式保留同policy/网格的情况下导入；验证四条历史中最多一条约束回答和每条prompt/variant，不把旧次数清零。同一命令可恢复新RLE目录，不要把新目录作为source_run。

原15个数值源码文件保持字节不变；通过独立runtime包装与只对tagged回答的附加验收接入。默认非RLE入口不启用这些约束，既有三次预算不变。

## WSL运行与产物

入口run_bash/rsunivlm_recover_rle.bash：合成回归→本地Qwen tokenizer全查询预检（不加载模型权重/GT）→LEVIR正值GT审计→导入旧结果→修复/补推理/约束兜底→终检/恢复汇总。完整命令见RUN_QUEUE顶部；旧任务运行中不要更新其源码。

默认优先复用没有_p后缀的_once_v2，若不存在则复用_retry2；实际目录不同要显式REUSE_RUN，不凭猜测覆盖。默认RLE_ROWS/RLE_COLS为24，变更参数会产生新身份；SRC/REUSE默认不选择本次新RLE目录本身，所以同命令续跑可保持稳定来源。

每个约束attempt.generation.rle_constraints记录policy/网格/类别/helper SHA，旧超预算兜底的prompt_variant为3并有exhausted_history_extra=True；新的最后retry仍为variant2。constrained_rle_rankNNN.json记录每卡实际生成/旧历史额外回答数，format_repair_report.constrained_rle_workers与recovery_counts汇总，paper_summary.complete给出选定人口完整性。

已成功评分且实际使用约束回答的整图会在各分片constrained_samples.json单列，预测receipt附constrained_rle_queries及数量；复用旧约束掩码时这些证据随receipt保留。format_repair_report.constrained_samples汇总逐集整图数，与查询级generated区分。paper_summary显式记录constrained_rle_policy和official_original_decoding_protocol=False，避免将混合恢复结果当作原自由解码的同一版本。

WSL回归覆盖随机合法语法路径、CPU HF logits processor屏蔽非法高分token、EOS/批成员/预算/tokenizer往返、实际generate方法hook/恢复、第四条真实历史、旧耗尽历史只补一次、最后既有retry切换约束、错误网格/外来类/EOS拒绝与不补掩码。当前只在Mac做AST/Bash/JSON/源码字节检查，没有运行这些测试、tokenizer或模型/CUDA；实际WSL格式覆盖和分割精度均未验证。

语法约束不能保证分割准确，也不能保证驱动/显存/进程永不出错。只有20,196张全部成功评分、failed=0且paper_summary.complete=True，才能报告本次恢复覆盖完整；不提前承诺全部成功。
