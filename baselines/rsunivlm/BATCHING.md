# LoveDA v5：裁块合批与 bareland 恢复

最新推荐入口为[PAPER_QUEUE.md](PAPER_QUEUE.md)的论文七集双GPU队列，`BATCH_SIZE`接受任意正整数，跨图流式填批；每GPU一份模型复用全队列。下文是旧LoveDA单卡入口的版本历史；旧入口也已解除batch参数固定枚举，新的队列不导入单图缓存。

2026-11-02用户回传：v4全量在163/1669后因`Val_Rural_2685_tile512_y0_x0_seg.json`中`bareland`未知标签中断。已完成的九图及部分全量记录不作为完整基准分数。v5已准备代码；本机只做静态检查，WSL回归、实际吞吐、峰值显存和最终分数尚待回执。执行命令统一放在仓库`RUN_QUEUE.md`最上方RSUniVLM段。

## 改动范围

- 显式添加`bareland`、`bare land`到LoveDA `barren`的同义映射。原有标准化使`bare_land`和`bare-land`也等价。没有模糊匹配；`class`、`remote-sensing`等未知词仍报错。缺少计数的单词`water`仍不能当作整块水体。
- 每张1024原图的四个512源图裁块作为四条独立序列一起推理；每条序列恰好一张RGB和同一提示词，不是将四图拼成一个问题。默认请求`TILE_BATCH_SIZE=4`，可选1/2/4；格式失败的子集按相同提示变体合批重试。
- 保留官方384视觉处理、固定expert2、greedy、float16、eager attention、KV cache、8192新token上限和至多两次既有固定格式重试。正常结束的原生矩形按最近邻还原各512块再拼回原尺寸。每序列按其首个EOS判断结束，不把其他序列继续生成时的填充计入有效文本。
- CUDA OOM时按4→2→1拆分并重试原请求；batch1仍超显存则停止。不跳过样本，不填背景。实际batch写入每条原文的`generation.inference_batch_size`。
- `SAVE_IMAGES=0`只保存预测掩码和必要文本/评分回执，不生成输入、GT、彩色掩码和叠加图副本。mIoU、逐类指标与3px边界WFm的原评分实现不改。

## 旧结果怎么复用

若命令中的v5目标已有不同协议，新入口打印差异字段、保留该目录，自动选择其同级`<原目录名>_p<当前协议哈希前12位>`，并在新目录写入`output_resolution.json`。再次使用同一命令和协议会选中同一目录；哈希目录本身若也冲突则停止，不覆盖、不删除。终端`Actual OUTPUT_DIR=`是实际产物位置。此次用户报错只证明v5身份不同，未提供其run_config，具体差异仍待该诊断输出；不能据此认定某个参数或权重已变化。

显式设置`IMPORT_V4_ROOT`，新输出为独立`..._rle_v5_bs4`；不覆盖v4目录。导入器核对旧`run_config.json`、完整输入清单、checkpoint身份、适配器源码、提示词及各次重试、裁块几何。旧运行仍持有写锁时拒绝导入。身份机制沿用原版：小型模型元数据及适配器文本有哈希，大权重/输入图像使用路径、大小和mtime；这不是对大文件载荷新增的完整加密证明。

通过后仅复制`raw_text`，不复制旧预测PNG、混淆矩阵或分数。新协议按扩展别名重新解析最早合法的已存回答，重建掩码并用原评分函数重新计算。原先已重试到后续答案的裁块，若更早答案只有`bareland`问题，会采用这个更早答案，所以最终分数可能与旧部分结果不同。全部原回答和实际尝试次数保留，另存`selected_prompt_variant`与`cache_import`来源/SHA256；旧文件保持原样。新的来源回执是`cache_import.json`。

导入成功时，之前163张及失败原图已保存的裁块可直接从原文重建，避免再次模型生成；这些图片仍需进行CPU评分。旧协议或数据身份不同则停止，不冒充缓存匹配。未保存的裁块才调用模型。若已有三次回答在别名扩展后仍全部非法，也会停止；不无限重试。

## 批量一致性与耗时

用户要求实际增大batch时，当前入口可用`TILE_BATCH_SIZE=4 VERIFY_BATCH=0`直接批量生成，不调用单图核验，因此不会因核验差异回退。此项会写入协议身份，不继承旧核验开启目录的serial_fallback。已经观察到批量与单图差异，二者不声明等价；当前RUN_QUEUE使用独立`..._v5_bs4_direct`目录、空IMPORT_V4_ROOT，从头跑完整1669张，不混入单图缓存。第一次每图4裁块合批；格式重试子集可以1–3，OOM仍按既有规则拆批。现有代码包已支持，新的真实速度/显存/指标仍待WSL回执。

默认`VERIFY_BATCH=1`。每种实际batch大小首次出现时，将同一RGB、同一提示的合批结果与原单图调用比较：合法输出要求恢复后的类别掩码完全相同；两边均非法时要求原文和EOS/截断标记相同。另核对首次合批的图像预处理是否与逐图完全相同。比较不读取GT。结果保存在`batch_verification.json`。它是首批运行检查，不能证明后续所有样本逐像素等价。

用户v5回执在163张后首次核验出现差异，旧版本因此中断；具体两份原文未提供，不能认定差异成因。当前`batch_mismatch_policy=use_serial_references_and_force_batch1`：差异出现时，对当前整组统一使用已经计算出的原单图参考，不按GT、分数或合法性逐块择优，也不再次生成这组单图参考；后续所有查询固定batch1。同协议重启读取该回退记录，不再尝试已失败的合批。合批/单图原文和case对应裁块路径/身份全部保留。参考答案若非法仍进入既有格式重试，不能补造掩码。最终metrics记录serial_fallback及effective_tile_batch_size，实际回退1就不能宣称batch4加速。

核验是额外控制调用，独立记录于`batch_verification.json`，不占逻辑查询的格式重试预算；`format_audit`统计正式选用路径的格式尝试，不能把它当全部GPU调用次数。新增回退策略和源码哈希参与协议身份；升级后自动选独立哈希目录，从v4原文重建旧163张并重新评分，旧v5核验原文仍保留。

首次权重加载及上述验证有额外开销，应看后续稳定图片。终端逐图打印`generate`和`verify`时间；`predictions.jsonl`包含`tile_generation_seconds`、`batch_verification_seconds`，`scoring_timing.jsonl`单独记录评分时间。`generate`包含缓存解析、模型加载、推理及首批验证，是墙钟时间，不是纯GPU kernel计时。已有163张重评分期间的吞吐不能用来估计新样本推理速度。不能预先保证倍数提速；批量4的自回归解码也会等待其中最长序列。

## 源码与证据边界

新入口及辅助代码在`tools/eval_rsunivlm_batched.py`、`tools/rsunivlm_batch_runtime.py`、`tools/rsunivlm_batch_import.py`。旧`baselines/rsunivlm/*.py`保持字节不变，旧v4/LEVIR默认入口继续保留；设置合批或旧原文导入时才选择v5入口。新协议版本4把别名、batch、验证设置及扩展源码哈希写入身份。

依据固定官方提交`66dce1dc68b81a6c62f02fb8e5b306571092a864`核对独立序列、图像与modalities对应关系及GMoE专家选择：[多模态输入](https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/llava/model/llava_arch.py)、[GMoE解码](https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/llava/model/language_model/llava_qwen_gmoe.py)。本机新增的WSL合成回归覆盖别名/未知词拒绝、合批对应、失败子集重试、缓存导入身份、无模型恢复、OOM拆分、独立EOS、批量比较和预算耗尽；尚未执行。

此前大量`copying ... to a meta parameter ... no-op`警告发生在官方加载阶段，不是每个裁块都重新载权重。嵌套SigLIP加载可能在外层空权重上下文触发警告；主checkpoint头部包含视觉编码器层（包括用户日志的layer15）权重键。终端`no remaining meta parameters`说明最后已物化，但不等于已核对WSL每个张量与checkpoint数值相等。本次不改加载器，也不把所有编码器警告解释为被删除的pooling head；视觉权重数值审计仍是独立事项。
