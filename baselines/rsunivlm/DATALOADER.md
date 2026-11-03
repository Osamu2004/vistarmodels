# RSUniVLM 有序 DataLoader 与原目录续跑

## 实现与参考

参考相邻Vistar仓库的验证加载方式：

- vistar/gencore/data_provider/loveda_unigen.py:141与second_unigen.py:130：验证集shuffle=False、drop_last=False，worker数量可配置。
- vistar/apps/data_provider/base.py:153的验证分支使用prefetch_factor=2、pin_memory=False；本队列沿用这些I/O选择，不沿用基类的shuffle=True。
- 本队列每个数据集只顺序遍历一次，因此persistent_workers=False，并在完整或异常退出时关闭该数据集的worker，不为七个集合长期保留七份进程池。

新增tools/rsunivlm_data_loading.py，接入已发布的guarded监督入口。每卡默认2个CPU worker，每worker预取2个源样本，共4个待加载源样本；双卡共4个CPU worker。DataLoader关闭自动组批，每次返回一张原图或一对时相原图。**GPU模型仍按原逻辑、按类别累积最多16条裁块查询**，I/O单元不是模型batch。

DataLoader以spawn启动独立CPU进程，不fork已初始化CUDA的模型进程；worker不加载模型、不调用CUDA，返回CPU uint8张量用于共享内存传输，再原样还原PIL RGB。pin_memory=False与当前返回PIL给官方processor的路径一致。使用独立CPU Generator，避免DataLoader初始化消耗模型进程的默认torch RNG。

## 为什么能接着原目录跑

本次不修改原15个数值身份源码，读取层只在监督入口中替换evaluate模块的RGB读取引用，真实protocol模块保留原解码函数供worker使用。运行时加载器选项与代码SHA单独落盘，不能声称已做真实GPU逐字等价验证。

- 原七集人口、max_samples选择和全局index modulo world_size分片不变，不使用会补齐样本的DistributedSampler。
- shuffle=False、FIFO返回、drop_last=False；即使后面的图片先解码完成，也等前面的图片，严格保持原类别/裁块查询顺序和batch成员。
- 继续调用原protocol.load_images，包括FLAIR五波段文件取原RGB三波段的方法；不增加缩放、增强、归一化或随机变换。模型的官方processor不动。
- 不提前读取GT像素。GT仍在该图完整预测落盘后由原评分代码读取；worker只按原协议核对文件元数据。
- 掩码receipt、像素SHA、原文identity、旧attempt预算和所有缓存判断仍由原evaluate执行。成功掩码跳过模型，部分原文重建预测，原零重试预算已耗尽的坏原文仍保留失败；不会借DataLoader隐式补跑或填背景。
- 原黑色右/下补边、512非重叠源裁块、时相图顺序、每类独立查询、解码、原分辨率还原与计分均不改。
- 继续使用原OUTPUT_DIR。原协议不变时无需_p新目录，也不移走、删除或重新标识原预测。

DataLoader会像旧实现一样读取成功缓存图的RGB，用于原图尺寸/完整性和后续原评分；模型生成不会因此重复。读取或元数据异常不会跳过、替换或重排样本，原数据集仍标记不完整。

## 设置与观测

| Bash变量 | 默认值 | 含义 |
| --- | --- | --- |
| NUM_WORKERS | 2 | 每张GPU对应的CPU读取进程数；0为单进程DataLoader |
| PREFETCH_FACTOR | 2 | 每个CPU worker预取的源样本数；NUM_WORKERS=0时不传给PyTorch |
| LOADER_TIMEOUT | 120 | 多worker读取等待超时秒数；0为不设置等待上限 |
| BATCH_SIZE | 16 | 每张GPU原有的模型查询batch上限，仍保留普通OOM减半 |

每卡、每集生成shards/rankNNN/data_loading.json：

- consumed_sources / assigned_sources：原图或图对计数，不是裁块/查询数。
- rgb_wait_seconds：模型进程等待DataLoader的时间，含首次spawn启动。
- consumed_worker_decode_seconds：已消费样本在CPU worker的解码时间之和，各worker会重叠，不能当作主进程墙钟时间或直接相减计算加速比。
- options、spawn、排序设置、loader源码SHA、原运行identity与清理状态。

只为数据I/O重叠提供实现，不宣称所有慢都来自读图或已有提速实测；自回归生成和CPU评分仍可能占主要时间。之前的CUDA_LAUNCH_BLOCKING=1是定位错误的同步调试选项，会限制异步执行；正常速度命令显式用0，且运行回执记录该值。若再次定位CUDA非法访问，可临时改1。

## 验证状态

tests/test_rsunivlm_data_loading.py为WSL合成回归，覆盖单/双时相、五波段FLAIR RGB、有序spawn、两卡分片、父进程RNG、七集输入批次/掩码/计数对照、完整缓存复用、部分原文续跑、预算耗尽格式失败、输入变更、未来坏图片不影响前面顺序、异常清理和空分片。使用CPU假模型，不加载权重或执行CUDA前向。

遵守用户要求，本机仅AST/Bash/JSON/数值源码哈希静态检查，不运行测试/模型。新增回归、真实吞吐与GPU恢复仍待用户WSL回执；完整命令和停止条件统一RUN_QUEUE.md。

官方说明：https://docs.pytorch.org/docs/2.11/data.html （FIFO、spawn、预取和专用generator）。
