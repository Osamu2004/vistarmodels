# RSUniVLM 双卡故障代码审查（2026-11-03）

## 结论与边界

已确认一个会扩大故障的控制流错误：原 worker 在 CUDA illegal memory access 后继续复用同一个 Engine 跑后续数据集。已收到的回执与该缺陷吻合。它能解释后续集合全部立即失败，不能独立解释首次非法地址，更不能据此认定第二张卡物理损坏。

用户最新明确报告：Windows PowerShell 和 WSL 的 nvidia-smi 都只显示一张卡。因此问题已不限于 WSL 内 CUDA_VISIBLE_DEVICES 的可见性或某个模型进程。尚无当前 Windows PnP 错误码、nvlddmkm/Display/WHEA 事件或首次出错的同步模型栈；不能把宿主驱动异常、TDR、PCIe/供电/硬件问题或工作负载触发驱动缺陷中的任何一项写成确定根因。也可能设备/驱动先异常，随后模型才报告非法访问。

检查的是本地适配器及固定官方66dce1d的相关源文件；未连接用户WSL，不声称已核对实际Windows驱动/WSL版本或服务器所有本地源码修改。没有运行本机测试/模型、WSL检查或GPU负载。

## 证据与源码发现

| 项目 | 核对结果 | 能支持的判断 |
| --- | --- | --- |
| 原任务失败 | GPU0 FLAIR 5696/7850、GPU1 6688/7850时报非法地址；同进程后续集合立即同样失败 | 首次异常以后有上下文失效的连锁传播；最初算子未确定 |
| 原worker异常处理 | tools/eval_rsunivlm_paper_queue.py:171-175捕获所有Exception后继续，Engine在循环外构造 | 已确认缺陷，应遇CUDA致命错误退出进程 |
| 启动崩溃 | 两worker exit_codes=[-11,-11]；单独诊断在torch.cuda.is_available SIGSEGV，导入已通过、权重未加载 | 当前启动故障发生于设备检查，不是本次数据解码/权重前向；初次非法访问原因仍未知 |
| 双卡绑定 | 父进程为每个child分别设置CUDA_VISIBLE_DEVICES；模型指定cuda:0和固定device_map | 每个进程的逻辑0对应其独立物理卡，未发现两模型误加载同一物理GPU0的代码路径 |
| GPU状态操作 | 审查适配器、队列、启动脚本，未发现GPU reset、设置功耗/频率、超频或设备禁用命令；无适配器自写CUDA kernel | 没有发现直接关闭/重配置宿主显卡的指令；仍不能排除框架/驱动kernel缺陷 |
| 批量与缓存 | B=16、use_cache=True、上限32000；仅在generate返回后记录输出长度，GPU输出转CPU，不跨批保存GPU输出 | 未发现Python层持续累积每批GPU输出；32k长尾会增加整批KV cache与运行时间，实际峰值未知 |
| 官方对照 | 官方app.py同样max_new_tokens=32000，但公开示例一次处理一个会话；本适配器为多会话批量 | 32k不是擅自发明的设置；B16的峰值/稳定性不能由单会话示例保证 |
| 输出预算 | 适配器检查文本提示长度，没有给多模态提示长度+生成预算做联合上限检查 | 存在预算保护缺口，不等于已证明越界或造成掉卡；不静默减少tokens混入旧协议 |
| OOM处理 | Engine.forward只捕获torch.cuda.OutOfMemoryError，离开except后gc/empty_cache并减半batch | 这是普通OOM恢复，不会修复illegal address；未收到此轮普通OOM的日志证据 |
| 父进程管理 | 原launch等待其他worker自然结束，对-11/致命CUDA失败没有立即停止同一队列的peer；日志按GPU追加且没有启动边界 | 崩溃信息不清、故障后继续消耗设备，且旧日志可能误认新进度 |

本机已下载checkpoint配置（不等于本轮核对WSL配置）为24层、hidden896、14个attention heads、2个KV heads、FP16。按2×layers×KVheads×head_dim×bytes×batch×tokens估算，B16、32k仅KV cache约5.86GiB，另有提示缓存、模型、视觉编码、logits/attention临时张量和allocator开销。不能从32k这个数字推导一定OOM，更不能从未收到的峰值证明供电/硬件问题。

官方GMoE与多模态拼接核对：固定granularity=2在三专家范围内；批量按request展平图像、每会话放一/两个图像token，modalities长度为会话数，与上游按会话zip的拼接路径一致。未发现可直接定位初次非法地址的Python索引缺陷；仍需真实故障batch及同步栈，不能把静态审查当CUDA正确性证明。

## 已实施的保护：独立controller，保留数值源码身份

新增tools/eval_rsunivlm_guarded.py；Bash启动入口改为该controller，再调用原规划/worker/评分实现。原15个数值身份源码保持字节不变，数据/提示/模型/生成参数/分片/掩码和指标定义不改；controller记录独立SHA256，清楚区分监督逻辑与数值协议。新保护不是对宿主掉卡的已验证修复。

- 启动前用有10秒超时的只读nvidia-smi查询核对所选index/UUID/PCI地址；缺卡、UUID别名重复或查询失败均阻止模型进程启动。不会将双卡任务静默改为单卡。
- 在每个dispatch前保存CPU元数据，含原文路径、query身份、batch及token预算。没有GPU同步/额外前向；正在执行的组可能包含原OOM分拆，未伪称记录精确故障token。
- Engine.evaluate遇非法地址、device-side assert等已知致命CUDA异常时，先保留原分片partial记录，再退出worker；不继续另一个数据集、不重试坏上下文。
- 子进程从启动时开启faulthandler；bootstrap打印阶段，独立启动ID的日志保留，不再只能看到旧追加尾部。
- 父进程看到信号退出或保护退出码即对本次调用所创建的peer进程组依次发出有界SIGINT/TERM/KILL；不触碰其他训练或用户进程；不自动重启GPU/WSL。驱动故障可能令进程不可中断，仍存活的本人PID会单独记录，不冒充已经终止，也不无限等日志管道关闭。
- 致命退出不重新汇总陈旧分片冒充本轮结果；原paper_summary保留，新guarded_runtime_finished.json明确complete=false与本次退出码。普通格式失败仍按原约定记失败并遍历其余样本。

新增tests/test_rsunivlm_guarded.py覆盖缺卡不启动、UUID重复、非法地址跳出旧Exception处理、普通格式/OOM不误判、对象透传及peer停止。遵照用户规定，本机只做AST/Bash/差异与哈希静态检查，不运行测试；WSL测试与真实恢复仍待设备恢复后的回执。用户随后要求提交推送，发布状态以Git回执为准，未在用户机器安装或验证。

## 当前执行顺序

当前两平台缺卡时先不再发起GPU负载。先在Windows PowerShell保存Display/PnP状态与最近24小时的nvlddmkm/Display/WHEA/Kernel-PnP事件（命令统一在RUN_QUEUE.md），再判断恢复路径。WSL重启只能重建WSL状态，不能据此保证修复Windows侧设备异常。采集后、工作已保存时可以正常重启Windows并重新核对设备；持续缺卡交由宿主驱动/硬件维护，不能靠继续跑模型判断。

只有Windows/WSL均重新识别两张卡、轻量CUDA检查通过，才验证新增监督代码；不能在缺卡状态强行跳过检测或改卡数冒充原双卡协议。保持当前评分产物，正式全量分数仍不完整。

## 一手来源

- CUDA非法地址/断言会破坏当前CUDA进程，需退出重启：https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html
- NVIDIA Xid目录区分应用/驱动/硬件原因，Xid79需有实际日志方能判定：https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
- Windows TDR会重置GPU并记录事件；长生成总时长不等于单个不可抢占任务超时：https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
- 官方单会话示例：https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/app.py
- 官方GMoE实现：https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/llava/model/language_model/llava_qwen_gmoe.py
- 官方图像嵌入拼接：https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/llava/model/llava_arch.py
