# OVCD适配工作记录

## 2026-11-08：缺环境与入口的实际交付恢复

WSL用户回执表明`vistar_dynamicearth`不存在，且OVCD Bash未导入。此前只给本地待交付路径不构成可运行命令，不能从本地文件存在推断WSL状态。此次沿原隔离OVCD分支补环境准备及ready队列，不改原数值实现、SAM/DINO/CLIP参数、数据人口、tile512或IoU/WFm函数。

准备两个独立环境（默认从vistar_longcat克隆），在安装前以distribution metadata封存Torch/torchvision/可选torchaudio，向嵌套pip传播精确constraints，安装后逐项核对；复用已存在目标环境与完整公开权重。动态依赖源码构建显式关闭build isolation以使用当前CUDA Torch，未声称已在2.11/cu130上编译成功。单入口按setup、全人口plan、每方法/集2例真实CUDA、正式128/593执行，失败停止。离线安装器只更新vistarmodels，不触碰已完成FLUX2的vistar源码或结果，不安装依赖或启动模型；第一次环境/上游源码/权重准备需要联网。

本机不运行pytest/Torch/模型或下载权重；AST、Bash、标准库元数据守卫隔离检查及Git导入/冲突/篡改回放写入交付回执。环境安装、扩展加载和实际CUDA指标待用户WSL回执。

## 2026-11-05：统一检测人口并补齐两个公开方法

用户要求检查SECOND所有划分、LongCat统一Alters593并补WFm，同时把两个OVCD方法实现进vistarmodels。已核对原稿和代码：变化生成Test1694/双向3388、变化检测canonical val593；LongCat旧自动评估和20k队列误用Test检测。训练暴露与FDr⁶真实参考属于不同证据层，未修改已有分数或模板。

本节点只负责DynamicEarth/OpenDPR的LEVIR/SECOND直接检测。已有DynamicEarth只支持LEVIR整图；OpenDPR此前无接入。本次读取两个官方固定commit源码和OpenDPR提供的支持集接口，逐类提取DynamicEarth SECOND参数，新增共享数据/掩码恢复/评分前端及独立后端、bootstrap、依赖检查和双环境双卡队列。原有模型/RSUniVLM数值源码保持；新增保存预测掩码，无输入/GT图片副本。

静态审查重点：不能把DynamicEarth每类参数抹平；不能缺OpenDPR原型仍打分；不能从来源不同的`dynamic_earth`包混合导入；SECOND必须保留六类成员IoU而只在边界使用固定顺序；不能因文件数相同冒称同一593清单。LEVIR tiling显式记为适配，原整图入口仍保留。

研究工作树只读审计提示旧仓库根文件多、旧DynamicEarth节点没有语义manifest；这些是已有源代码组织，不据此移动/清理文件。新节点用README/WORKLOG区分代码、状态与真实结果，不建立伪benchmark结果目录。用户此前要求本机不测试，因此只执行AST/Bash/JSON/官方源码接口核对；新增CPU合同测试留在WSL，未启动Torch/模型/下载权重/训练/评分。下一步以用户WSL导入检查及2对真实前向完成运行验证，再跑128/593完整指标。


## 2026-11-08：正式纳入vistarmodels/main并通过Git离线同步

用户要求目标为vistarmodels，采用Git离线同步。本次将已审查OVCD树按范围纳入主仓库main，并保留此前OVCD交付历史祖先，以兼容旧main/b11/aa5恢复包的fast-forward。根README/RUN_QUEUE仅按OVCD段落分块提交；InstructPix2Pix、SyntheticGen、RCDGen等其他工作区改动保持原样。新的直接Git bundle入口和用户命令见根RUN_QUEUE的OVCD-GIT-OFFLINE-SYNC。

本次仅Git集成、AST/Bash、标准库导入验证及本地隔离仓库的bundle重放；没有新数值实现、WSL环境安装、CUDA模型前向或指标。代码离线同步与后续环境/权重联网准备分开，WSL是否完成仍需实际回执。

## 统一DINOv2 M-C-I（2026-11-09）

用户纠正按数据集选择最高配置的方案，要求两集统一一个模型。新增显式DYNAMIC_COMPARATOR=DINOv2，使用同一SAM-H/DINOv2-B14/SegEarth-OV＋SimFeatUp；仅沿用类别相关官方词组/阈值。ground在DINOv2官方脚本为145，与DINO的135不同；配置脚本SHA均已本机静态核对。保持LEVIR四512块拼回、SECOND原512和历史评分。原DINO/OpenDPR结果及数值实现保留；新结果必须独立完成真实2例、全量128/593与归档审计，运行状态见唯一RUN_QUEUE。
