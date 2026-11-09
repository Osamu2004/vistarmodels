# RCDNet 校园重新推理

2026-11-09已在校园GPU5的RTX3090完成LEVIR-CD test128重新推理及边界评分，两个方案均128/128、0失败、退出0。使用公开SECOND官方权重；论文合成预训练最优权重未取得。

| 方案 | IoU (%) | 3px WFm (%) |
| --- | ---: | ---: |
| 整图缩放512（预先指定主方案） | 40.117765 | 16.586877 |
| 原图512分块（预先指定对照） | 46.035337 | 28.453593 |

分块结果与此前WSL的46.034321/28.452220几乎一致。本次没有明显提高分数。

完整版本、环境、协议、归档SHA和限制见[SUMMARY](SUMMARY.md)。实际运行及失败准备过程见[WORKLOG](WORKLOG.md)，最终状态见[active_execution](active_execution.json)，唯一命令/状态队列见[RUN_QUEUE](../../RUN_QUEUE.md)。原始结果回执在[receipts/RESULT.json](receipts/RESULT.json)。

校园持久产物：/data1/zzy26/outputs/vistar_rcdnet_campus_20261109/evaluations/{whole_resize512,native_tile512}。大掩码归档留在校园，小型JSON已取回。临时缓存9058c567af724d29a7e7c54b13571c2a已清理，GPU已释放。

campus_prepare.py准备CPU依赖/固定官方资产；campus_controller.py逐阶段等待新GPU授权，保留真实PID和退出；authorize_phase.py冻结实际部署源码及扩展SHA，campus_infer.py复用既有数值适配器并记录同forward的512网格输出。runtime_helpers是私有缓存助手副本，lstat计数修复通过20项校园实机基础设施检查，未修改全局工具或其他任务。deployment_manifest.json只记录最初部署历史；本次最终部署身份以receipts/runtime_manifest_frozen.json为准。
