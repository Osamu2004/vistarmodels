
## 2026-11-09 · 独立保存掩码WFm入口

新增tools/score_rcdnet_boundary.py，复用共享3px WFm，核对原RCDNet输出人口、图名、每图/全局计数与IoU/F1、原图尺寸及源文件SHA后单独写出评分；原推理与数值核心未改。本机仅AST/CLI帮助与包CRC/SHA通过，SciPy不可用，未运行数值或真实评分。环境/命令/产物/停止条件见唯一RUN_QUEUE的RCDNET-SAVED-WFM节。

## 2026-11-09 · Git提交、离线快进同步与线上发布

按用户要求将评分入口、RCDNet说明和本任务队列段落纳入main，使用完整Git bundle交付；安装仅做SHA/目标提交校验及ff-only同步，保留已知数据路径JSON与Git对其他改动的保护，不运行模型或评分。原先的纯源码ZIP仍是历史交付。本次提交与线上核验、离线导入演练的具体commit和SHA以工作区外层output/rcdnet_wfm_20261109/git_delivery/中的回执为准。其他未提交基线改动不纳入此次提交。

## 2026-11-09 · WSL完整边界回执及论文原值对照

用户切换到原rcdnet环境，使用实际旧目录/root/data/experiment/rcdnet_levircd_test_native1024_tile512，128/128、complete=true；IoU46.03432144395981%、WFm28.45222020216903%。此前默认目录和vistar_flux2缺扩展的失败不代表旧预测不存在。全精度终端回执已归档；本机未重新读取WSL源JSON、checkpoint或重算分数。正式原论文Table1真实数据版53.98、合成预训练版60.21均高于本次，WFm是统一追加指标，不能称严格复现。论文表2换为同批实测46.03/28.45，原60.21保留为历史文献值；Alters对当前评估RCDNet的IoU差更新为31.75pp。
