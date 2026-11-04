# LEVIR-CD和SECOND：沿用LoveDA整体语义边界WFm

2026-11-04用户要求补齐两个变化检测集的边界指标，并明确“使用loveda的方式”。本入口在完整预测之上进行CPU后处理，不改已完成推理、标签源文件、旧分数、数值依赖或旧缓存身份。

## 固定协议

统一调用`baselines/binary_boundary_wfm.py:score_semantic_boundary_wfm`和`aggregate_binary_boundary_wfm`，与已完成LoveDA整体语义边界相同：原图尺度标签 → 3×3 Sobel → 一轮3×3膨胀 → Margolin加权F-measure（beta1/sigma5）；GT-ignore及margin2沿用相同实现。逐图统计GT出现的edge/nonedge类，各自取图像均值，再按原IDGBR规则取正均值中较小者。没有对六个变化类别分别求WFm再平均，也不使用全前景二值并集替代SECOND语义类别边界。

- **LEVIR-CD**：复用保存的`forward`二值建筑变化掩码，类别0未变化、1建筑变化；没有反向推理或双方向OR。GT严格使用原source plan记录的`levircd_gt_reader`；当前用户完成回执使用显式`vistar_positive`，即PIL灰度后`>0`，保留已接受的156/254等灰度为前景，不在此处改成新的阈值或重写GT。
- **SECOND**：六类查询按原固定ID1..6（水、裸地、低植被、树、建筑、运动场）依次叠加，后类覆盖前类，未覆盖用0未变化。GT首先保持原六类归属定义：类别在任一原变化标注中出现即属于该类；然后使用与预测相同的固定顺序，得到用于边界的单标签GT视图。若两个时相标注对应同一像素的不同变化类别，则较大的固定类别ID胜出。预测合并不读取GT，不按图像或得分选顺序。

SECOND上述单标签视图**只用于边界评分**，因此保留了类别间的边界。原主指标`mIoU_c`仍使用六个可重叠查询的全数据TP/TN/FP/FN；不修改成单标签mIoU，也不新增有方向语义。这是用户指定的显式边界适配，应随论文说明记录，不能称作作者官方边界协议。

每张记录保存预测/GT未覆盖与重叠数量。工具同时重建LEVIR原混淆矩阵、SECOND六类原计数及主IoU，与原summary核对；主IoU容差1e-12，计数必须精确相同。核对失败或缺失mask时，不能向主scores发布WFm，保存失败说明并返回非零。

## 复用路径和产物

默认原mask来源：

`/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1_formatfix_rle_v1`

默认读取最新七集summary：上述路径后加`_fixedorder_fusion_v1`。新输出再加`_cd_boundary_v1`。同时验证原plan/config、source人口/分片、融合summary身份与依赖、GT reader、数值库版本、每张原二值mask/receipt内容hash。源队列/融合输出必须已退出，源目录只持共享锁。

每个变化集读取原native掩码并重验内容hash；有效逐图评分缓存按独立新策略/代码/来源身份和记录hash续算。默认两个CPU进程、最多4张任务在途，无模型、Qwen、CUDA或新的预测图片/彩色图副本；所有旧掩码和记录保持。每集保存`rows/*.json`、`boundary_report.json`、`metrics.json`，总体保存新版`paper_summary.json`和`paper_summary.csv`。

新summary继承其他五集的全部原分数，只对两个变化集添加WFm；精确保留其原IoU，并继承训练来源星号。CSV百分比两位小数，JSON保留完整原值。SECOND与LEVIR均显示`boundary_metric=wfm_3px`。

## WSL一条入口

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm
SCORE_WORKERS=2 bash run_bash/rsunivlm_change_boundary.bash
```

默认先执行`tests/test_rsunivlm_change_boundary.py`，失败就停止；成功才进行保存掩码的CPU评分。已完成检查后重试可使用`RUN_TESTS=0`跳过WSL回归。路径不同时显式给SOURCE_RUN（原mask/plan）、SUMMARY_RUN（最新融合结果）和OUTPUT_DIR；不要把缺少plan.json的融合结果目录填成SOURCE_RUN。

通过条件：LEVIR128/128、SECOND593/593边界全部成功，原IoU/计数核对通过、无失败、七集summary complete=True。

## 最新WSL完成回执

2026-11-04用户已完成7822e5d：6项回归通过（1.74s），LEVIR128/128、SECOND593/593均失败0；原IoU/计数核对通过，七集complete=True。新增LEVIR WFm为0.1208252242700772（12.08%），SECOND为0.0797504020771827（7.98%）；原IoU27.65%和mIoU_c9.49%保持。记录为用户WSL终端实际回执，不是本机评分或远端原始JSON的独立读取。

最新七集原精度与显示表分别为receipts/wsl_complete_all7_boundary_scores_20261104.json、results/complete_scores_training_marked_20261104.csv/md，保留LoveDA/UAVid/LEVIR的训练来源星号。旧融合回执及其边界空值不覆盖，本次新目录作为当前完整版本；无需重新推理或再次评分。本机只做转录/记录校验，没有执行回归、解码或模型。
