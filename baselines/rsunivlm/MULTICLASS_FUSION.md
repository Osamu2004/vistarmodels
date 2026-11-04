# 逐类二值掩码合并与标准分割指标

## 本次设置与来源

2026-11-04用户要求参考Text2Seg进行多类合并，在明确其公开资料没有完整跨类别冲突算法之后，选择“按固定类别顺序合并”。本工具只处理LoveDA、FLAIR、UAVid，新增后处理，不重新推理或训练。

核对了[Text2Seg论文](https://arxiv.org/abs/2304.10597)第3节、4.3节和[作者代码](https://github.com/zhangjielu321/Text2Seg/blob/dbf6d3df026b5865517defec56decd47ce95226b/text2seg/text2seg.py)，固定上游提交`dbf6d3df026b5865517defec56decd47ce95226b`。公开实现提供文本指定类别的掩码；没有发现完整互斥多类评测中的跨类重叠、未覆盖像素归属规则。根目录utils.py的面积排序用于SAM可视化，不能冒称为官方语义合并协议。本次是**参考逐类接口的VISTAR固定顺序适配**，不是Text2Seg官方算法复现，也不是RSUniVLM官方Table 7结果。没有引入SAM、CLIP打分、面积排序或精修。

## 唯一的合并规则

策略ID：`ascending_class_id_last_positive_wins_v1`。

- 按原数据集评测类别ID从小到大遍历，每个类别掩码为1的像素写入该类别；后写覆盖前写。全部原类别查询必须存在且通过原回执校验，不能把缺失或失败查询当作空掩码。
- LoveDA依次为background、building、road、water、barren、forest、agriculture。UAVid依次为background clutter、building、road、tree、low vegetation、moving car、static car、human。显式背景查询也参与，先写入，之后正前景可覆盖。
- FLAIR使用原12类ID顺序：building、pervious surface、impervious surface、bare soil、water、coniferous、deciduous、brushwood、vineyard、herbaceous vegetation、agricultural land、plowed land。
- 全部查询均为0时：LoveDA/UAVid填其原背景0；FLAIR填预测专用背景12。FLAIR背景12不计入12类均值，但有效GT像素上预测背景仍计入真实类别FN，保留在12×13混淆矩阵中。不能填255忽略掉这些错误。
- 合并在已拼回原图的二值掩码上进行，无缩放或GT参与；不改原512裁块几何、文本、自由/约束解码历史。每张合并标签图在首次读取GT像素之前保存。

固定顺序会影响重叠像素的结果，已完整写入run_config和指标文件。顺序不根据测试集GT或分数选择。

## 指标

复用当前仓库原GT读取和评分实现：`tools/rsunivlm_paper_protocol.py`、`baselines/segearth_ov/protocols.py`、`baselines/binary_boundary_wfm.py`。

**mIoU**从所有图像累加的原图尺度混淆矩阵计算，并保留每类IoU、F1、mAcc、pixel accuracy；不平均各图/各进程mIoU。LoveDA保留原raw0/255忽略、1..7→0..6；FLAIR保留原12类及GT忽略范围；UAVid保留原8类和原ID/RGB解码。新结果是互斥标签mIoU，旧query mIoU留在源目录，新metrics另有source_query_miou便于核对。

**WFm**从合并语义标签的类别边界计算，保留类与类之间的边界，不能先把所有非零类并成一个前景。使用原IDGBR口径：Sobel 3×3、一轮3×3膨胀、Margolin beta=1/sigma=5；GT忽略区域及margin=2支持区域排除。逐图计算GT出现的edge/nonedge分数，先分别取图像均值，再取正均值中的较小值。不是Boundary IoU。全部GT被忽略或忽略扩张后没有有效支持时，该图对边界类均值没有贡献，不能伪造0分或把它算作推理失败。

SECOND是六类可重叠、无方向变化查询，原GT允许同一个像素属于前后时相的不同类别，因此本次不转成单标签。新七集summary直接继承SECOND及xBD-pre、CHN6-CUG、LEVIR原指标，标注来源；原本缺少的边界值仍为空，不写0、不假称重新评分。

## WSL运行

原队列退出后执行；使用原vistar_rsunivlm环境，无新权重、无GPU操作：

```bash
cd /root/code/vistarmodels
source /root/anaconda3/etc/profile.d/conda.sh
conda activate vistar_rsunivlm

python -m pytest tests/test_rsunivlm_fused_scores.py -q
SOURCE_RUN=/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1_formatfix_rle_v1 \
SCORE_WORKERS=2 MAX_SAMPLES=0 \
bash run_bash/rsunivlm_fuse_rescore.bash
```

测试失败先停止，不执行下一条评分命令。完整测试及真实评分尚未在本机运行，WSL运行回执待用户提供。正式评分使用上述MAX_SAMPLES=0；若临时指定开发前缀，必须另给OUTPUT_DIR，不能与全量目录混用。

默认新目录为源目录后加`_fixedorder_fusion_v1`。中断后运行同一命令续算。workers只控制CPU进程，可用SCORE_WORKERS=0单进程；默认两个spawn进程，最多4张任务在途，不整集预读，不依赖GPU batch。大图WFm需要较多内存，沿用2进程开始即可。

新目录包含：

- `run_config.json`：源plan/summary SHA、固定顺序、代码与环境身份。
- `<dataset>/pred_mask/<sample>.png`：每图一张uint8合并类别ID掩码，不保存RGB、可视化、GT副本或逐类掩码副本。
- `<dataset>/rows/<sample>.json`：逐图混淆矩阵、WFm、来源回执hash、掩码hash和记录hash。断点续算前重验原二值PNG，合并结果和行记录均匹配才跳过GT读取/WFm计算。
- `<dataset>/metrics.json`：合并后的全量/部分指标、重叠与未覆盖计数、原query mIoU、失败清单。
- `paper_summary.json` / `paper_summary.csv`：三个新指标＋其他数据集原指标，CSV百分比保留两位小数，JSON保留完整数值；不改论文TeX中的结果。
- `runtime_finished.json` / `status.json`：处理完成状态，失败时complete=False，返回非零退出码。

源目录使用共享队列锁、输出目录使用独占后处理锁；源plan/vocabulary、GT和评分依赖、每类mask内容hash必须一致。原预测、文本、回执和指标不修改。损坏/缺失源掩码会单独记录失败并继续处理其余样本，始终不补背景冒充成功。

## 当前状态

已实现，待WSL回归与全量重计。此前用户提供的35.19/20.61/36.70是原query mIoU百分比，不能改名为合并后的标准mIoU；新数字必须由本工具运行产物确认。
