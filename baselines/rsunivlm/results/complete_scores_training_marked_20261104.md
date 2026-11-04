# RSUniVLM七数据集完整指标（含两个变化集边界，2026-11-04）

用户WSL完整回执：六项新增边界回归通过（1.74s），LEVIR-CD128/128、SECOND593/593全部成功，原IoU/计数核对通过，七集20,196张/对均complete=True。此处归档用户终端标量，未独立读取远端plan/summary或重算分数。

| 数据集 | 主指标 | 精度（%） | 边界WFm（%） | 成功/目标 | 失败 |
|---|---|---:|---:|---:|---:|
| LoveDA\* | mIoU | 41.34 | 11.21 | 1669/1669 | 0 |
| FLAIR | mIoU | 18.94 | 15.06 | 15700/15700 | 0 |
| UAVid\* | mIoU | 39.17 | 14.52 | 270/270 | 0 |
| xBD-pre | building IoU | 27.25 | 12.15 | 933/933 | 0 |
| CHN6-CUG | road IoU | 31.05 | 9.85 | 903/903 | 0 |
| LEVIR-CD\*† | building IoU | 27.65 | 12.08 | 128/128 | 0 |
| SECOND | mIoU_c | 9.49 | 7.98 | 593/593 | 0 |

\*：训练使用该基准或其源影像的派生数据。†：LEVIR-CD为经LEVIR-MCI→LEVIR-CC的相关影像训练来源；当前官方test具体重叠未逐项审计。其他未标记基准仅为公开清单未列名，不等于已排除所有训练影像重叠。

LEVIR与SECOND本次只补WFm，IoU27.65/mIoU_c9.49保持；其他五集边界与原结果完全一致。SECOND将预测和GT类别归属按固定ID升序后类覆盖合成0..6边界视图，沿用LoveDA整体语义边界；原六类可重叠IoU仍保留。没有逐类WFm平均，也没有反向OR或压成单一前景。

原图尺度；Sobel3×3、一轮3×3膨胀、Margolin(beta1/sigma5)、原IDGBR图像聚合。记录用户指定的VISTAR适配，不声明官方原版解码/合并协议或全部严格零样本。

WSL最终summary：

`/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1_formatfix_rle_v1_fixedorder_fusion_v1_cd_boundary_v1/paper_summary.json`

结构化原精度标量：`baselines/rsunivlm/receipts/wsl_complete_all7_boundary_scores_20261104.json`。原融合阶段及先前边界空值回执保持，新增此版本作为最新表。

训练来源：[RSUniVLM表10/12](https://arxiv.org/html/2412.05679v2#S7)、[LEVIR-MCI来源](https://arxiv.org/html/2403.19646v3#S3)、[LEVIR-CC影像来源](https://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf)。
