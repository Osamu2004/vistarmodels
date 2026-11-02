# RSUniVLM与Alters源图分块核对

2026-11-02。结论：当前七集队列与Alters的**512源图裁块规则一致**，不需要改动推理逻辑。此次只增加核对记录与供WSL运行的合成回归；既有队列源码/协议身份不变，同协议有效结果可以继续恢复。没有在本机运行测试、模型或WSL实验。

## 实际源码依据

核对本工作区`vistar`代码仓库，而非仅引用论文描述或当前启动脚本默认值。函数行号与SHA-256见`alters_tiling_audit_20261102.json`。

| 任务 | Alters实际空间处理实现 | 与队列的对应 |
|---|---|---|
| LoveDA | `eval_flux2_loveda.py`的`_load_rgb_tensor`、`_pad_image_to_tile`、`_tile_coords`、`_predict_tiled_mask` | 原生1024×1024通常切四块512，无全图预缩放 |
| FLAIR#1 | `eval_flux2_flair.py`逐批读取原生512×512并严格检查大小；`tools/flair_gsnet_protocol.py`读取前三个RGB波段 | 每张原生patch就是一块，不再次切小或缩放源图 |
| UAVid | `eval_flux2_uavid.py`同名裁块/补边/拼接函数 | 原生1920×1080图像为4列×3行，共12块；右/下补黑后恢复原生范围 |
| CHN6-CUG | `eval_flux2_chn6_road.py`区分source tile和model input | 正式512/512设置下无额外块内resize；原生图切块、拼接 |
| xBD-pre | `eval_flux2_xbd_pre.py`导入上述CHN6的`_predict_tiled_mask` | 使用同一源图裁块引擎，source tile=512 |
| SECOND | `eval_flux2_second.py`严格检查T1/T2尺寸相同，再按同一坐标裁两张图 | 每个请求使用配准的同位置图块；交换时相仍保持相同坐标 |
| LEVIR-CD | `eval_flux2_levircd.py`直接复用SECOND入口 | 空间处理同上；方向并集属于后续预测聚合，不改变裁块 |

四个裁块引擎的`_ceil_to_multiple`、`_pad_image_to_tile`、`_tile_coords`经AST比较完全相同。Alters RGB张量在[0,1]中补0，对应队列PIL RGB补(0,0,0)，是黑色。不是在[-1,1]张量中补0。

具体共同规则：左上角原点(0,0)，源图块512×512，x/y步长都为512；逐行从左到右，无重叠、无中心裁剪、无全图缩小。边缘不足时只向右/下补齐。预测放回对应坐标后裁回原图H×W，再计算IoU及适用的边界指标。批量大小和GPU分片只控制调度，不改变源图坐标或边缘有效范围。

队列实现位于`tools/rsunivlm_paper_worker.py:Engine.evaluate/Engine.apply`：每块先恢复完整512预测，再取其左上有效h×w区域写回原图。这与Alters先拼到补齐画布、最后`pred_pad[:height,:width]`的边缘处理相同。不能把边缘有效区域直接拉伸成整块预测。

## 历史运行与默认值

LoveDA论文37.16/17.17对应归档目录`/root/data/experiment/eval_loveda_seg_cyclefullema0p99_step300000_cycleema_tile512_step1_seed0`，原生1024、512源图裁块、1步。依据外层`paper/docs/loveda_segmentation_provenance_20261101.md`和`aaai27_handoff.md`已有记录。本次没有重新连接WSL读取这些目录。

旧LoveDA、SECOND、LEVIR启动脚本仍有256默认值，不能用这些默认值替代论文512主结果的实际设置。Alters SECOND最终结果的历史交接记录也明确512原生裁块、593对、双方向；FLAIR归档为原生512的15,700张。其余集合按当前源码与稿件核对；本次不声称重新审计了所有远端历史运行配置，也不修改论文数值。

## 内部分辨率差异必须保留

RSUniVLM仍使用其官方SigLIP视觉处理器，把每个512**源图块**处理成384×384视觉输入；输出原生矩形RLE网格（可能23×24），用OpenCV INTER_NEAREST还原到512后拼接。Alters的512预测没有这个低分辨率RLE输出步骤。因此此次对齐的是源图取样范围、补边、拼接和评分范围；不是两个模型的内部视觉分辨率或输出细节能力。不要把修改SigLIP为512当作源图分块对齐。

## WSL验证

`tests/test_rsunivlm_alters_tiling.py`从实际Alters源码用AST提取三项几何函数，校验其审计哈希，再与队列产生的真实输入图块逐像素比较。合成尺寸含1×1、512×512、1024×1024、513×777、1920×1080；另查双时相同位裁块、交换方向和边缘恢复。用23×24非恒定模拟网格检查最近邻恢复与最终拼图，不加载预训练权重。命令统一在`RUN_QUEUE.md`；本机只解析语法，WSL测试尚未执行，真实推理精度/速度仍待用户回执。
