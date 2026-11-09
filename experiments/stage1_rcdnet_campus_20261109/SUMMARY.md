# RCDNet 校园重新推理结果（2026-11-09）

校园 GPU5 的 RTX 3090 上已完成两套 LEVIR-CD test128 推理及原图3px边界WFm评分，均128/128、0失败、真实退出0。预测、GT和评分均已持久化，私有节点缓存已标准回收，GPU预留已结束。

| 方案 | 原图IoU (%) | 原图F1 (%) | 原图3px WFm (%) |
| --- | ---: | ---: | ---: |
| 预先指定主方案：整图缩放512 | 40.117765 | 57.262920 | 16.586877 |
| 预先指定对照：原图512分块 | 46.035337 | 63.046841 | 28.453593 |
| 之前WSL原图512分块（用户回执） | 46.034321 | — | 28.452220 |

同一次整图前向在512输出网格的IoU为40.076556%，与上表还原1024后计算的原图IoU分别报告。分块方案比旧WSL回执的IoU高0.001015个百分点、WFm高0.001372个百分点，两位小数一致；本次没有取得明显提升。FP32、运行库/设备差异可能影响极少量边界像素，但未读取旧WSL全部掩码，不能确定微小差异的具体原因。

## 使用的版本

官方源码固定0966e96ff7075476d77442bbf6623ed5086d52da。检查[官方仓库](https://github.com/yilmazkorkmaz1/referring_change_detection)及其[权重发布目录](https://drive.google.com/drive/folders/1foXpLPz3jtaQN7l6UdlDFVSgakgm6RXP)，公开目录只有SECOND与CNAM-CD真实数据权重。SECOND包含building类别，适用于此次查询；CNAM-CD类别不含building，因此选择SECOND。未取得论文最优合成预训练版的权重，不能把此次运行称作60.21%那一版本的复现。

- checkpoint：SECOND-model.safetensors，352523328字节，SHA256 f4b772d1a22334a25d9435f4f32a0e0c3db18111e9454bd92586e55535f24a4e。
- CLIP：openai/clip-vit-base-patch32，revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268；官方权重SHA256 a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f。
- 活动参数严格加载，missing/unexpected均为空。仅跳过88个官方checkpoint中已知未参与当前模型前向的legacy cross_mamba/norm2张量，名单保留在metrics.json。
- 环境：隔离Python3.9.23，Torch2.1.0+cu121，Transformers4.30.2，CUDA扩展以CUDA12.9和Conda GCC/G++11.4编译；运行GPU为RTX3090。实际扩展导入已通过。
- 查询building，阈值固定0.5，FP32，单方向；未通过测试GT选阈值、权重或主方案。matmul TF32=False，cuDNN TF32=True，详见runtime_diagnostics.json。

## 产物与核验

校园任务根目录：/data1/zzy26/outputs/vistar_rcdnet_campus_20261109。

两套产物分别在evaluations/whole_resize512和evaluations/native_tile512；其中metrics.json、per_image_metrics.jsonl、boundary_wfm_3px/paper_summary.json与predictions_and_gt.tar可对应检查。主方案tar有512张掩码（128原图预测+128原图GT+128网格预测+128网格GT），对照tar有256张掩码。控制器逐张检查尺寸和0/255二值取值，并记录归档SHA：

- 整图：aa4c8d24d8348624c28c30121662c1c357f84fd7518809c20fa4e0e2adf1f17e，3471360字节。
- 分块：bb8b8e7da8739cc36b5f721f7e606b4e4b1a983792371104bc7665138d55da82，2457600字节。

评分器确认逐图/全局原计数、IoU/F1与预测一致，并核对源文件前后SHA，两个paper_summary均complete=true。最终checkpoint SHA再次核对通过。小型JSON回执已取回本仓库receipts/，大图像tar留在校园共享盘。真实终端退出0、缓存清理deleted=true见receipts/cache_receipts/9058c567af724d29a7e7c54b13571c2a/。

GPU阶段从本对话实际Popen获取PID。整图PID2640714的早期登记使用了尚未刷新的GPU进程缓存而被拒，没有冒领其他进程；随后刷新确认其已退出并释放预留。分块PID2655717经新采样确认UID1115、start_ticks和boot_id后成功登记，结束后新扫描确认gone并完成释放。

## 时间与准备过程

整图进程内部计时63.84秒，分块114.36秒，包含各自模型装载与推理，不等于纯前向吞吐量。总任务7197.96秒，包括准备、失败重试、等待、两套评分与归档。早期CPU准备遇到共享盘I/O、头文件软链接和缺少系统g++；最终使用受限头文件提取、有界gzip流及已核验Conda工具链完成。失败缓存均按生命周期回收。不能把开启节点缓存本身当作提速结论。
