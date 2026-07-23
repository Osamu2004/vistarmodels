# VIP 接入工作记录

## 官方实现核对

- 官方仓库：`MiSsU-HH/VIP`，固定提交
  `5bd25ee03ec25c1538622cf7da661e8c0461e769`。
- 方法没有单独训练的 VIP checkpoint，运行时需要 DINOv3 ViT-L/16、
  dino.txt 视觉头与文本编码器、BPE 词表三项公开资产。
- 当前官方提交在 `dinov3/hub/backbones.py`、`dinov3/hub/dinotxt.py` 和
  `dinov3/eval/text/tokenizer.py` 中保留了空的本地路径，原始入口无法直接
  加载权重。因此适配器不改第三方源码，而是在本仓库中按官方结构构建模型并
  注入显式权重路径。
- 论文附录写的是短边 336、窗口 224、步长 112；公开遥感配置实际采用最长边
  448，而公开视觉头固定 21×21 token 网格，对应 336 窗口。默认采用可执行的
  公开源码协议，并把这一差异写入运行元数据。

## 本仓库评价约束

- 复用 SegEarth-OV 已验证的数据发现与标签解码，覆盖 LoveDA 1,669 张、
  FLAIR#1 15,700 张、UAVid 270 张、xBD-pre 933 张和 CHN6-CUG 903 张。
- 所有指标在原始影像范围计算；输出区域指标、逐图结果及 IDGBR 3-pixel WFm。
- VIP 论文仅报告 iSAID、Vaihingen、Potsdam 和 VDD。上述五组入口属于固定词表
  和统一评价协议下的新复现，不能把论文中的其他数据集数值直接抄入主表。

## 待服务器验证

- 在 CUDA 环境完成 bootstrap 和依赖检查。
- 对每个数据集先运行 `MAX_SAMPLES=2 STRICT_PROTOCOL=0` 冒烟测试，再运行完整
  数据集并核对样本覆盖、混淆矩阵和 WFm。
- 当前本地工作站没有 PyTorch/CUDA，尚未执行真实 VIP 前向推理。
