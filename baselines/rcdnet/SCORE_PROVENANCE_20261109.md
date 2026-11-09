# RCDNet 历史分数来源核查（2026-11-09）

## 结论

原论文表格中的60.21是已发表的合成数据预训练版本的LEVIR-CD changed-class IoU，沿用到本项目旧表，不能视为本地公开权重推理的实测结果。此次46.03432144395981%与28.45222020216903%是用户对同一批128对保存掩码运行边界评分所得。WFm评分没有把60.21降为46.03：评分入口要求重算的每图/全局混淆计数及IoU/F1与原metrics.json一致，否则不会返回complete=true。

## 可复查证据

1. 工作区`aaai27_handoff.md:2589`的2026-07-18记录明确：使用RCDNet synthetic-pretrained论文IoU=60.21，旧表F1=75.16由二值恒等式2*IoU/(1+IoU)换算；先前53.98/70.11也是原论文真实数据版本的IoU及其换算F1，不是本机推理结果。
2. `aaai27_handoff.md:2423`记录7月19日Setup由SECOND + synthetic缩写为SECOND，数值仍为合成预训练版本；方法名也省略了synthetic后缀。简化显示并未改变数值来源，但容易与后来实测混淆。
3. 7月21日代码提交6370668中`baselines/rcdnet/README.md:18-26`已明确区分公开SECOND真实数据权重（论文对照53.98）与synthetic版本（论文60.21）。相同区分见`aaai27_handoff.md:6099-6129`；该记录明确禁止将默认Figure5预测当作60.21的来源。
4. 旧AAAI稿`AuthorKit27zzy/AnonymousSubmission2027.tex:396`仍可见RCDNet=60.21；本轮编辑前快照`output/pdf/cvpr27_rcdnet_wfm_20261109/before/paper/sec/3_method.tex:201`显示CVPR稿继续沿用60.21且WFm空缺。
5. 用户7月22日启动日志`/Users/zhouziyu/.codex/attachments/e0d3e7f8-9d83-40c3-ac74-6dc92bb4f067/pasted-text.txt:34-58`绑定环境rcdnet、输出`/root/data/experiment/rcdnet_levircd_test_native1024_tile512`、公开SECOND真实数据权重、building查询、阈值0.5、512非重叠分块及fp16。该启动片段本身不是成功完成回执，也没有checkpoint文件SHA。
6. 用户11月9日提供同目录128/128、complete=true、IoU46.03432144395981%、WFm28.45222020216903%的终端回执，已转录到`receipts/wsl_levircd_wfm_20261109.json`。`tools/score_rcdnet_boundary.py:76-93`验证原计数/IoU/F1与输入SHA，124-137行只读已有预测并输出边界结果，不运行网络。

## 原论文交叉验证

正式WACV 2026论文Table1报告真实数据版LEVIR-CD IoU53.98、synthetic版60.21；未报告本项目3px WFm。本次46.03分别低7.95和14.18个百分点。真实数据版53.98是更相关的版本对照，差距的具体原因仍未确定，不能只归因于分块或环境。

原文：https://openaccess.thecvf.com/content/WACV2026/papers/Korkmaz_Referring_Change_Detection_in_Remote_Sensing_Imagery_WACV_2026_paper.pdf

## 当前边界

当前主稿表2使用同批实测46.03/28.45；60.21保留为历史文献值。上述历史文件位置相对本机外层工作区`/Users/zhouziyu/code/vistar`，代码路径相对vistarmodels仓库。未独立读取用户WSL的原JSON、掩码或checkpoint SHA，未启动新推理或重算数值。complete=true仅支持保存预测与原指标一致，不支持宣称严格复现原论文准确率。
