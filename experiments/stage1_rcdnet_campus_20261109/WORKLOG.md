# 执行记录

## 准备

- 2026-11-09：用户授权校园重新推理，优先最强可获得官方版本。官方主README声明公开权重仅真实数据训练；合成数据本身已公开不等于合成预训练checkpoint已公开。
- 全节点fresh scan后自动预留GPU2 P100，读到官方selective-scan setup仅编译SM70/80/90，释放无运行P100预留并改为Ada UUID GPU-e3898b86-d113-6c5e-0108-0b829ebf9ec8。
- 共享盘git status检查超过60秒，已中断本会话只读检查，后续采用有界文件检查及独立本地缓存。
- 本次任务不重用旧预测充当新推理，不根据test GT选阈值或最佳checkpoint。

## 权重与空间协议

官方Drive目录已实查，仅SECOND-model和CNAM-CD-model两份真实数据权重。选用含building类别的SECOND，352523328字节，SHA256 f4b772d1a22334a25d9435f4f32a0e0c3db18111e9454bd92586e55535f24a4e。校园Google不可达，经本机官方直下和受限单文件HTTP服务传至校园，两端SHA相同，服务已停止。

主方案whole resize512，次方案native tile512均预先冻结、同权重FP32，不按测试GT挑选；主方案只读forward hook额外保存同一次forward的512预测网格IoU，原图1024指标和3px WFm独立报告。CPU准备期间更新控制/IO脚本并归档哈希，数值适配器/官方源不改；尚未启动CUDA。

## 存储与依赖检查

环境归档4.17GB在GPU2的NFS首读较慢，单进程/proc读取量持续增长，无GPU启动；32MiB四路有界读取14.40s未显示并行收益，因此保留原恢复流程，没有重启数值任务或更换来源。官方源码归档507308字节SHA256 62b5c8ead30183449a189818525272fc8b48042d23a3a8b99bfd424a903bd8a0已在校园提前下载并核验。源码静态检查发现vmamba需要fvcore，已加入CUDA启动前CPU导入门禁。

本任务节点根部脚本、部署文件、公开权重与状态JSON已明确按语义清单归类；audit_semantics含链接检查通过，无需移动文件。

## CLIP离线准备

校园官方HuggingFace不可达；镜像公开API可读但文件路径返回Cloudflare1010。改为本机从官方openai/clip-vit-base-patch32固定revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268获取，文件SHA核验后通过受限单模型LAN服务传到校园HF缓存。新环境加入显式临时CPU bootstrap门禁：只等待这次CLIP保存命令所需的已核验资产，完成后恢复普通离线配置，在CUDA前归档其哈希；没有tensor或forward补丁。

## 有界I/O恢复重启（CPU，无新模型结果）

v1完整校验环境归档已通过，但tar流默认10KiB块在NFS解包极慢，记录/proc读取量确认。仅结束本对话启动的CPU worker3073645，UID/命令/缓存登记核验后发SIGTERM；真实退出-15、缓存079d90af8e7d4c1ca0282e340a90ed29已通过标准管理器回收。保持同逻辑任务8GiB，不绕过保留额度。v2复用完整SHA校验且核对归档dev/inode/size/mtime/ctime未变，改8MiB读缓冲；不改官方源、掩码评分或模型数值。v2使用显式离线CLIP资产门禁，v1临时sitecustomize随旧缓存回收，不再使用。模型输入仅在总额度含128MiB输出余量可容纳时整体缓存，否则直读data1。

## GPU5迁移与缓存回退

GPU5同一归档16MiB有界读取1.54s，自动预留其空闲3090 GPU1 UUID GPU-8e865959-cdfe-459b-228e-f7ff8a483cc7；GPU2预留已释放。v2只停止本对话CPU worker3279478，真实退出-15，标准管理器已回收缓存。GPU5分配新缓存时，管理器在统计其他既有缓存时遇到symlink报错，尚未为本任务创建新缓存。未修改其他缓存、配额、权限或全局管理器，按规则准备direct_data1回退及最小独立环境。

## 缓存计数修复与实机检查

仅本任务runtime_code/node_cache.py的allocated改为os.walk不跟随链接且lstat计入链接自身，避免既有venv符号链接导致所有新分配报错。不修改全局工具、其他目录、配额和磁盘余量条件。私有副本SHA0b7d6afddcb1f52c8e50b999f3c569099e4bf133c580e8e8ff932ebd6abd9967；校园实际20项生命周期/配额/身份/不跟随外链检查通过，含2项新增回归。GPU5缓存0842c319d5f04ab09df18a8240a84fe7开始准备；GPU2旧两份均已回收，未跨节点重复领取8GiB。

## 恢复失败及有界gzip修正

v3未启动CUDA，因不需要的RPM系统配置软链接scratch/python_headers/usr/libexec/platform-python-config触发安全拒绝，真实exit1且缓存标准回收。头文件范围现缩至原已验证的usr/include/python3.9。进程CPU计数显示大tar流缓冲引入明显CPU复制开销；用BufferedReader8MiB + gzip.GzipFile + tar64KiB在同归档读取64MiB/653项用5.51s（只读CPU检查，不是模型提速结果）。v4按此恢复，同源完整SHA和8GiB额度保持。准备期间空闲GPU预留已释放，CPU门禁通过后再自动领取。SSH断线后原tmux状态已恢复检查，没有重复启动模型。

## 扩展构建工具链

v4已恢复5043359825字节，CPU依赖安装通过，但系统/usr/bin/g++缺失导致扩展build失败，尚未CUDA推理。改回此前校园OVCD已使用的Conda gcc/g++绝对路径，先实际--version验证，再恢复本次环境；模型源码及数值协议不变。

## 最终运行及收尾（2026-11-09 18:28 CST）

v5选择Conda GCC/G++11.4完成官方selective-scan构建与CPU导入；smoke2退出0。新鲜按UUID预留并复查后完成whole_resize512、native_tile512各128对，退出0；两次边界评分亦退出0。分块PID2655717的新采样身份登记成功，结束后确认gone并完成释放。整图早期登记因进程缓存尚未刷新被拒，未声称其登记成功。最终RESULT.complete=true。整图IoU40.117765/WFm16.586877；分块IoU46.035337/WFm28.453593。两套归档掩码二值/尺寸/数量及SHA已验证。缓存管理器真实退出0、experiment_success=true、CLEANUP.deleted=true。结果文件持久化后取回36个小型回执/逐图文件；图像tar留在校园。用户同意后编写SUMMARY.md。旧WSL46.034321/28.452220与此次分块几乎一致；不以对照结果替换预先指定的主方案，不宣称论文合成预训练版已复现。
