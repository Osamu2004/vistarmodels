# 一次调用完成失败恢复流程（2026-11-04）

用户明确选择保留当前推理协议，自动修复、补跑并汇总剩余失败，不增加约束RLE解码。一次调用不意味着将任意坏输出变成合法预测：原最多三次真实回答均缺计数、破损词或行宽不合法时，保留失败记录及非零退出状态，不补背景、计数、行或EOS。

## 原入口的缺口与本次改变

原入口只在GPU补跑前规范旧文本；新生成的road**2等错误可能继续消耗备用提示预算，结束后还需要手工再调用一次。工具更新又会改变派生identity，如果只从最初source导入，将无法直接复用上一轮派生目录中已经恢复的图片。

本次补全：

1. 每次模型返回后先应用同样的两条分隔符规则，再经原解码器完整验证。合法才接受；不改变原模型前向、提示词、裁块、batch、网格、自由输出或每查询最多两个备用模板。不合法仍进入原重试/跳过逻辑。被接受的机械修复保存完整原回答及SHA/编辑记录，仍是一条真实模型回答，不虚构尝试。
2. GPU worker正常结束（exit_codes全部0/1，当前identity/launch_id/PID回执匹配、无存活子进程/监督故障）后自动执行一次格式终检。只有终检新修好的数据集才做CPU重建/重计数，避免全部数据无意义再算一遍。已有GPU汇总完整且没有新修复时直接使用该汇总。GPU崩溃/中断不被后处理伪装成成功。
3. --reuse_run先导入已结束的旧派生运行，再由最初source填空。验证同一原来源、人口/分片/模型/数值源码、格式政策、路径元数据及逐份预测/原文身份；保留已成功掩码、合法回答和实际尝试次数。旧派生目录活动锁仍被占用时拒绝迁移，不能边跑边改同一来源。
4. 用尽原三次模板的无效历史不重置。若CPU阶段已完整就不启动GPU；若仍失败但没有任何剩余预算/缺失查询，也不启动空的GPU扫描，直接报告剩余失败。真正需要生成的缺失/未耗尽查询才进入原guarded监督入口。
5. 最终format_repair_report逐集记录original_failed、recovered_original_failed、original_failed_not_yet_recovered、newly_scored_prior_unaccounted、successful_images。终端输出[recovery result]。complete只表示成功覆盖全部选定样本；原失败总数中尚未恢复和此前GT中断未计分的图片分别统计。

## 目录与策略证据

仓库内单命令入口为run_bash/rsunivlm_recover_once.bash，默认在当前WSL环境执行回归→正值GT审计→带reuse_run的修复/补跑/终检汇总，失败即停；SOURCE_RUN和REUSE_RUN可由环境变量显式指定。默认RUN_TESTS=1，本机不执行这个脚本；它会在WSL真实调用GPU。旧队列结束后才能更新并运行，详见RUN_QUEUE。

带--reuse_run的默认输出为原source同级_formatfix_once_v2，身份冲突仍用_p后缀；旧source和旧派生目录都只读导入。新protocol.runtime_format_repair记录在线机械处理政策/工具SHA，format_repair.reuse_run记录复用来源plan SHA和导入器SHA。原15个数值源码不变，LEVIR继续显式vistar_positive。

相同原source、相同reuse_run和参数可续跑新目录，已有有效预测与回答不会再前向。不要把新的_formatfix_once_v2当作source_run，也不要用旧主队列Bash覆写这个派生协议。若以后再换工具版本，仍通过显式reuse_run校验导入旧结果；不强改缓存身份。

runtime_format_repair_rankNNN.json记录每卡checked/already_valid/repaired/unrepairable查询数；单位为回答查询，不是整图。format_repair_scan_post_gpu.jsonl与post_gpu_normalization记录末尾扫描，post_gpu_rescored_datasets记录确实重算的集合。原GPU结束回执保留真实0/1退出码，不被CPU收尾改写；最终是否恢复完整看format_repair_report.final_complete和paper_summary.complete。

## 用户当前快照与验证边界

用户贴出的运行中快照：63条RLE查询通过离线修复；原失败整图已恢复LoveDA64、FLAIR1152、UAVid9、xBD1、CHN6为0、LEVIR为0、SECOND1；LEVIR另补计分3张。这个快照不代表整轮终态，尚未恢复包括尚未处理。

本对话无用户WSL可连接入口，未启动本地/其他服务器模型；当前只交付代码与命令。不要在旧队列运行中git pull，更不能把新工具覆盖到它正在使用的文件。先让旧队列完成/正常退出，再更新并显式复用旧派生目录。

WSL回归覆盖：新回答先规范再由原解码器接受、原文本/标志不变；GPU正常部分失败后自动修复并选择性重算；GPU致命退出不能变成成功；先导入旧派生成功/失败历史、保留三次已用预算、不重复GPU扫描；活动锁、源模型设置不符与原目录字节保留。本机遵守用户约定，只做AST/Bash/JSON/源码哈希静态检查，不执行这些测试、解码、模型或CUDA。实际一次调用恢复数量/完整性需WSL回执，不能承诺100%图片都合法。
