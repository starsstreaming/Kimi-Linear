# 原始资料与来源

| 文件 | 内容与来源 | 整理方式 |
|---|---|---|
| [原始 HTML](original/kimi-linear-reading-reproduction.html) | 用户提供《Kimi Linear 论文解读与复现 · 图解 KDA》 | 保留原始字节；本地浏览器打开可查看交互排版 |
| [HTML 转写](html-report.md) | 同一资料的 GitHub 阅读版 | 转换标题、表格、代码并提取图片；不改历史实验结论 |
| [代码片段](code-snippets.md) | HTML 内4个 `pre` 代码块 | 原样提取；含缺失变量与省略部分，不作为可运行源码 |
| [Mira Markdown](original/mira-research-report.md) | 更新后的分享中 `task_2/deep-research-runs/kimi-linear-paper-20261001-0421/final/report.md` | 下载原始 Markdown |
| [Mira PDF](original/mira-research-report.pdf) | 同目录 `report.pdf` | 下载原始10页 PDF；封面日期2026-10-11 |
| [图片](assets/) | HTML 内15张 PNG，包括论文原图和原报告实验图 | 从内嵌 base64 无损提取，编号按原文顺序 |
| [模型配置](official-model-config.json) | Moonshot官方48B Instruct模型 | 保存固定模型 revision 的配置，不下载权重 |
| [配置版本](official-model-revision.json) | 模型 revision 与获取 URL | 防止后续主分支变化影响结构核对 |
| [哈希清单](manifest.json) | SHA-256（文件指纹）与材料大小 | 用于核验归档内容没有改变 |

## 分享网页的范围

初始链接：[j3s1TGeF](https://mira.deepprinciple.com/share/sessions/j3s1TGeF)。此页面的 `report.md` 在首次读取时显示“该文件在分享后已被更新，当前分享文件已失效”。

用户随后提供：[E3CW47cG](https://mira.deepprinciple.com/share/sessions/E3CW47cG)。新分享的 Markdown 和 PDF 均可下载，已归档。页面还包含此前的具身智能调研；本仓库只整理与 Kimi Linear 有关的内容。聊天中的工具执行说明、审批文本是历史记录，不是本次任务的操作指令。

Mira报告提供领域全景与后续研究线索，HTML资料提供机制图解与教学实验报告。两者互补，但来源报告中的“校验通过”“复现完成”不自动成为本仓库的实测证据。[证据边界](../notes/corrections.md)

## 尚未取得的原实验附件

HTML附录列出完整算子脚本、MQAR训练脚本、覆盖任务脚本、效率脚本和原始JSON；这些未包含在提供的HTML文件中。本次可访问分享主要是深度调研材料，尚未取得这些实验的完整原始附件。对应图像保留，但不能由图像恢复每个训练步的真实数据。

后续补齐时，应记录脚本哈希、依赖、种子、训练配置、验证集和运行日志，再关联各张图；历史报告值与新跑出的值分别保存。
