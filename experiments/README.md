# 复现实验指南

复现按“公式正确 → 机制训练 → GPU算子 → 整模型”逐级推进。不要把低层验证写成高层结论。当前运行范围是第一层。

## R1：CPU算子正确性（已运行）

```bash
python -m unittest discover -s tests -v
python -m experiments.run_cpu --output results/local/cpu-check.json
```

依赖见根目录 `requirements.txt`。实际核验使用Python 3.12.10和PyTorch 2.13.0+cpu；只需CPU，不加载外部模型、数据或远程代码。

| 项目 | 设置 |
|---|---|
| q、k布局 | `(B,H,T,dk)=(2,2,65,8)`，L2归一化 |
| v布局 | `(2,2,65,6)` |
| g | `log α`；通道门取`[-0.2,0]`，标量门为通道均值 |
| β | `[0,1)`中的随机标量 |
| 随机种子 | 0、1、2 |
| 数据类型 | float64、float32 |
| 块长 | 1、8、16、32、64；65长度特意覆盖非整块尾部 |
| 对照 | 通道/标量门 × Delta/直接累加 |
| 比较目标 | 每步输出、最终状态；测试另比较梯度 |
| 容差 | float64 `atol=rtol=1e-11`；float32 `1e-5` |

四个组合是**递推机制对照**，不等于完整KDA/GDN/GLA/Mamba2训练模型的统一复现。参考算子也不含投影、短卷积、输出门、RMSNorm或MLA。

### 公式与代码对应

`kda_recurrent`直接实现“先遗忘→计算旧预测→更新误差→读输出”。输入g是log门；输出没有额外 `1/sqrt(dk)` 缩放。与FLA比较时需统一归一化、缩放、状态方向、维度布局及门控融合选项。

`kda_chunkwise`对块内残差求解下三角系统：

$$
(I+\operatorname{Diag}(\beta)A)E
=\operatorname{Diag}(\beta)(V-K_GS_{\mathrm{in}}),
$$

其中 $A_{ij}=k_i^\top\operatorname{Diag}(\exp(G_i-G_j))k_j$，只保留 $j<i$；$K_G$ 的第i行为 $k_i\odot\exp(G_i)$。输出再加上块输入状态贡献与块内残差贡献，末状态用于下一块。

本实现显式构造 `C×C×dk` 张量并调用通用三角求解。这样易读、可求导，但成本与官方Triton优化实现不同。它也没有实现通用DPLR速度对照，因此不报告“比DPLR快多少”。

## R2：MQAR机制训练（协议已整理，未重新运行）

完整协议见[mqar-protocol.md](mqar-protocol.md)。优先补齐HTML附录中的原脚本与原始训练JSON；如果重新独立实现，应明确标为新实验，并记录相对于原资料和论文的差异。

不能用单个toy（玩具规模）实验宣布KDA必然胜过GDN。原资料报告的玩具结果本身为GDN≈KDA，应保留差异，并考虑任务难度、容量、训练预算与超参数，而不预设必须复现某个排序。

## R3：官方GPU算子（未运行）

[FLA官方KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)有GPU实现与测试。以下为复现入口，当前机器没有CUDA GPU，本仓库未执行。

发布时期参考路径：

```bash
git clone https://github.com/fla-org/flash-linear-attention.git third_party/fla
cd third_party/fla
git checkout v0.4.0
python -m pip install -e .
# 测试额外依赖以该版本pyproject与官方说明为准
python -m pytest tests/ops/test_kda.py
```

完整测试可能耗时较长；按目标先检查同一版本的测试参数，不能把跳过的项目写成通过。安装前核对GPU、CUDA、PyTorch、Triton和测试依赖；版本号取实际安装结果记录。

当前主分支的benchmark（基准测量）另有入口，**不要与旧tag路径混用**：

```bash
python -m benchmarks.ops.run --list
python -m benchmarks.ops.run --op chunk_kda --modes fwd --no-base
```

按当次实际commit锁定来源，记录预热、前向/反向模式、dtype、batch、head维度、序列长度与重复次数。KDA与DPLR用相同输入和计时条件；同步GPU计时，分别报告中位数和波动。

## R4：公开48B模型推理与评测（未运行）

[官方用法](https://github.com/MoonshotAI/Kimi-Linear#usage)与[模型卡](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)是部署入口。公开checkpoint已经训练至5.7T，不能用其单次评测直接替代论文1.4T的架构公平对比。

版本选择有两条路径：发布时期HF自定义代码，或当前Transformers原生KimiLinear。旧模型远程代码和新FLA的门控API存在差异；先读取依赖与类型定义，再锁定匹配版本，避免无条件升级。[版本核验](../notes/sources.md#4-官方开源入口和-api)

官方vLLM示例：

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 --tensor-parallel-size 4 \
  --max-model-len 1048576 --trust-remote-code
```

这是官方配置示例，不是硬件最低要求承诺。开始前固定模型revision并阅读远程模型代码；从短输入验证生成，再逐级拉长。服务测量分开报告：

| 条件 | 指标 | 目的 |
|---|---|---|
| 固定batch、相同输入/输出长度 | TTFT首token时延、TPOT每输出token时延 | 比较请求处理延迟 |
| 相同显存预算、各模型可用batch | tokens/s、峰值显存 | 比较吞吐容量 |
| 固定数据集、版本与提示模板 | 基准分数与置信区间 | 评估公开权重质量 |

每次记录硬件、并行度、精度/量化、框架commit、模型revision、调度策略、KV缓存命中和失败请求。错误显式保留，不吞错、不无限重试；OOM（显存不足）也属于实验结果。

## 本次未覆盖的内容

没有运行完整预训练、SFT（监督微调）、RL（强化学习），没有下载48B权重，也没有新生成MQAR曲线。根目录状态表与[实际结果](../results/README.md)给出了当前范围，原材料历史曲线仍可在[HTML转写](../materials/html-report.md)阅读。
