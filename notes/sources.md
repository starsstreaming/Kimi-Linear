# Kimi Linear 一手来源核验

核验日期：2026-10-13。只依据论文、Moonshot 官方模型代码，以及论文明确链接的 FLA（Flash Linear Attention，线性注意力算子库）。本文提供复现入口，没有在本机运行 GPU 实验。

## 1. 递推式与门控：公式顺序不能交换

采用论文约定，单头状态为 `S_t ∈ R^(d_k × d_v)`，`q_t,k_t` 是列向量。KDA 的式(1)：

$$
S_t=(I-\beta_t k_t k_t^\top)D_tS_{t-1}+\beta_t k_tv_t^\top,
\quad D_t=\operatorname{Diag}(\alpha_t),
\quad o_t=S_t^\top q_t.
$$

Gated DeltaNet（带遗忘门的 DeltaNet）的区别是 `D_t=α_t I`，每个头只有一个标量遗忘率；KDA 对每个 key 通道分别控制遗忘。来源：[论文§2.2、§3式(1)](https://arxiv.org/html/2510.26692v1#S3)。

便于教学和实现的等价展开：

$$
\widetilde S_{t-1}=D_tS_{t-1},\qquad
\widehat v_t=\widetilde S_{t-1}^{\top}k_t,\qquad
S_t=\widetilde S_{t-1}+\beta_t k_t(v_t-\widehat v_t)^\top.
$$

这里依次是「遗忘旧状态 → 读出当前 key 的旧预测 → 写入预测误差」。这是公式的代数展开，也与官方朴素递推的 `S *= exp(g)` 后再执行误差更新一致。[FLA naive.py，`naive_recurrent_kda`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py#L54)

关键校正：不能写成 `D_t(I-βkkᵀ)S`，也不能用未经遗忘的 `S_{t-1}ᵀk_t` 计算误差。一般矩阵不满足交换律；标量门控时这个问题被掩盖，逐通道门控时会真实改变算法。

官方发布模型的具体约束：

| 项目 | 核验结果 | 一手来源 |
|---|---|---|
| 写入门 `β` | 每头标量，`sigmoid` 输出；数学实数计算时在 `(0,1)`，浮点实现可能接近或取到端点 | [官方 modeling_kimi.py](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py#L507) |
| 遗忘门 `α` | 逐通道，核函数接收的是 `g=log α`，不是 `α` 本身 | [FLA chunk.py 参数文档](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py#L236) |
| 默认门控函数 | `g=-exp(A_log)·softplus(z+dt_bias)`，因此 `α=exp(g)`；`A_log` 为每头参数，`z`、bias 为逐通道量 | [FLA gate.py 参考实现](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py#L23) |
| q、k 稳定性 | 发布模型调用算子时启用 `use_qk_l2norm_in_kernel=True`，即 L2 归一化（向量长度规范化） | [官方 modeling_kimi.py](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py#L513) |
| 输出门 | 与遗忘门和写入门是不同参数；发布模型使用带 sigmoid 门的 RMSNorm（均方根归一化） | [官方 modeling_kimi.py](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py#L449) |

由上述约束可自行推导：归一化 key 下，`I-βkkᵀ` 的 key 方向特征值是 `1-β`，其他方向是 1；加上 `0≤α_i≤1` 时，状态线性转移的谱范数不超过 1。这只说明转移不会放大已有状态；不代表持续注入任意大的 value 后总状态必然有统一上界。

当前 FLA 主分支还支持 `allow_neg_eigval=True`，令 `β=2·sigmoid(logit)`，以及 `safe_gate/lower_bound` 门控。这些是库现在支持的变体，不能倒写成原始发布模型的默认配置。[FLA chunk.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py#L264)

## 2. 3:1 混合结构与真实模型配置

论文选择按层混合：三个 KDA 层后放一个全局 MLA（Multi-head Latent Attention，多头潜变量注意力）层。它不是同一层把注意力头按 3:1 切分。整体仍包含全注意力层，不能称为纯线性注意力模型。[论文§4](https://arxiv.org/html/2510.26692v1#S4)

真实发布 checkpoint（模型权重快照）有 27 层：20 个 KDA、7 个 MLA。MLA 的 **1 起始编号**是 `[4,8,12,16,20,24,27]`。尾部是 `KDA25→KDA26→MLA27`，所以「每第四层 MLA」是主要规律，不能覆盖最后一层。配置另给出 KDA 32 头、头维度128、短卷积核4，以及 `mla_use_nope=true`。[官方 config.json](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)

NoPE（无显式位置编码）不等于模型完全没有顺序信息：KDA 的递推和因果短卷积有顺序依赖；官方 MLA 没有应用 RoPE（旋转位置编码）。即使配置保留 `qk_rope_head_dim=64`，也不能据此断言实际使用旋转。[HF 原生模型文档](https://huggingface.co/docs/transformers/main/model_doc/kimi_linear#overview)

**推导限定**：对实际 27 层，在同样每层 MLA 缓存规格的理想估算里，保留的 MLA 层数比例是 `7/27≈25.93%`，其 token KV 缓存减少约 `74.07%`；完整占用还包括 KDA 状态、卷积缓存、权重与运行临时量。因此宣传的「最高75%」不能解释为总显存减少75%。推导依据是上述 config 层数，并非额外实测。

## 3. 性能宣称的限定

以下数值均为论文报告，未在本机复测。公平质量对比用 1.4T 训练 token 的同规模 MLA/GDN-H；公开权重训练至5.7T，不能把两套结果混为一谈。[官方 README](https://github.com/MoonshotAI/Kimi-Linear#overview)

| 宣称 | 应保留的限定 |
|---|---|
| 最高6.3倍 | 1M上下文；扩大批量后的吞吐收益。论文§6.3称其理论提升，固定 `batch=1` 在1M时为2.3倍；不可说所有请求延迟都降低6倍 |
| 1M预填充2.9倍 | §5.6同规模对比；短至4k–16k与MLA接近，128k之后优势明显 |
| RULER 84.3 | 128k质量评测；3.98倍是图1效率口径，不能改写为1M RULER成绩 |
| 优于全注意力 | 整体与平均指标优势；表4、5存在例外，例如EvalPlus、MATH500、LongBench V2、Frames，不能说每项均胜出 |

来源：[论文图1、§5.5表4–5、§5.6图7、§6.3](https://arxiv.org/html/2510.26692v1#S5.SS6)。论文§5.6行文把6倍与解码性能相连，图7明确 batch=1，§6.3进一步区别2.3倍与扩大batch后的6.3倍，讲解时应采用后者的明确区分。本文未核实到足够完整的端到端硬件、软件、调度参数，不能声称这些数字可由公开代码一键完全复现。

## 4. 官方开源入口和 API

| 用途 | 入口 |
|---|---|
| 官方总入口 | [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)；主要为README、报告、图像，不是完整预训练工程 |
| 基座权重 | [moonshotai/Kimi-Linear-48B-A3B-Base](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) |
| 指令权重 | [moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) |
| 官方引用的算子 | [FLA KDA目录](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) |
| 发布时间附近稳定代码 | [FLA v0.4.0](https://github.com/fla-org/flash-linear-attention/tree/v0.4.0) |
| 新的原生模型支持 | [Transformers KimiLinear 文档](https://huggingface.co/docs/transformers/main/model_doc/kimi_linear) |

Base/Instruct均为总48B、激活3B参数、1M上下文。激活参数是一次 token 计算实际参与的参数数量，并不等于只需存储3B权重。[官方模型卡](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

发布时官方推理路径：Python≥3.10、torch≥2.6、fla-core≥0.4.0，`AutoModelForCausalLM.from_pretrained(...,trust_remote_code=True)`，经 `apply_chat_template` 构造输入后 `model.generate(...)`。[官方 README Usage](https://github.com/MoonshotAI/Kimi-Linear#usage)

模型自定义代码还显式要求 `transformers>=4.56.0`。[官方 modeling_kimi.py 开头](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py#L20)

**已核实的版本陷阱**：旧模型调用 `fused_kda_gate(g,A_log,head_dim,g_bias=...)`，匹配[v0.4.0 gate.py](https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/fla/ops/kda/gate.py#L307)；当前主分支调用约定为 `fused_kda_gate(g,A_log,dt_bias=...,lower_bound=...)`，输入形状也改为`[...,H,K]`。所以不能假设`pip install -U fla-core`永远兼容旧模型远程代码。[当前 gate.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py#L338)

当前 Transformers 主分支已有原生 KimiLinear，文档示例省去远程代码；默认纯 PyTorch，可以安装 `kernels` 并传 `use_kernels=True` 使用优化算子。主分支文档要求从源码安装，具体稳定发行版是否支持须逐版本核对。[HF原生用法](https://huggingface.co/docs/transformers/main/model_doc/kimi_linear#usage-examples)

部署时官方示例为：

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 --tensor-parallel-size 4 \
  --max-model-len 1048576 --trust-remote-code
```

它提供 OpenAI-compatible API（请求格式兼容OpenAI的服务接口），不是论文确认的 Moonshot 商业托管模型名称。示例的四路张量并行是配置示例，不是最低GPU数量承诺。[官方部署说明](https://github.com/MoonshotAI/Kimi-Linear#deployment)、[模型卡本地 `/v1/chat/completions` 示例](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

## 5. 可执行复现路径：按层级验证

1. **机制正确性**：先读 `naive_recurrent_kda`，固定随机种子，用归一化q/k、负值log门g、sigmoid后的β。比较逐token递推、chunk（分块）输出及最终状态；继续比较梯度。官方测试已这样组织，包含变长、布局、门控融合等分支。[当前 `tests/ops/test_kda.py`](https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_kda.py)、[v0.4.0测试](https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/tests/ops/test_kda.py)
2. **算子性能**：在锁定的当前FLA提交上运行 `python -m benchmarks.ops.run --op chunk_kda --modes fwd --no-base`。另用 `--list` 检查算子名。该测试隔离算子调用并预热自动调优；这不是48B整模型的6倍复现。[官方 benchmark runner](https://github.com/fla-org/flash-linear-attention/blob/main/benchmarks/ops/run.py)
3. **模型推理**：选择发布时自定义代码栈，或当前原生Transformers栈，分别锁定模型revision与所有依赖；从短输入开始验证prefill（预填充）、逐token生成和缓存延续。不要混用两套门控API。
4. **服务端评测**：用vLLM记录TTFT（首token耗时）、TPOT（每输出token耗时）、总吞吐、峰值显存，分别比较固定batch和相同显存预算下最大可用batch；记录上下文长度、输出长度、硬件、并行度、量化、缓存命中率、版本与调度策略。这是建议的方法，不是声称已经复现。
5. **训练与质量表**：当前官方总入口没有公开完整1.4T同配方预训练工程或全部数据；开源权重可做独立评测，不能等同完全复现受控训练对比。FLA另提供通用训练/评测入口，不能自动视为论文原始训练流程。[官方总仓库文件树](https://github.com/MoonshotAI/Kimi-Linear)、[FLA训练与评测说明](https://github.com/fla-org/flash-linear-attention#training)

依赖审核也应分时代：[v0.4.0 pyproject](https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/pyproject.toml) 与[当前 pyproject](https://github.com/fla-org/flash-linear-attention/blob/main/pyproject.toml)要求不同。当前CUDA extra为torch≥2.7、Triton≥3.3；不能将这个要求写成论文发布时统一环境。

## 6. 高风险误解清单

- 把遗忘门α、误差写入门β、输出门混成同一个gate。
- 把内核的log门g直接当成概率α传入。
- 先算误差再遗忘，或任意调换两矩阵相乘顺序。
- 把3:1解释为头数比例，或者画出实际不存在的28层模型。
- 把全模型说成O(1)缓存：KDA固定状态，但7层MLA仍有随长度增长的缓存。
- 把最高6.3倍吞吐写成任意硬件、任意长度、batch=1都降低6倍延迟。
- 把75% KV缓存下降写成75%总显存下降。
- 将5.7T公开checkpoint成绩与1.4T公平架构对比直接拼接。
- 将3B激活参数误当成3B总参数或3B权重存储成本。
- 将当前FLA的新变体和新API倒写成原模型的默认行为。

## 7. 2026 年后续工作追加核验

本节按2026-10-13的当前时间核验。后续工作不能仅因基础报告是2025年发表，就被判为不存在。

### Kimi K3：已经找到官方证据

- `arXiv:2607.24653` 对应 **Kimi K3: Open Frontier Intelligence**，不是Complex KDA；v1提交于2026-07-27，v2为2026-08-07。摘要说明2.8T总参数、104B激活、原生视觉、1M上下文、KDA与AttnRes。[arXiv记录](https://arxiv.org/abs/2607.24653)
- 官方仓库模型表写93层，由69个KDA与24个Gated MLA组成；2.8T是总参数而非每token激活参数。它还加入Stable LatentMoE与视觉编码器，不能说只是把48B Kimi Linear直接放大。[MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3#2-model-summary)
- 官方权重入口可由论文脚注追到[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)，厂商介绍另见[Kimi模型页](https://www.kimi.ai/zh-hant/ai-models/kimi-k3)。模型开放权重不自动意味着完整训练数据与训练代码开放；许可也须单独阅读，不能继承48B模型的MIT结论。

K3与原始Kimi Linear最值得明确的差异：其写入系数仍为 `β=sigmoid(Wβx)∈(0,1)`；遗忘log门改为

$$
g=-5\sigma(e^{A_h}z),\quad\alpha=e^g\in(e^{-5},1).
$$

这给遗忘量一个下界，使16-token小块的倒数衰减缩放可留在BF16动态范围内，便于Tensor Core矩阵乘法。KDA输出门改为全秩，MLA也有输出门；骨干末尾额外放一个Gated MLA层。来源：[K3论文§2.1式(2)、式(5)–(7)](https://arxiv.org/html/2607.24653v2#S2.SS1)。所以「safe sigmoid」是K3时代的具体改进；不能替换2025原始模型的negative-softplus叙述。

### Complex KDA：已经找到作者论文、代码与权重

- 正确论文是 `arXiv:2609.24797`，2026-09-21提交，作者为Julien Siems等；这是一篇后续研究，并非Moonshot原始Kimi Linear论文。[论文记录](https://arxiv.org/abs/2609.24797)
- 核心扩展是 `α_i∈[-1,1]` 与 `β∈[0,2]` **共同**启用，负通道门与delta变换组合可产生平面旋转。单位key条件下转移保持非扩张（向量长度不会因线性转移放大）；不能只看参数范围就忽略key归一化。[论文§3及附录A](https://arxiv.org/html/2609.24797v1#S3)
- 作者源码在[OpenEuroLLM/ComplexKDA](https://github.com/OpenEuroLLM/ComplexKDA)，模型集合包括纯CKDA、混合CKDA与sigmoid-KDA对照的1.3B/100B模型。[作者HF集合](https://huggingface.co/collections/openeurollm/complexkda)
- README列出群状态跟踪、周期音频与语言建模复现材料。`lm_scaling/reproduce_tables.sh --check`重建已存结果表，不等于重新训练；完整训练需要集群。文档中的clone示例使用旧的`automl/ComplexKDA`命名，使用前宜跟随当前仓库实际地址。[作者仓库说明](https://github.com/OpenEuroLLM/ComplexKDA#contents)

结果范围：论文的1.3B/100B-token语言模型与KDA相近，保留约96%–97%吞吐；状态跟踪上有长度外推优势，但可表达不等于一定能学到，作者明确指出标准随机初始化下A5构造未学成。这些不能写成「CKDA在任意LLM上全面超越KDA」。[论文§6–7](https://arxiv.org/html/2609.24797v1#S7)

**实现误解提醒**：负α不能直接放进原KDA的实数`log α`接口。论文附录E采用`log|α|`并把累计符号吸收到q/k坐标中，从而复用正幅值核函数。是否允许负特征值的β扩展，与是否实现有符号α，是两个不同开关。[Complex KDA附录E](https://arxiv.org/html/2609.24797v1#A5)

### 状态结论与尚未覆盖部分

K3 2.8T、论文2607.24653、Complex KDA论文2609.24797及其作者开源路径均已核验。Mira报告若含其他2026项目、具体API型号、价格、发布日期或商业评测数字，需要逐条查其官方证据；本次没有覆盖的内容应标「待核验」，不能因本次未检索就宣称不存在。K3和CKDA的所有性能数字在本文均是来源报告，未独立复测。

