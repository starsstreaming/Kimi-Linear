# HTML 代码片段原文

> 这些是片段，不是完整可运行源码；修复后的教学算子见 experiments/kda_reference.py。

## 片段 1

```python
def kda_recurrent(q, k, v, g, beta, delta=True):
    """逐 token 递推，O(T·dk·dv)，用于验证分块实现的正确性。
    q,k : (B,H,T,dk)   v : (B,H,T,dv)
    g   : (B,H,T,dk) = log α   (≤0)
    beta: (B,H,T)
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv)
    out = q.new_zeros(B, H, T, dv)
    for t in range(T):
        a = g[:, :, t].exp().unsqueeze(-1)          # (B,H,dk,1) 每通道衰减
        S = S * a                                   # Diag(α_t) S_{t-1}
        kt = k[:, :, t].unsqueeze(-1)               # (B,H,dk,1)
        vt = v[:, :, t].unsqueeze(-2)               # (B,H,1,dv)
        bt = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)
        if delta:
            # delta rule: 用"预测误差"(v - S^T k) 做秩-1 修正
            pred = (S * kt).sum(-2, keepdim=True)   # (B,H,1,dv) = S^T k_t
            S = S + bt * kt * (vt - pred)
        else:
            # 无 delta rule（Mamba2 / GLA 风格）: 直接累加
            S = S + bt * kt * vt
        out[:, :, t] = (S * q[:, :, t].unsqueeze(-1)).sum(-2)
    return out, S
```

## 片段 2

```python
def kda_chunkwise(q, k, v, g, beta, chunk=64, delta=True, return_state=False):
    """论文 §3.1 的 chunkwise 算法：块间递推传状态，块内稠密矩阵并行。"""
    B, H, T, dk = q.shape; dv = v.shape[-1]
    NT, C = T // chunk, chunk
    r = lambda x, d: x.reshape(B, H, NT, C, d)
    q, k, v, g = r(q, dk), r(k, dk), r(v, dv), r(g, dk)
    beta = beta.reshape(B, H, NT, C)

    gc = g.cumsum(-2)                       # log γ^{1→r}，块内累积衰减
    Kg = k * gc.exp()                       # Γ ⊙ K
    Kd = k * (-gc).exp()                    # K / Γ   ← 数值不稳定的源头
    Qg = q * gc.exp()                       # Γ ⊙ Q

    if delta:
        # Eq.6: M = (I + StrictTril(Diag(β)(Γ⊙K)(K/Γ)^T))^{-1} Diag(β)
        Akk = (Kg @ Kd.transpose(-1, -2)) * tril_strict
        N = beta.unsqueeze(-1) * Akk
        M = inv_unit_lower(N) * beta.unsqueeze(-2)
    else:
        M = torch.diag_embed(beta)          # 无 delta rule 时退化为对角

    W = M @ Kg                              # Eq.7
    U = M @ v
    Aqk = (Qg @ Kd.transpose(-1, -2)) * tril_diag   # 块内打分

    S = q.new_zeros(B, H, dk, dv)
    for t in range(NT):                     # 块间：递推
        gC = gc[:, :, t][:, :, -1]          # log γ^{1→C}
        decay_C = gC.exp()
        Ut_eff = U[:, :, t] - W[:, :, t] @ S if delta else U[:, :, t]
        o = Qg[:, :, t] @ S + Aqk[:, :, t] @ Ut_eff      # 块间 + 块内
        Kc = k[:, :, t] * (gC.unsqueeze(-2) - gc[:, :, t]).exp()
        S = decay_C.unsqueeze(-1) * S + Kc.transpose(-1, -2) @ Ut_eff
    return torch.stack(outs, 2).reshape(B, H, T, dv)
```

## 片段 3

```python
# 统一的四种变体 —— 只切两个开关，就能把「细粒度门控」与「delta 规则」拆开实验：
kda_chunkwise(q, k, v, g_channel, beta, delta=True)   # KDA     （本文）
kda_chunkwise(q, k, v, g_scalar,  beta, delta=True)   # GDN     （Gated DeltaNet）
kda_chunkwise(q, k, v, g_channel, beta, delta=False)  # GLA     （Gated Linear Attention）
kda_chunkwise(q, k, v, g_scalar,  beta, delta=False)  # Mamba2  （仅标量衰减）
```

## 片段 4

```python
class MixerLayer(nn.Module):
    """论文 §4 的神经参数化：ShortConv + Swish + L2Norm，低秩门控，Sigmoid 输出门"""
    def __init__(self, d_model, n_heads, gate="channel", delta=True, chunk=16):
        ...
        self.a_down = nn.Linear(d_model, self.dk, bias=False)   # 低秩下降投影
        self.a_up   = nn.Linear(self.dk, n_gate)                # 上升投影
        nn.init.constant_(self.a_up.bias, -4.0)  # 初始 α≈0.98（先记得久一点）

    def forward(self, x):
        qkv = torch.cat([self.wq(x), self.wk(x), self.wv(x)], -1)
        qkv = self.conv(qkv.transpose(1, 2))[..., :T].transpose(1, 2)  # 因果短卷积
        q, k, v = qkv.chunk(3, -1)
        q = F.normalize(F.silu(q), dim=-1)   # Swish + L2Norm：稳定特征值
        k = F.normalize(F.silu(k), dim=-1)
        a = self.a_up(self.a_down(x))
        g = -F.softplus(a).clamp(max=1.0)    # log α ∈ [-1,0]：块内 1/Γ ≤ e^16，float32 安全
        beta = torch.sigmoid(self.wbeta(x))  # Δ 规则的"学习率"
        o = kda_chunkwise(q, k, v, g, beta, chunk=self.chunk, delta=self.delta)
        o = o * torch.sigmoid(self.g_up(self.g_down(x)))   # Sigmoid 输出门（缓解 attention sink）
        return self.wo(o), g
```
