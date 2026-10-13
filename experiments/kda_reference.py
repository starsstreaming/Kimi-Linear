"""KDA 教学参考：与 HTML 的递推相同，补全分块形式与返回状态。

布局为 B,H,T,D；g=log(alpha)，beta 为每头每步标量。这里不应用
1/sqrt(d_k) 输出缩放，也不包含卷积、投影或输出门，不是完整模型。
块内显式形成 C×C×d_k 张量，适合小规模验证，不是 GPU 优化 kernel。
"""
import torch


def _prepare(q, k, v, g, beta, initial_state):
    tensors = (q, k, v, g, beta)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise ValueError('Inputs must be torch tensors')
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError('Expected q,k: B,H,T,dk and v: B,H,T,dv')
    b, h, t, dk = q.shape
    if min(b, h, dk, v.shape[-1]) < 1 or t < 1:
        raise ValueError('All dimensions must be positive')
    if v.shape[:3] != (b, h, t) or beta.shape != (b, h, t):
        raise ValueError('Batch, head and sequence dimensions must match')
    if g.shape not in [(b, h, t, dk), (b, h, t, 1)]:
        raise ValueError('g must have dk channels or one broadcast channel')
    if not q.is_floating_point():
        raise ValueError('Floating point inputs required')
    if any(x.dtype != q.dtype or x.device != q.device for x in tensors):
        raise ValueError('All inputs must share dtype and device')
    if any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Inputs must be finite')
    if (g > 0).any() or ((beta < 0) | (beta > 1)).any():
        raise ValueError('Expected log(alpha)<=0 and beta in [0,1]')
    expected = (b, h, dk, v.shape[-1])
    if initial_state is None:
        state = q.new_zeros(expected)
    else:
        if (initial_state.shape != expected or initial_state.dtype != q.dtype
                or initial_state.device != q.device or not torch.isfinite(initial_state).all()):
            raise ValueError('Initial state must match shape, dtype and device and be finite')
        state = initial_state.clone()
    return g.expand_as(q), state


def kda_recurrent(q, k, v, g, beta, *, delta=True, initial_state=None):
    """逐 token 更新，返回 (outputs, final_state)，不修改调用方输入。"""
    g, state = _prepare(q, k, v, g, beta, initial_state)
    outputs = []
    for index in range(q.shape[2]):
        state = state * g[:, :, index].exp().unsqueeze(-1)
        key = k[:, :, index].unsqueeze(-1)
        value = v[:, :, index].unsqueeze(-2)
        prediction = (state * key).sum(-2, keepdim=True) if delta else 0
        state = state + beta[:, :, index, None, None] * key * (value - prediction)
        outputs.append((state * q[:, :, index].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=2), state


def kda_chunkwise(q, k, v, g, beta, *, chunk=16, delta=True, initial_state=None):
    """分块等价形式；支持尾块，因果区间先遮罩再 exp，避免形成 1/Gamma。

    教学实现直接计算 exp(G_i-G_j)，只在 j<=i 的因果区间取值。
    比原 HTML 的 exp(G_i)*exp(-G_j) 更稳定，但不可据此推断 GPU 速度。
    """
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError('chunk must be a positive integer')
    g, state = _prepare(q, k, v, g, beta, initial_state)
    outputs = []
    for start in range(0, q.shape[2], chunk):
        stop = min(start + chunk, q.shape[2])
        qc, kc, vc = (x[:, :, start:stop] for x in (q, k, v))
        bc = beta[:, :, start:stop]
        prefix = g[:, :, start:stop].cumsum(dim=-2)
        length = stop - start
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        # 非因果差值可能为正且很大：必须在 exp 前清零，之后再置零权重。
        differences = prefix.unsqueeze(-2) - prefix.unsqueeze(-3)
        differences = torch.where(causal[..., None], differences, 0.)
        decay = differences.exp() * causal[..., None]
        qk = torch.einsum('bhiu,bhju,bhiju->bhij', qc, kc, decay)
        qg = qc * prefix.exp()
        if delta:
            kk = torch.einsum('bhiu,bhju,bhiju->bhij', kc, kc, decay).tril(-1)
            system = torch.eye(length, dtype=q.dtype, device=q.device) + bc.unsqueeze(-1) * kk
            rhs = bc.unsqueeze(-1) * (vc - (kc * prefix.exp()) @ state)
            residual = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        else:
            residual = bc.unsqueeze(-1) * vc
        outputs.append(qg @ state + qk @ residual)
        last = prefix[:, :, -1]
        weighted_key = kc * (last.unsqueeze(-2) - prefix).exp()
        state = last.exp().unsqueeze(-1) * state + weighted_key.transpose(-1, -2) @ residual
    return torch.cat(outputs, dim=2), state
