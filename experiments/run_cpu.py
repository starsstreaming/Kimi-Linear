"""CPU 数值复核；生成原始 JSON，不将算术估算当作 GPU 测速。"""
import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiments.kda_reference import kda_chunkwise, kda_recurrent


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/local/cpu-check.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = []
    for dtype in [torch.float64, torch.float32]:
        tolerance = 1e-11 if dtype == torch.float64 else 1e-5
        for seed in [0, 1, 2]:
            gen = torch.Generator().manual_seed(seed)
            shape = (2, 2, 65, 8)
            q = torch.nn.functional.normalize(torch.randn(shape, generator=gen, dtype=dtype), dim=-1)
            k = torch.nn.functional.normalize(torch.randn(shape, generator=gen, dtype=dtype), dim=-1)
            v = torch.randn(2, 2, 65, 6, generator=gen, dtype=dtype)
            channel_g = -0.2 * torch.rand(shape, generator=gen, dtype=dtype)
            beta = torch.rand(2, 2, 65, generator=gen, dtype=dtype)
            for scalar in [False, True]:
                g = channel_g.mean(-1, keepdim=True) if scalar else channel_g
                for delta in [True, False]:
                    variant = ('scalar' if scalar else 'channel') + ('-delta' if delta else '-additive')
                    expected, expected_state = kda_recurrent(q, k, v, g, beta, delta=delta)
                    for chunk in [1, 8, 16, 32, 64]:
                        actual, state = kda_chunkwise(q, k, v, g, beta, chunk=chunk, delta=delta)
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(state, expected_state, atol=tolerance, rtol=tolerance)
                        records.append({'dtype': str(dtype), 'seed': seed, 'variant': variant,
                                        'shape_q': list(shape), 'dv': 6, 'chunk': chunk,
                                        'output_max_abs_error': (actual - expected).abs().max().item(),
                                        'state_max_abs_error': (state - expected_state).abs().max().item()})
    log_max = math.log(torch.finfo(torch.float32).max)
    overflow = [{'alpha': alpha, 'first_integer_chunk_exceeding_float32': math.floor(log_max / -math.log(alpha)) + 1}
                for alpha in [0.99, 0.9]]
    # 这里只按层数估算随 token 增长的 MLA 缓存，不含状态和其他内存。
    result = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'educational recurrence correctness; not full-model or GPU performance reproduction',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'platform': platform.system(), 'device_used': 'cpu', 'threads': torch.get_num_threads()},
              'reference_sha256': hashlib.sha256(Path('experiments/kda_reference.py').read_bytes()).hexdigest(),
              'cases': records, 'float32_inverse_gamma_overflow_thresholds': overflow,
              'ideal_token_kv_only': {'mla_layers': 7, 'total_layers': 27,
                                     'remaining_fraction': 7 / 27, 'reduction_fraction': 20 / 27},
              'not_run': ['MQAR training', 'DPLR speed comparison', 'GPU kernels', '48B model quality', 'SFT/RL']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Passed {len(records)} equivalence cases; wrote {args.output}')
    for dtype in ['torch.float64', 'torch.float32']:
        error = max(x['output_max_abs_error'] for x in records if x['dtype'] == dtype)
        print(f'{dtype}: maximum output absolute error {error:.3e}')


if __name__ == '__main__':
    run()
