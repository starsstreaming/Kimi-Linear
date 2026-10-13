"""真实数学行为测试：能发现更新顺序、因果性、尾块与梯度错误。"""
import unittest
import torch
from experiments.kda_reference import kda_recurrent, kda_chunkwise


class KdaTests(unittest.TestCase):
    def inputs(self, length=7, scalar=False):
        generator = torch.Generator().manual_seed(42)
        def rand(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64)
        q = torch.nn.functional.normalize(rand(2, 2, length, 3), dim=-1)
        k = torch.nn.functional.normalize(rand(2, 2, length, 3), dim=-1)
        v = rand(2, 2, length, 2)
        g = -torch.sigmoid(rand(2, 2, length, 1 if scalar else 3))
        beta = torch.sigmoid(rand(2, 2, length))
        return q, k, v, g, beta

    def test_decay_precedes_delta_update(self):
        # 手算：旧状态(2,3)先衰减为(1,3)，投影擦除(2,2)，余(-1,1)。
        q = torch.tensor([[[[0., 1.]]]], dtype=torch.float64)
        k = torch.tensor([[[[2**-0.5, 2**-0.5]]]], dtype=torch.float64)
        v = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        g = torch.tensor([[[[0.5, 1.]]]], dtype=torch.float64).log()
        beta = torch.ones(1, 1, 1, dtype=torch.float64)
        state = torch.tensor([[[[2.], [3.]]]], dtype=torch.float64)
        for fn in [kda_recurrent, kda_chunkwise]:
            out, final = fn(q, k, v, g, beta, initial_state=state)
            torch.testing.assert_close(out, torch.ones_like(out))
            torch.testing.assert_close(final, torch.tensor([[[[-1.], [1.]]]], dtype=torch.float64))

    def test_delta_overwrites_same_key(self):
        q = k = torch.ones(1, 1, 2, 1, dtype=torch.float64)
        v = torch.tensor([[[[2.], [5.]]]], dtype=torch.float64)
        g = torch.zeros_like(q)
        beta = torch.ones(1, 1, 2, dtype=torch.float64)
        for fn in [kda_recurrent, kda_chunkwise]:
            out, _ = fn(q, k, v, g, beta)
            torch.testing.assert_close(out, v)
            added, _ = fn(q, k, v, g, beta, delta=False)
            torch.testing.assert_close(added, torch.tensor([[[[2.], [7.]]]], dtype=torch.float64))

    def test_chunk_tail_and_four_recurrence_variants(self):
        for scalar in [False, True]:
            for delta in [False, True]:
                args = self.inputs(scalar=scalar)
                expected = kda_recurrent(*args, delta=delta)
                for chunk in [1, 3, 4, 16]:
                    actual = kda_chunkwise(*args, chunk=chunk, delta=delta)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_state_can_resume_without_mutating_caller(self):
        args = self.inputs()
        out, state = kda_recurrent(*args)
        prefix = [x[:, :, :3] for x in args]
        suffix = [x[:, :, 3:] for x in args]
        for fn in [kda_recurrent, kda_chunkwise]:
            first, initial = fn(*prefix)
            copy = initial.clone()
            second, final = fn(*suffix, initial_state=initial)
            torch.testing.assert_close(torch.cat([first, second], dim=2), out)
            torch.testing.assert_close(final, state)
            torch.testing.assert_close(initial, copy)

    def test_future_values_cannot_change_prefix(self):
        args = self.inputs()
        actual, _ = kda_chunkwise(*args, chunk=4)
        changed = list(args)
        changed[2] = args[2].clone()
        changed[2][:, :, 3:] += 1000
        perturbed, _ = kda_chunkwise(*changed, chunk=4)
        torch.testing.assert_close(actual[:, :, :3], perturbed[:, :, :3])

    def test_large_decay_does_not_form_inverse_gamma(self):
        args = list(self.inputs(length=9))
        args[3] = torch.full_like(args[3], -100.)
        actual = kda_chunkwise(*args, chunk=8)
        expected = kda_recurrent(*args)
        for got, want in zip(actual, expected):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_chunk_and_recurrent_gradients_match(self):
        args = self.inputs(length=5)
        def gradient(fn):
            inputs = [x.clone().requires_grad_() for x in args]
            out, state = fn(*inputs)
            return torch.autograd.grad(out.square().sum() + state.square().sum(), inputs)
        for actual, expected in zip(gradient(kda_chunkwise), gradient(kda_recurrent)):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

    def test_invalid_inputs_fail_explicitly(self):
        args = list(self.inputs())
        for fn in [kda_recurrent, kda_chunkwise]:
            with self.assertRaises(ValueError):
                fn(*args[:-1], args[-1][:, :, :-1])
            with self.assertRaises(ValueError):
                fn(*args[:3], -args[3], args[4])
            invalid = args[2].clone()
            invalid[0, 0, 0, 0] = float('nan')
            with self.assertRaises(ValueError):
                fn(args[0], args[1], invalid, args[3], args[4])
        with self.assertRaises(ValueError):
            kda_chunkwise(*args, chunk=0)


if __name__ == '__main__':
    unittest.main()
