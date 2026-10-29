import torch
from torch.func import functional_call

from faircast import ef_cast_bf16, export_bf16, served_forward, sr_cast_bf16, ste_params, sync_to_sampler


def _grid(n=100_000, scale=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, generator=g) * scale).to(torch.bfloat16).float()
    w[::2] *= -1
    _, e = torch.frexp(w.abs())
    ulp = torch.exp2(e.float() - 8)  # exact bf16 spacing at w
    return w, ulp


def test_sr_is_unbiased_and_within_one_ulp():
    w, ulp = _grid()
    x = w + 0.37 * ulp
    g = torch.Generator().manual_seed(1)
    n = 400
    acc = torch.zeros_like(w)
    for _ in range(n):
        q = sr_cast_bf16(x, g)
        assert ((q.float() - x).abs() / ulp).max() <= 1.0 + 1e-6
        acc += q.float()
    bias = ((acc / n - x) / ulp).mean().item()
    assert abs(bias) < 0.01, bias


def test_sr_flips_with_fractional_probability_where_nearest_is_dead():
    w, ulp = _grid()
    x = w + 0.05 * ulp
    g = torch.Generator().manual_seed(2)
    assert (x.to(torch.bfloat16) != w.to(torch.bfloat16)).float().mean().item() == 0.0
    frac = (sr_cast_bf16(x, g) != w.to(torch.bfloat16)).float().mean().item()
    assert 0.04 < frac < 0.06, frac


def test_sr_handles_signs_zero_and_nonfinite():
    g = torch.Generator().manual_seed(3)
    x = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 3.4e38], dtype=torch.float32)
    q = sr_cast_bf16(x, g).float()
    assert q[0] == 0 and q[1] == 0 and q[2] == 1 and q[3] == -1
    assert torch.isinf(q[4]) and q[4] > 0 and torch.isinf(q[5]) and q[5] < 0 and torch.isnan(q[6])
    assert torch.isfinite(q[7]) or torch.isinf(q[7])


def test_sr_rejects_non_fp32():
    try:
        sr_cast_bf16(torch.zeros(4, dtype=torch.bfloat16))
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def _spacing(x):
    """Exact bf16 grid spacing at x (8 significand bits incl. the implicit one): x = m * 2**e, m in [0.5, 1) -> 2**(e - 8)."""
    _, e = torch.frexp(x.abs())
    return torch.exp2(e.float() - 8)


def test_ef_carries_subthreshold_updates_and_bounds_residual():
    w, ulp = _grid()
    m = w.clone()
    r = torch.zeros_like(w)
    for step in range(1, 41):
        m += 0.05 * ulp
        y = m + r  # the value that gets rounded this step
        q = ef_cast_bf16(m, r)
        assert ((r.abs() / _spacing(y)) <= 0.5 + 1e-6).all()
    moved = ((q.float() - w) / ulp).mean().item()
    assert 1.9 < moved < 2.1, moved


def test_ef_time_average_is_unbiased_where_nearest_is_stuck():
    # master parked 0.3 ulp above a grid point: nearest serves the grid point forever (constant -0.3 ulp bias);
    # EF alternates between the two neighbours so the served copy equals the master on average over time.
    w, ulp = _grid()
    m = w + 0.3 * ulp * w.sign()  # away from zero so the step stays inside one binade
    r = torch.zeros_like(w)
    acc = torch.zeros_like(w)
    T = 100
    for _ in range(T):
        acc += ef_cast_bf16(m, r).float()
    assert ((m.to(torch.bfloat16).float() - w) / ulp).abs().max().item() == 0.0
    assert ((acc / T - m) / ulp).abs().max().item() <= 0.5 / T + 1e-3


def test_ef_error_is_bounded_and_does_not_accumulate():
    w, ulp = _grid(n=1000)
    m = w.clone()
    r = torch.zeros_like(w)
    g = torch.Generator().manual_seed(4)
    y_prev = m.clone()
    for _ in range(200):
        m += torch.randn(m.shape, generator=g) * 0.2 * ulp
        r_prev = r.clone()
        y = m + r
        q = ef_cast_bf16(m, r)
        assert torch.allclose(q.float() + r, m + r_prev, atol=0, rtol=1e-5)  # q_t + r_t = theta_t + r_{t-1}
        bound = _spacing(y_prev) / 2 + _spacing(y) / 2  # |q_t - theta_t| = |r_{t-1} - r_t| <= u
        assert ((q.float() - m).abs() <= bound * (1 + 1e-6)).all()
        y_prev = y


def test_sync_to_sampler_and_export_dispatch():
    master = {"a": torch.randn(64), "b": torch.randn(8, 8)}
    served = {k: torch.empty_like(v, dtype=torch.bfloat16) for k, v in master.items()}
    sync_to_sampler(master, served, mode="nearest")
    for k in master:
        assert torch.equal(served[k], master[k].to(torch.bfloat16))
    resid = {k: torch.zeros_like(v) for k, v in master.items()}
    sync_to_sampler(master, served, mode="ef", resid=resid)
    for k in master:
        assert torch.allclose(served[k].float() + resid[k], master[k])
    g = torch.Generator().manual_seed(0)
    sync_to_sampler(master, served, mode="sr", gen=g)
    for k in master:
        assert ((served[k].float() - master[k]).abs() <= master[k].abs() * 2 ** -7 + 1e-6).all()
    try:
        export_bf16(master["a"], mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_ste_params_match_served_and_route_gradient_to_master():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 4)
    served = {n: sr_cast_bf16(p.data) for n, p in model.named_parameters()}
    params = ste_params(model, served)
    for n, p in params.items():
        assert torch.equal(p, served[n])
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    out = functional_call(model, params, (x,))
    out.float().sum().backward()
    for _, p in model.named_parameters():
        assert p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()


def test_served_forward_swaps_and_restores_master():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 4)
    master_before = {n: p.data.clone() for n, p in model.named_parameters()}
    served = {n: sr_cast_bf16(p.data) for n, p in model.named_parameters()}
    x = torch.randn(3, 16)
    with served_forward(model, served):
        for n, p in model.named_parameters():
            assert torch.equal(p.data, served[n].float())
        model(x).sum().backward()
    for n, p in model.named_parameters():
        assert torch.equal(p.data, master_before[n])
        assert p.grad is not None
