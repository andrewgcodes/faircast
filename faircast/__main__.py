"""Self-test: ``python -m faircast``. Checks unbiasedness of stochastic rounding, the dead zone of
round-to-nearest, and that error feedback carries sub-half-ulp updates through to the served copy."""
import torch

from .cast import ef_cast_bf16, sr_cast_bf16


def main() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(200_000) * 0.02).to(torch.bfloat16).float()  # on the bf16 grid, as the master is at init
    w[::2] *= -1
    ulp = torch.exp2(torch.frexp(w.abs())[1].float() - 8)  # exact bf16 spacing at w

    n = 400
    acc = torch.zeros_like(w)
    for _ in range(n):
        acc += sr_cast_bf16(w, g).float()
    bias = (acc / n - w) / ulp
    maxerr = ((sr_cast_bf16(w, g).float() - w).abs() / ulp).max()
    print(f"SR:      mean bias / ulp = {bias.mean():+.4f} (expect ~0); max |q-w| / ulp = {maxerr:.3f} (expect <= 1)")

    delta = 0.05 * ulp
    w2 = w + delta
    frac_nearest = (w2.to(torch.bfloat16) != w.to(torch.bfloat16)).float().mean()
    frac_sr = (sr_cast_bf16(w2, g) != w.to(torch.bfloat16)).float().mean()
    print(f"nearest: fraction of served weights changed after a +0.05 ulp update = {frac_nearest:.4f} (dead zone)")
    print(f"SR:      fraction changed = {frac_sr:.4f} (expect ~0.05)")

    r = torch.zeros_like(w)
    m = w.clone()
    for _ in range(40):
        m += delta
        q = ef_cast_bf16(m, r)
    moved_ef = ((q.float() - w.to(torch.bfloat16).float()) / ulp).mean()
    moved_nearest = ((m.to(torch.bfloat16).float() - w.to(torch.bfloat16).float()) / ulp).mean()
    print(f"EF:      after 40 x 0.05 ulp steps the served copy moved {moved_ef:.3f} ulp on average (expect ~2); nearest shows {moved_nearest:.3f}")


if __name__ == "__main__":
    main()
