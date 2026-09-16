"""다음에 무엇을 더 만들어야 하나 — 항체를 늘릴까, 항체당 링커를 늘릴까.

7절의 검정력 기계를 그대로 써서 설계 대안들의 검출률을 잰다.
답은 직관과 다를 수 있다: 쌍 수는 링커 수에 대해 **제곱으로** 늘지만
(항체당 C(k,2)), 항체 수에 대해서는 **선형으로**만 는다.

돌리는 법:  python3 tools/design_power.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvml


def power(n_grp, n_lk, beta, n_sim=60, n_perm=150, seed=0):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame([dict(항체=f"Ab{b}", 링커=f"L{i}")
                      for b in range(n_grp) for i in range(n_lk)])
    F["x"] = rng.normal(0, 1, len(F))
    D = fvml.Design(F, block="항체")
    z = ((F.x - F.x.mean()) / F.x.std()).values
    hit = 0
    for _ in range(n_sim):
        y = D.center(beta * z + rng.normal(0, 1, len(z)))
        hit += int(fvml.perm_test_signal(z[:, None], y, D, n_perm=n_perm,
                                         seed=int(rng.integers(1 << 30)))["순열p"] < 0.05)
    return hit / n_sim, len(D.pairs), len(F)


PLANS = [("지금:   항체5 × 링커4", 5, 4),
         ("링커+2: 항체5 × 링커6", 5, 6),
         ("링커+4: 항체5 × 링커8", 5, 8),
         ("항체+3: 항체8 × 링커4", 8, 4),
         ("항체+7: 항체12 × 링커4", 12, 4)]

if __name__ == "__main__":
    print("설계 대안별 검출률 (β = 특징 1 SD 당 타깃 σ)")
    print(f"{'설계':<24}{'구성체':>6}{'쌍':>6}   β=0.5   β=1.0")
    print("-" * 56)
    for label, g, k in PLANS:
        r = [power(g, k, b)[0] for b in (0.5, 1.0)]
        _, npair, n = power(g, k, 0.0, n_sim=1, n_perm=1)
        print(f"{label:<24}{n:>6}{npair:>6}   {r[0]:>5.2f}   {r[1]:>5.2f}")
    print()
    print("쌍 수 = 항체 × C(링커,2) — 링커에는 제곱으로, 항체에는 선형으로 는다.")
