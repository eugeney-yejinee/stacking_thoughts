"""계면 상호작용 검정의 검정력 — 항체를 늘릴까, 항체당 링커를 늘릴까.

★ 주효과(7절 ①)와 **반대의 답**이 나온다.
  주효과는 항체 안 쌍이 통계량이라 링커에 제곱으로 는다.
  상호작용의 정보는 **항체별 기울기의 흩어짐**에 있으므로, 사실상 항체 수가
  그대로 표본 수다. 링커를 늘리면 각 기울기가 정밀해질 뿐 기울기 개수는 안 는다.

돌리는 법:  python3 tools/iface_power.py
"""
import os
import sys
from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvml


def make(n_ab, n_lk, seed=0):
    rows = [dict(항체=f"Ab{a}", 링커=f"L{i}", 길이=10 + 3 * i)
            for a in range(n_ab) for i in range(n_lk)]
    F = pd.DataFrame(rows)
    D = fvml.Design(F, block="항체")
    rng = np.random.default_rng(seed)
    # 열림분율은 항체 안에서 길이를 따라 변한다. 계면강도는 항체마다 상수.
    op = 0.02 + 0.012 * (F.길이 - 10) + rng.normal(0, .004, len(F))
    S = np.repeat(rng.uniform(0.70, 0.95, n_ab), n_lk)
    z = lambda v: (np.asarray(v, float) - np.mean(v)) / np.std(v)
    return F, D, z(op), z(S)


def slope_corr_stat(y, zo, zs, D):
    """항체별 기울기와 계면강도의 상관. 상호작용에 대한 직접 통계량이다.

    Δ일치도보다 이쪽이 강한 이유: 일치도는 **부호만** 보므로 기울기의 크기를 버린다.
    상호작용의 신호는 바로 그 크기 차이에 들어 있다.
    """
    sl, ss = [], []
    for b in D.blocks:
        x = zo[b] - zo[b].mean()
        if np.std(x) < 1e-12:
            continue
        sl.append(float(np.polyfit(x, y[b] - y[b].mean(), 1)[0]))
        ss.append(float(zs[b][0]))
    if len(sl) < 3:
        return np.nan
    if np.std(sl) < 1e-12 or np.std(ss) < 1e-12:
        return np.nan
    return float(np.corrcoef(sl, ss)[0, 1])


def exact_p(y, zo, zs, D, one_sided=True, max_exact=5040):
    """항체 수준 계면강도를 **항체끼리 재배치**하는 정확 순열.

    H0: 계면강도는 그 항체의 링커→물성 기울기와 무관하다.
    항체 안의 링커 배열은 **건드리지 않는다** — 기울기 자체는 관측된 그대로 두고,
    그 기울기에 어느 계면강도가 붙느냐만 끊는 것이 이 귀무가설이다.
    """
    k = len(D.blocks)
    vals = np.array([zs[b][0] for b in D.blocks])
    obs = slope_corr_stat(y, zo, zs, D)
    if not np.isfinite(obs):
        return np.nan, np.nan, 0
    perms = (list(permutations(range(k))) if factorial(k) <= max_exact
             else [np.random.default_rng(s).permutation(k) for s in range(max_exact)])
    null = []
    for pm in perms:
        zp = zs.copy()
        for t, b in enumerate(D.blocks):
            zp[b] = vals[pm[t]]
        v = slope_corr_stat(y, zo, zp, D)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    p = ((np.sum(null <= obs) + 1) / (len(null) + 1) if one_sided
         else (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
    return obs, float(p), len(null)


def power(n_ab, n_lk, gamma, n_sim=120, sigma=1.0, seed=0):
    F, D, zo, zs = make(n_ab, n_lk, seed)
    rng = np.random.default_rng(seed + 7)
    hit = 0
    for _ in range(n_sim):
        # ★ 부호: 계면이 셀수록 같은 열림이 더 많은 응집으로 간다 → y(클수록 좋음)에서 음수
        y = D.center(-1.0 * zo - gamma * (zs * zo) + rng.normal(0, sigma, len(zo)))
        _, p, _ = exact_p(y, zo, zs, D, one_sided=True)
        hit += int(np.isfinite(p) and p <= 0.05)
    return hit / n_sim


PLANS = [("지금:   항체5 × 링커4", 5, 4),
         ("링커+2: 항체5 × 링커6", 5, 6),
         ("항체+3: 항체8 × 링커4", 8, 4),
         ("항체+7: 항체12 × 링커4", 12, 4)]

if __name__ == "__main__":
    print("계면 상호작용 검정력 (γ = 계면강도 1 SD 당 기울기 변화, 단측 α=0.05)")
    print(f"{'설계':<24}{'구성체':>6}{'순열가짓수':>10}   γ=0    γ=2    γ=4")
    print("-" * 66)
    for label, a, k in PLANS:
        F, D, zo, zs = make(a, k)
        npm = min(factorial(a), 5040)
        r = [power(a, k, g) for g in (0.0, 2.0, 4.0)]
        print(f"{label:<24}{a*k:>6}{npm:>10}   "
              + "  ".join(f"{x:>5.2f}" for x in r))
    print()
    print("★ 주효과와 답이 반대다. 상호작용의 정보는 **항체별 기울기의 흩어짐**에 있어서,")
    print("  사실상 항체 수가 표본 수다. 링커를 늘리면 기울기가 정밀해질 뿐 개수는 안 는다.")
