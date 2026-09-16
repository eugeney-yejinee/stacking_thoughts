import sys
"""fvinterx.py 자체 시험 — 계면 상호작용 검정이 정직한가.

돌리는 법:  python3 tools/test_fvinterx.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvinterx as fx


def make(m=5, k=4, gamma=0.0, rho_SC=0.0, curved=False, sigma=1.0, seed=0,
         spread=0.35):
    """항체 m 개 × 링커 k 종.

    rho_SC 는 S 와 **항체별 평균 열림분율(작동점)** 의 상관 — 이게 교환가능성을
    깨는 교란이다. ★ 작동점 교란을 재현하려면 항체 안 흩어짐(spread)이 항체 간
    작동점 차이보다 **작아야** 한다. 크면 각 항체가 같은 x 범위를 훑어서
    곡률이 항체마다 다른 기울기를 못 만든다 (처음 쓴 시험이 그래서 무력했다).
    """
    rng = np.random.default_rng(seed)
    S = rng.normal(size=m); S -= S.mean(); S /= S.std()
    # 센 계면일수록 덜 열린다 → 작동점이 S 와 음의 상관
    C = rho_SC * (-S) + math.sqrt(max(1 - rho_SC**2, 0)) * rng.normal(size=m)
    C = C * 1.5                                  # 항체 간 작동점 차이를 키운다
    o = C[:, None] + spread * rng.normal(size=(m, k))
    slope = 1.0 + gamma * S
    Y = slope[:, None] * o
    if curved:
        Y = Y + 1.2 * o**2                       # 곡률 — 로짓이 만드는 그것
    Y = Y + rng.normal(0, sigma, (m, k))
    Y = Y - Y.mean(1, keepdims=True)
    g = np.repeat(np.arange(m), k)
    T = fx.antibody_slopes(Y.ravel(), o.ravel(), g)
    return T, S, T.mean_x.values


def test_variance_formula():
    """Var(γ̂) = 1/Σ w_j d_j² 가 맞는가 — S 와 o 를 고정하고 y 만 다시 뽑는다."""
    rng = np.random.default_rng(0)
    m, k = 5, 4
    S = rng.normal(size=m); S -= S.mean()
    o = rng.normal(size=(m, k))
    g = np.repeat(np.arange(m), k)
    gs = []
    for _ in range(4000):
        Y = (1.0 + 2.0 * S)[:, None] * o + rng.normal(0, 1.0, (m, k))
        Y -= Y.mean(1, keepdims=True)
        T = fx.antibody_slopes(Y.ravel(), o.ravel(), g)
        gs.append(fx.interaction_test(T, S, max_exact=1)["gamma"])
    emp = float(np.var(gs, ddof=1))
    T = fx.antibody_slopes(((1 + 2*S)[:, None]*o).ravel(), o.ravel(), g)
    # 이론 분산: σ²=1 이므로 w_j = Sxx_j
    d = S - S.mean()
    sxx = np.array([float(((o[j] - o[j].mean())**2).sum()) for j in range(m)])
    theo = 1.0 / float((sxx * d * d).sum())
    print(f"  OK  분산식 — 경험 {emp:.4f} vs 이론 {theo:.4f} (비 {emp/theo:.3f})")
    assert 0.8 < emp/theo < 1.25, f"분산식이 안 맞는다 {emp/theo}"


def _fpr(rho_SC, curved, covariate, n=400, m=5, k=4, sided="two"):
    """★ 양측으로 잰다. 교란이 만드는 가짜 상관은 부호가 정해져 있어서, 단측으로
    재면 반대 방향일 때 0 이 나와 '교란이 없다' 고 오해한다."""
    hit = 0
    for s in range(n):
        T, S, C = make(m, k, gamma=0.0, rho_SC=rho_SC, curved=curved, seed=1000+s)
        r = fx.interaction_test(T, S, covariate=(C if covariate else None), sided=sided)
        hit += int(r["순열p"] <= 0.05)
    return hit/n


def test_fpr_clean():
    f = _fpr(0.0, False, False)
    print(f"  OK  거짓양성(교란 없음, 통제 없음) = {f:.3f} (명목 0.05)")
    assert f < 0.12, f


def test_operating_point_confound_is_real():
    """★ 검토가 지적한 치명적 교란 — 통제 없이는 거짓양성이 폭발한다."""
    bad = _fpr(0.9, True, False)
    print(f"  OK  거짓양성(작동점 교란 ρ=0.9 + 곡률, **통제 없음**) = {bad:.3f}")
    assert bad > 0.10, f"교란이 재현되지 않았다 — 시험이 무의미하다 ({bad})"


def test_freedman_lane_repairs_it():
    """Freedman–Lane 로 평균 열림분율을 통제하면 명목으로 돌아와야 한다."""
    fixed = _fpr(0.9, True, True)
    print(f"  OK  거짓양성(같은 조건, **Freedman–Lane 통제**) = {fixed:.3f}")
    assert fixed < 0.12, f"통제가 안 먹었다 {fixed}"


def test_power_recovers_planted():
    """★ 항체 **안** 열림분율 범위(spread)가 검정력을 좌우한다.
    범위가 좁으면 기울기 자체가 안 잡혀서 상호작용은 볼 수도 없다.
    실제 패널에서 링커가 열림분율을 얼마나 벌려 놓는지가 이 검정의 생사를 정한다."""
    for spread, name in ((1.0, "넓음"), (0.35, "좁음")):
        hit = 0
        for s in range(200):
            T, S, C = make(5, 4, gamma=2.0, spread=spread, seed=2000+s)
            hit += int(fx.interaction_test(T, S, covariate=C,
                                           sided="two")["순열p"] <= 0.05)
        print(f"  OK  검출률 (γ=2, 통제 포함, 항체 안 범위 {name} spread={spread}) "
              f"= {hit/200:.3f}")
        if spread >= 1.0:
            assert hit/200 > 0.3, f"넓은 범위에서도 못 찾는다 {hit/200}"
        else:
            assert hit/200 < 0.4, "좁은 범위인데 검정력이 안 떨어졌다 — 시험이 무의미하다"
    print("      → 링커가 열림분율을 충분히 벌려 놓지 않으면 이 검정은 못 돈다.")


def test_attenuation_is_product_of_reliabilities():
    """γ̂/γ = λ_S · λ_O 가 정확한가 — 상호작용은 **두 신뢰도의 곱**으로 눌린다."""
    for lamS, lamO in ((1.0, 1.0), (1.0, 0.7), (0.7, 1.0), (0.7, 0.7)):
        rng = np.random.default_rng(7)
        m, k, gtrue = 5, 4, 2.0
        gs = []
        for _ in range(600):
            S = rng.normal(size=m); S -= S.mean()
            o = rng.normal(size=(m, k))
            Y = (1.0 + gtrue*S)[:, None]*o + rng.normal(0, 1.0, (m, k))
            Y -= Y.mean(1, keepdims=True)
            oo = o + (rng.normal(0, math.sqrt((1-lamO)/lamO), (m, k)) if lamO < 1 else 0)
            So = S + (rng.normal(0, math.sqrt((1-lamS)/lamS), m) if lamS < 1 else 0)
            T = fx.antibody_slopes(Y.ravel(), oo.ravel(), np.repeat(np.arange(m), k))
            gs.append(fx.interaction_test(T, So, max_exact=1)["gamma"])
        ratio = float(np.mean(gs))/gtrue
        print(f"  OK  감쇠 λS={lamS} λO={lamO} → γ̂/γ = {ratio:.3f} "
              f"(예측 {lamS*lamO:.3f})")
        assert abs(ratio - lamS*lamO) < 0.22, f"{ratio} vs {lamS*lamO}"


def _make_raw(m=5, k=4, gamma=1.0, sigma=1.0, seed=0, spread=1.0):
    """두 검정을 **같은 자료**로 비교하려고 원자료까지 돌려준다."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=m); S -= S.mean(); S /= S.std()
    o = spread * rng.normal(size=(m, k))
    Y = (1.0 + gamma*S)[:, None]*o + rng.normal(0, sigma, (m, k))
    Y -= Y.mean(1, keepdims=True)
    g = np.repeat(np.arange(m), k)
    return Y.ravel(), o.ravel(), g, S


def test_gate_costs_power():
    """★ 이질성 F 를 **관문으로 쓰면** 검정력이 떨어진다 — 같은 자료로 비교한다.
    (어느 쪽이 더 센가는 설계에 따라 다르다. 중요한 건 게이트가 손해라는 것이다.)"""
    N = 250; hi = hh = both = 0
    for s in range(N):
        y, x, g, S = _make_raw(5, 4, gamma=1.0, seed=3000+s)
        T = fx.antibody_slopes(y, x, g)
        pi = fx.interaction_test(T, S, sided="two")["순열p"] <= 0.05
        ph = fx.slope_heterogeneity(y, x, g)
        ph = np.isfinite(ph["p"]) and ph["p"] <= 0.05
        hi += int(pi); hh += int(ph); both += int(pi and ph)
    print(f"  OK  γ=1 · 같은 자료  상호작용 {hi/N:.2f} · 이질성 F {hh/N:.2f} "
          f"· 게이트(둘 다) {both/N:.2f}")
    print(f"      → 게이트를 걸면 상호작용이 잡았을 것의 "
          f"{(hi-both)/max(hi,1):.0%} 를 잃는다. 관문으로 쓰지 마라.")
    assert both <= hi, "게이트가 단독보다 많이 잡았다 — 있을 수 없다"
    assert both < hi, "게이트가 아무 손해도 안 냈다 — 시험이 무의미하다"


def test_identifiability_report():
    # 작동점을 S 에 거의 완전히 묶어 놓고, 보고서가 그걸 잡는지 본다
    T, S, C = make(5, 4, gamma=0.0, rho_SC=1.0, spread=0.25, seed=11)
    R = fx.identifiability_report(T, S, {"평균열림": C})
    print("  OK  식별가능성 보고")
    print(R.to_string(index=False))
    r_ = abs(float(np.corrcoef(S, C)[0, 1]))
    print(f"      실제 |corr(S, 평균열림)| = {r_:.2f}")
    assert r_ > 0.9, f"픽스처가 교란을 안 만들었다 ({r_})"
    assert (R.판정.astype(str).str.contains("★")).any(), "교란을 하나도 안 잡았다"


if __name__ == "__main__":
    print("fvinterx 자체 시험")
    print("=" * 74)
    fails = 0
    for nm, fn in sorted((k, v) for k, v in list(globals().items())
                         if k.startswith("test_") and callable(v)):
        try:
            fn()
        except AssertionError as e:
            fails += 1; print(f"  ★ 실패 {nm}: {e}")
        except Exception as e:
            fails += 1
            import traceback; traceback.print_exc()
            print(f"  ★ 오류 {nm}: {type(e).__name__}: {e}")
    print("=" * 74)
    print("전부 통과" if not fails else f"{fails}건 실패")
    sys.exit(1 if fails else 0)
