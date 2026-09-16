"""fvml.py 자체 시험 — 심은 효과를 되찾는가, 없을 때 안 만들어내는가.

돌리는 법:  python3 tools/test_fvml.py
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvml


def make_design(n_block=5, n_linker=4, seed=0):
    """실제 설계와 같은 모양: 블록 5개 × 링커 4종. 길이는 블록마다 같은 4값."""
    rng = np.random.default_rng(seed)
    L = [15, 18, 21, 25]
    rows = [dict(블록=b, 항체=f"Ab{b}", 링커=f"L{i}", 길이=L[i])
            for b in range(1, n_block + 1) for i in range(n_linker)]
    F = pd.DataFrame(rows)
    # 앙상블 특징 — 길이와 거의 같은 것 하나, 길이와 무관한 것 하나
    F["열림분율"] = 0.05 + 0.01 * (F.길이 - 15) + rng.normal(0, 0.005, len(F))
    F["접촉"] = rng.normal(0, 1, len(F))
    return F


def _planted(D, z, beta, sigma, seed):
    rng = np.random.default_rng(seed)
    return D.center(beta * z + rng.normal(0, sigma, len(z)))


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


# ─────────────────────────────────────────────────────────────────────────────
def test_target_centering():
    F = make_design()
    F["HMW"] = np.r_[np.linspace(2, 8, 4), np.linspace(20, 26, 4),
                     np.linspace(1, 3, 4), np.linspace(10, 14, 4),
                     np.linspace(5, 9, 4)]
    y = fvml.make_target(F, "HMW")
    assert y.groupby(F.블록).mean().abs().max() < 1e-9, "블록 평균이 0 이 아니다"
    for b, g in F.groupby("블록"):
        assert g.HMW.idxmin() == y[g.index].idxmax(), "부호가 뒤집히지 않았다"
    print("  OK  make_target — 블록 중심화 + 부호 (HMW 는 낮을수록 좋다)")


def test_design_pairs():
    F = make_design()
    D = fvml.Design(F)
    assert len(D.pairs) == 5 * 6, f"쌍이 30개가 아니다: {len(D.pairs)}"
    # 쌍이 블록을 가로지르지 않는지
    for i, j in D.pairs:
        assert F.블록.iloc[i] == F.블록.iloc[j], "블록을 가로지르는 쌍이 있다"
    y = np.arange(len(F), dtype=float)
    assert fvml.concordance(y, y, D)[2] == 1.0
    assert fvml.concordance(-y, y, D)[2] == 0.0
    print(f"  OK  Design — 블록 안 쌍 {len(D.pairs)}개, 완전일치 1.0 / 완전불일치 0.0")


def test_concordance_ignores_nan_and_ties():
    F = make_design(n_block=2, n_linker=3)
    D = fvml.Design(F)
    y = np.arange(6, dtype=float)
    p = y.copy(); p[0] = np.nan                       # 한 점이 NaN
    a, n, c = fvml.concordance(p, y, D)
    assert n == 2 * 3 - 2, f"NaN 이 낀 쌍을 안 뺐다: {n}"     # 블록당 3쌍, NaN 이 2쌍 죽임
    p2 = np.zeros(6)                                  # 전부 동점
    assert fvml.concordance(p2, y, D)[1] == 0, "동점을 셌다"
    print("  OK  concordance — NaN 과 동점을 세지 않는다")


def test_lobo_no_leakage():
    """시험 블록이 학습에 안 들어갔는지 — 시험 블록만 통째로 평행이동해도
    그 블록 안의 예측 **순서**는 안 바뀌어야 한다."""
    F = make_design()
    D = fvml.Design(F)
    z = _z(F.열림분율)
    y = _planted(D, z, 1.0, 0.5, 7)
    p1, _ = fvml.lobo_predict(z[:, None], y, D)
    z2 = z.copy(); z2[D.blocks[0]] += 100.0
    p2, _ = fvml.lobo_predict(z2[:, None], y, D)
    b = D.blocks[0]
    o1 = np.argsort(p1[b]); o2 = np.argsort(p2[b])
    assert (o1 == o2).all(), f"시험 블록의 예측 순서가 바뀌었다 {o1} vs {o2}"
    print("  OK  lobo_predict — 시험 블록이 학습(표준화 포함)에 안 샌다")


def test_lobo_standardizes_inside_fold():
    """폴드 안 표준화인지 직접 확인: 학습 블록만 스케일을 바꾸면 예측이 그대로여야 한다
    (능선 벌점이 표준화된 좌표에 걸리므로, 폴드 안 표준화면 스케일 불변이다)."""
    F = make_design(); D = fvml.Design(F)
    z = _z(F.열림분율); y = _planted(D, z, 1.5, 0.4, 3)
    p1, _ = fvml.lobo_predict(z[:, None], y, D)
    p2, _ = fvml.lobo_predict((z * 1000.0)[:, None], y, D)
    assert np.allclose(p1[np.isfinite(p1)], p2[np.isfinite(p2)], atol=1e-6), \
        "특징 스케일을 바꿨더니 예측이 달라졌다 — 폴드 안 표준화가 아니다"
    print("  OK  lobo_predict — 특징 스케일에 불변 (= 폴드 안에서 표준화한다)")


def test_null_is_calibrated():
    """β=0 일 때 순열 p 가 균등이어야 한다 = 거짓양성률이 명목값이다."""
    F = make_design(); D = fvml.Design(F)
    X = _z(F.열림분율)[:, None]
    ps = []
    for s in range(200):
        rng = np.random.default_rng(5000 + s)
        y = D.center(rng.normal(0, 1, len(F)))
        ps.append(fvml.perm_test_signal(X, y, D, n_perm=300, seed=s)["순열p"])
    ps = np.array(ps)
    fpr = float((ps < 0.05).mean())
    print(f"  OK  perm_test_signal — β=0 거짓양성률 {fpr:.3f} (명목 0.05, "
          f"200시행 95% 구간 ≈ 0.02~0.08) · p 중앙값 {np.median(ps):.2f}")
    assert fpr < 0.12, f"거짓양성률이 너무 높다: {fpr}"
    assert 0.35 < np.median(ps) < 0.68, f"p 중앙값이 0.5 근처가 아니다: {np.median(ps)}"


def test_recovers_planted_signal():
    F = make_design(); D = fvml.Design(F)
    z = _z(F.열림분율)
    y = _planted(D, z, 3.0, 0.5, 42)
    r = fvml.perm_test_signal(z[:, None], y, D, n_perm=1000, seed=1)
    print(f"  OK  심은 신호 회수 — 일치도 {r['일치도']:.2f} "
          f"(귀무평균 {r['귀무평균']:.2f}) 순열p {r['순열p']:.4f} 이항p {r['이항p']:.4f}")
    assert r["순열p"] < 0.05, f"강한 신호를 못 찾았다: {r}"
    assert r["일치도"] > 0.75


def test_increment_null_when_feature_is_length_in_disguise():
    """앙상블 특징이 길이의 변장일 때 '증분'이 유의하면 안 된다."""
    F = make_design(); D = fvml.Design(F)
    L = _z(F.길이)
    dis = L * 2.0 + 1e-9                               # 길이와 r = 1.0
    y = _planted(D, L, 2.5, 0.5, 5)
    r = fvml.perm_test_increment(L[:, None], dis[:, None], y, D, n_perm=600, seed=2)
    print(f"  OK  증분 검정(변장) — 길이만 {r['길이만']:.2f} → "
          f"+앙상블 {r['길이_앙상블']:.2f} · 증분 {r['증분']:+.2f} · p {r['순열p']:.3f}")
    assert r["순열p"] > 0.05, f"길이의 변장이 유의하게 나왔다: {r}"


def test_increment_finds_real_extra_signal():
    """길이와 무관한 특징에 진짜 신호를 심으면 증분이 잡혀야 한다."""
    F = make_design(); D = fvml.Design(F)
    rng = np.random.default_rng(11)
    L, C = _z(F.길이), _z(F.접촉)
    y = D.center(1.0 * L + 3.0 * C + rng.normal(0, 0.4, len(F)))
    r = fvml.perm_test_increment(L[:, None], C[:, None], y, D, n_perm=1000, seed=3)
    print(f"  OK  증분 검정(진짜) — 길이만 {r['길이만']:.2f} → "
          f"+앙상블 {r['길이_앙상블']:.2f} · 증분 {r['증분']:+.2f} · p {r['순열p']:.4f}")
    assert r["순열p"] < 0.05, f"진짜 증분을 못 찾았다: {r}"


def test_increment_null_is_calibrated():
    """증분 검정도 β_추가=0 에서 거짓양성률이 명목이어야 한다."""
    F = make_design(); D = fvml.Design(F)
    L, C = _z(F.길이), _z(F.접촉)
    ps = []
    for s in range(120):
        rng = np.random.default_rng(9000 + s)
        y = D.center(1.5 * L + rng.normal(0, 1.0, len(F)))   # 길이에만 신호
        ps.append(fvml.perm_test_increment(L[:, None], C[:, None], y, D,
                                           n_perm=200, seed=s)["순열p"])
    fpr = float((np.array(ps) < 0.05).mean())
    print(f"  OK  증분 검정 영점 — 거짓양성률 {fpr:.3f} (명목 0.05)")
    assert fpr < 0.15, f"증분 검정의 거짓양성률이 너무 높다: {fpr}"


def test_partial_spearman_vs_v11_formula():
    from scipy import stats as st
    rng = np.random.default_rng(0)
    n = 20
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, .3, n)
    y = z + rng.normal(0, .3, n)                  # x-y 상관은 전부 z 를 통해서만
    rxy, rxz, ryz = (st.spearmanr(x, y)[0], st.spearmanr(x, z)[0], st.spearmanr(y, z)[0])
    v11 = rxy - rxz * ryz
    ok = fvml.partial_spearman(x, y, z)
    print(f"  OK  편상관 — v11식 {v11:+.3f} · 올바른 식 {ok:+.3f} "
          f"(|올바른| ≥ |v11| — 분모가 1 이하라 나누면 커진다)")
    assert abs(ok) >= abs(v11) - 1e-9, "올바른 편상관이 v11 식보다 작다 — 방향이 틀렸다"
    assert abs(ok) < 0.45, f"편상관이 0 근처가 아니다: {ok}"


def test_confound_ledger():
    F = make_design()
    T = fvml.confound_ledger(F, ["열림분율", "접촉"])
    print("  OK  교란 원장")
    print(T.to_string(index=False))
    assert T.set_index("특징").loc["열림분율", "판정"] == "길이의 변장"
    assert T.set_index("특징").loc["접촉", "판정"] == "독립적"


def test_power_curve_shape():
    F = make_design(); D = fvml.Design(F)
    t0 = time.time()
    P = fvml.power_curve(_z(F.접촉), D, betas=(0.0, 1.0, 2.0, 4.0),
                         n_sim=120, n_perm=200, seed=4)
    print(f"  OK  검정력 곡선 ({time.time()-t0:.0f}s)")
    print(P.to_string(index=False))
    mde = fvml.min_detectable_effect(P, 0.8)
    print(f"      검출률 80% 에 닿는 최소 효과크기 ≈ {mde:.2f} "
          f"(특징 1 SD 당 타깃 {mde:.1f} σ)")
    assert P.검출률.iloc[0] < 0.15, f"β=0 에서 검출률이 너무 높다: {P.검출률.iloc[0]}"
    assert P.검출률.iloc[-1] > 0.5, f"큰 효과도 못 잡는다: {P.검출률.iloc[-1]}"
    assert P.검출률.is_monotonic_increasing or P.검출률.iloc[-1] > P.검출률.iloc[0]


def test_bootstrap_se_and_attenuation():
    rng = np.random.default_rng(0)
    fr = pd.DataFrame(dict(항체="A", 링커=np.repeat(["a", "b"], 150),
                           OCD6=np.r_[rng.normal(3, 1, 150), rng.normal(5, 2, 150)]))
    T = fvml.bootstrap_feature_se(fr, ["항체", "링커"], "OCD6", np.median)
    print("  OK  부트스트랩 SE")
    print(T.to_string(index=False))
    exp = 1.253 * 1 / np.sqrt(150)                # 중앙값의 SE ≈ 1.253·σ/√n
    assert abs(T.SE.iloc[0] - exp) < 0.06, f"SE 가 이론값과 다르다 {T.SE.iloc[0]} vs {exp}"
    print("      " + fvml.attenuation_note(0.3, 1.0))
    print("      " + fvml.attenuation_note(0.9, 1.0))
    assert "★" in fvml.attenuation_note(0.9, 1.0), "큰 오차에 경고가 안 붙는다"


if __name__ == "__main__":
    print("fvml 자체 시험")
    print("=" * 74)
    fails = 0
    for name, fn in sorted((k, v) for k, v in list(globals().items())
                           if k.startswith("test_") and callable(v)):
        t0 = time.time()
        try:
            fn()
        except AssertionError as e:
            fails += 1; print(f"  ★ 실패 {name}: {e}")
        except Exception as e:
            fails += 1
            import traceback; traceback.print_exc()
            print(f"  ★ 오류 {name}: {type(e).__name__}: {e}")
    print("=" * 74)
    print("전부 통과" if not fails else f"{fails}건 실패")
    sys.exit(1 if fails else 0)
