"""fvfeat.py 자체 시험 — 가짜 PDB 프레임을 만들어 특징 추출이 맞는지 본다.

돌리는 법:  python3 tools/test_fvfeat.py
"""
import sys, os, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvfeat


# ─────────────────────────────────────────────────────────────────────────────
def write_fake_pdb(path, ca):
    """CA 만 든 최소 PDB. BioEmu 출력과 같은 열 위치를 쓴다."""
    with open(path, "w") as f:
        for i, p in enumerate(ca, 1):
            f.write(f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
                    f"{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}  1.00  0.00           C\n")
        f.write("END\n")


def make_ca(b1=120, nl=15, b2n=120, seed=0, linker_radius=3.0):
    """도메인 두 덩어리 + 그 사이 링커. 링커를 도메인에 가깝게/멀게 둘 수 있다."""
    rng = np.random.default_rng(seed)
    d1 = rng.normal(0, 8, (b1, 3)) + np.array([-18, 0, 0])
    d2 = rng.normal(0, 8, (b2n, 3)) + np.array([+18, 0, 0])
    t = np.linspace(0, 1, nl)[:, None]
    lk = (1 - t) * d1[-1] + t * d2[0]
    lk = lk + rng.normal(0, linker_radius, (nl, 3))
    return np.vstack([d1, lk, d2])


# ─────────────────────────────────────────────────────────────────────────────
def test_ca_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        ca = make_ca()
        p = os.path.join(td, "x.pdb")
        write_fake_pdb(p, ca)
        back = fvfeat.ca_xyz(p)
        assert back.shape == ca.shape, f"모양이 다르다 {back.shape} vs {ca.shape}"
        assert np.abs(back - ca).max() < 1e-3, "좌표가 왕복에서 깨졌다"
    print("  OK  ca_xyz — PDB 왕복")


def test_frame_geometry_units():
    ca = make_ca(b1=120, nl=15, b2n=120, seed=1)
    g = fvfeat.frame_geometry(ca, 120, 135)
    assert set(g) >= {"링커접촉_잔기당", "링커밀착율", "링커Rg_잔기당", "링커신장도"}
    assert 0 <= g["링커밀착율"] <= 1, g["링커밀착율"]
    assert g["링커접촉_잔기당"] >= 0
    print("  OK  frame_geometry — 키와 범위")
    print(f"      접촉/잔기 {g['링커접촉_잔기당']:.2f} · 밀착율 {g['링커밀착율']:.2f} "
          f"· 신장도 {g['링커신장도']:.2f}")


def test_contact_responds_to_linker_position():
    """링커가 도메인에서 멀어지면 접촉이 줄어야 한다 — 특징이 진짜로 무언가를 잰다.

    ★ 처음 쓴 시험은 틀렸었다: 링커를 도메인 1 **위로** 접어 놓고 접촉이 늘기를
      기대했는데, 접촉은 두 도메인 **모두**와의 쌍을 세므로 한쪽에 달라붙으면
      다른 쪽과의 접촉을 잃어 총합이 오히려 준다. 그건 이 지표의 정의가 그런 것이고,
      계면 홈에 걸쳐 있는 링커가 최대가 된다는 뜻이다 (= Okazaki 의 Glue 자리).
      그래서 시험을 '계면에 걸침 vs 완전히 떨어짐' 으로 고쳤다.
    """
    ca = make_ca(seed=2, linker_radius=0.5)             # 계면을 가로지르는 링커
    inner = fvfeat.frame_geometry(ca, 120, 135)
    far_ca = ca.copy()
    far_ca[120:135] = ca[120:135] + np.array([0.0, 200.0, 0.0])   # 링커만 멀리
    outer = fvfeat.frame_geometry(far_ca, 120, 135)
    assert inner["링커접촉_잔기당"] > outer["링커접촉_잔기당"], \
        f"{inner['링커접촉_잔기당']} vs {outer['링커접촉_잔기당']}"
    assert outer["링커접촉_잔기당"] == 0.0 and outer["링커밀착율"] == 0.0
    print(f"  OK  접촉 감도 — 계면 가로지름 {inner['링커접촉_잔기당']:.1f} → "
          f"멀리 떨어짐 {outer['링커접촉_잔기당']:.1f} (잔기당 쌍 수)")

    # 그리고 한쪽 도메인에만 달라붙으면 계면에 걸친 것보다 **적다** — 정의 확인
    one = ca.copy()
    one[120:135] = ca[:15] + np.array([0.0, 6.0, 0.0])
    g1 = fvfeat.frame_geometry(one, 120, 135)
    print(f"      한쪽 도메인에만 붙음 {g1['링커접촉_잔기당']:.1f} "
          f"— 계면에 걸친 것보다 적다. 이 지표의 최대는 **계면 홈**이다.")
    assert g1["링커접촉_잔기당"] < inner["링커접촉_잔기당"]


def test_extended_linker_score():
    """완전히 편 링커의 신장도가 1 에 가까워야 한다 (CA-CA 3.8 Å 기준)."""
    nl = 15
    lk = np.column_stack([np.arange(nl) * 3.8, np.zeros(nl), np.zeros(nl)])
    ca = np.vstack([np.zeros((10, 3)), lk, np.zeros((10, 3)) + 100])
    g = fvfeat.frame_geometry(ca, 10, 25)
    assert abs(g["링커신장도"] - 1.0) < 1e-6, g["링커신장도"]
    print(f"  OK  링커신장도 — 완전신장 = {g['링커신장도']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def make_frames(seed=0, n=150):
    """frames.csv 와 같은 모양의 가짜 표. 링커마다 분포를 다르게 심는다."""
    rng = np.random.default_rng(seed)
    rows = []
    # (링커, Δdc 평균, Δdc 퍼짐) — 세 번째 링커만 자주 떨어지게 만든다
    spec = [("L15", 0.0, 0.3), ("L18", 0.2, 0.4), ("L21", 0.1, 2.5), ("L25", 0.3, 0.5)]
    for lk, mu, sd in spec:
        for i in range(n):
            d = {f"Δ{k}": rng.normal(0, fvfeat.AB_SD[k]) for k in fvfeat.AB6}
            d["Δdc"] = rng.normal(mu, sd)
            d["OCD6"] = sum(abs(d[f"Δ{k}"]) / fvfeat.AB_SD[k] for k in fvfeat.AB6)
            rows.append(dict(블록=1, 항체="Ab1", 배향="HL", 링커=lk,
                             길이=int(lk[1:]), 생성기="BioEmu",
                             태그=f"{lk}__BioEmu_s{i}", **d))
    return pd.DataFrame(rows)


def test_build_features_shape():
    fr = make_frames()
    F = fvfeat.build_features(fr, None)
    assert len(F) == 4, f"구성체 4개가 안 나왔다: {len(F)}"
    assert fvfeat.PRIMARY in F.columns
    for c in fvfeat.EXPLORATORY:
        assert c in F.columns, f"탐색 특징 {c} 가 없다"
    print("  OK  build_features — 구성체당 한 줄")
    print(F[["링커", "길이", "n프레임", "n붙음", fvfeat.PRIMARY,
             "OCD6중앙", "OCD6_MAD", "자연2σ밖"]].round(3).to_string(index=False))


def test_open_fraction_tracks_planted_spread():
    """Δdc 퍼짐을 크게 심은 링커의 열림분율이 제일 커야 한다."""
    fr = make_frames()
    F = fvfeat.build_features(fr, None).set_index("링커")
    assert F.loc["L21", fvfeat.PRIMARY] > F[fvfeat.PRIMARY].drop("L21").max(), \
        F[fvfeat.PRIMARY].to_dict()
    print(f"  OK  열림분율 — 심은 링커 L21 {F.loc['L21', fvfeat.PRIMARY]:.2f} > "
          f"나머지 최대 {F[fvfeat.PRIMARY].drop('L21').max():.2f}")


def test_paired_filter_actually_filters():
    """붙은 프레임만 쓰는지 — 안 붙은 프레임에 극단 OCD6 를 심어 확인한다."""
    fr = make_frames()
    bad = fr.Δdc.abs() >= 2.0
    assert bad.any(), "시험 데이터에 안 붙은 프레임이 없다"
    fr2 = fr.copy()
    fr2.loc[bad, "OCD6"] = 999.0                       # 떨어진 프레임만 오염시킨다
    a = fvfeat.build_features(fr, None).set_index("링커").OCD6중앙
    b = fvfeat.build_features(fr2, None).set_index("링커").OCD6중앙
    assert np.allclose(a.values, b.values), \
        f"떨어진 프레임이 각도 중앙값에 샜다\n{a}\n{b}"
    print("  OK  짝지음 거르개 — 떨어진 프레임이 각도 요약에 안 샌다")


def test_mad_matches_sigma():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 2.0, 200000)
    assert abs(fvfeat.mad(v) - 2.0) < 0.02, fvfeat.mad(v)
    print(f"  OK  mad — 정규 σ=2 에서 {fvfeat.mad(v):.3f}")


def test_w1():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)
    d = fvfeat.w1(a, b)
    assert abs(d - 3.0) < 0.1, d
    print(f"  OK  w1 — 평균 3 떨어진 두 정규분포에서 {d:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def test_linker_descriptors():
    d = fvfeat.linker_descriptors("GGGGSGGGGSGGGGS")
    assert d["길이"] == 15
    assert abs(d["G분율"] - 12 / 15) < 1e-9, d["G분율"]
    assert abs(d["S분율"] - 3 / 15) < 1e-9
    assert abs(d["순전하_잔기당"]) < 1e-9, "G4S 에 전하가 있으면 안 된다"
    e = fvfeat.linker_descriptors("EAAAKEAAAKEAAAK")
    assert e["나선성향"] > d["나선성향"], "EAAAK 가 G4S 보다 나선성향이 높아야 한다"
    assert e["음전하분율"] > 0
    print(f"  OK  linker_descriptors — G4S 나선성향 {d['나선성향']:.2f} < "
          f"EAAAK {e['나선성향']:.2f}")


def test_panel_is_orthogonal():
    """합성 패널의 핵심 주장: 길이와 조성이 **직교**한다."""
    P = fvfeat.panel_linkers()
    assert len(P) == 24, len(P)
    # 각 조성이 4길이 전부에, 각 길이가 6조성 전부에 나온다
    assert (P.groupby("조성").설계길이.nunique() == 4).all()
    assert (P.groupby("설계길이").조성.nunique() == 6).all()
    # 조성 기술자가 길이와 상관이 (거의) 0 이어야 한다
    worst = 0.0
    for c in ("G분율", "순전하_잔기당", "나선성향", "P분율"):
        r = abs(np.corrcoef(P[c], P.설계길이)[0, 1])
        worst = max(worst, 0.0 if np.isnan(r) else r)
    print(f"  OK  합성 패널 — 6조성 × 4길이 = 24, "
          f"조성 기술자 vs 길이 최대 |r| = {worst:.3f}")
    assert worst < 0.25, f"패널이 직교하지 않는다: |r|={worst}"
    print(P.pivot_table(index="조성", columns="설계길이", values="길이",
                        aggfunc="first").to_string())


def test_real_panel_is_confounded():
    """대조: 실제 패널(링커마다 길이 하나)에서는 |r| = 1 이 된다."""
    R = pd.DataFrame([fvfeat.linker_descriptors(s) for s in
                      ["GGGGSGGGGSGGGGS",                   # 15, G4S
                       "GGGGSGGGGSGGGGSGGG",                # 18
                       "GGGGSGGGGSGGGGSGGGGSG",             # 21
                       "GGGGSGGGGSGGGGSGGGGSGGGGS"]])       # 25
    r = abs(np.corrcoef(R.G분율, R.길이)[0, 1])
    print(f"  OK  실제 패널 대조 — G분율 vs 길이 |r| = {r:.3f} "
          f"(같은 모티프를 늘리면 조성은 거의 안 변한다 = 조성축이 없다)")
    assert R.G분율.std() < 0.05, "실제 패널에서 조성이 생각보다 변한다"


def test_two_way_anova_decomposes():
    """길이에만 신호를 심으면 η²(길이) 가 크고 η²(조성) 은 작아야 한다."""
    rng = np.random.default_rng(0)
    P = fvfeat.panel_linkers()
    P["관측"] = 0.1 * P.설계길이 + rng.normal(0, 0.2, len(P))
    A = fvfeat.two_way_anova(P, "관측").set_index("요인")
    print("  OK  two_way_anova — 길이에만 심었을 때")
    print(A.to_string())
    assert A.loc["설계길이", "eta2"] > 0.7, A.to_dict()
    assert A.loc["조성", "eta2"] < 0.15, A.to_dict()

    P["관측2"] = 0.8 * (P.조성 == "EAAAK") + rng.normal(0, 0.1, len(P))
    B = fvfeat.two_way_anova(P, "관측2").set_index("요인")
    print("       조성에만 심었을 때")
    print(B.to_string())
    assert B.loc["조성", "eta2"] > 0.7, B.to_dict()
    assert B.loc["설계길이", "eta2"] < 0.15, B.to_dict()


def test_anova_refuses_F_without_replication():
    """칸당 1개면 F·p 를 내면 안 된다 — 잔차 자유도가 0 이기 때문이다."""
    rng = np.random.default_rng(0)
    P = fvfeat.panel_linkers()
    P["관측"] = 0.1 * P.설계길이 + rng.normal(0, 0.2, len(P))
    A = fvfeat.two_way_anova(P, "관측")
    assert A.F.isna().all(), "반복이 없는데 F 를 냈다"
    assert "상호작용+오차" in set(A.요인), A.요인.tolist()
    print(f"  OK  반복 없음 방어 — F 를 내지 않는다. 주석: {A.attrs['note'][:60]}…")


def test_split_replicates_gives_real_error_term():
    """프레임을 쪼개 칸 안 반복을 만들면 상호작용과 오차가 분리된다."""
    rng = np.random.default_rng(3)
    P = fvfeat.panel_linkers()
    rows = []
    for _, r in P.iterrows():
        mu = 0.05 * r.설계길이 + 0.6 * (r.조성 == "EAAAK")
        for i in range(120):
            rows.append(dict(항체="Ab1", 링커=r.패널링커, 조성=r.조성,
                             설계길이=r.설계길이, OCD6=rng.normal(mu, 1.0)))
    fr = pd.DataFrame(rows)
    R = fvfeat.split_replicates(fr, np.median, "OCD6", key_cols=("항체", "링커"), k=3)
    R = R.merge(P[["패널링커", "조성", "설계길이"]],
                left_on="링커", right_on="패널링커")
    assert len(R) == 24 * 3, len(R)
    A = fvfeat.two_way_anova(R, "OCD6").set_index("요인")
    print("  OK  split_replicates — 칸당 3반복 → F 검정이 산다")
    print(A.to_string())
    print(f"      {A.attrs if hasattr(A,'attrs') else ''}")
    assert A.F.notna().sum() >= 3, "F 가 안 나왔다"
    assert A.loc["조성", "p"] < 0.05, "심은 조성 효과를 못 잡았다"
    assert A.loc["설계길이", "p"] < 0.05, "심은 길이 효과를 못 잡았다"
    assert A.loc["상호작용", "p"] > 0.01, "안 심은 상호작용이 유의하게 나왔다"


def test_residualize():
    P = fvfeat.panel_linkers()
    R = fvfeat.residualize_on_length(P, ["G분율"], length_col="설계길이")
    r = abs(np.corrcoef(R.G분율_길이제거, R.설계길이)[0, 1])
    assert r < 1e-8, f"잔차가 길이와 아직 상관이 있다: {r}"
    print(f"  OK  residualize_on_length — 잔차 vs 길이 |r| = {r:.2e}")


def test_build_geom_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "flow")
        os.makedirs(d)
        for i in range(12):
            write_fake_pdb(os.path.join(d, f"L15__BioEmu_s{i}.pdb"),
                           make_ca(seed=i))
        CONS = pd.DataFrame([dict(항체="Ab1", 링커="L15", b1=120, b2=135)])
        G = fvfeat.build_geom(CONS, lambda r: d, verbose=False)
        assert len(G) == 12, len(G)
        assert G.태그.nunique() == 12
        assert G.링커접촉_잔기당.notna().all()
    print("  OK  build_geom — 12프레임 통과, 태그 유일")


if __name__ == "__main__":
    print("fvfeat 자체 시험")
    print("=" * 70)
    fails = 0
    for name, fn in sorted((k, v) for k, v in list(globals().items())
                           if k.startswith("test_") and callable(v)):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  ★ 실패 {name}: {e}")
        except Exception as e:
            fails += 1
            import traceback; traceback.print_exc()
            print(f"  ★ 오류 {name}: {type(e).__name__}: {e}")
    print("=" * 70)
    print("전부 통과" if not fails else f"{fails}건 실패")
    sys.exit(1 if fails else 0)
