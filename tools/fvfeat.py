"""FvFlow · 앙상블 → 구성체당 한 줄 특징표 (v12 6절이 그대로 쓰는 코드)

프레임 단위 표 두 개를 구성체 단위 한 줄로 접는다.

    frames.csv  구조 하나당 한 줄 · ΔABangle 6축 · OCD6 · 짝지음      (v09 의 flow.csv)
    geom.csv    구조 하나당 한 줄 · 링커 접촉 · Rg · 말단간            (v12 가 새로 만든다)
        ↓
    features.csv  **구성체당 한 줄** — 7절 ML 이 먹는 유일한 입력

접는 방식이 이 연구의 논지다. 가설은 "링커가 평균을 옮긴다"가 아니라
**"링커가 분포를 바꾼다"** 이므로, 중앙값뿐 아니라 퍼짐·꼬리질량·인구분율을 같이 낸다.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats as st

# ABangle 저장소 All_Angles.dat (SAbDab 1296 구조) 에서 잰 자연 변이 폭.
# Δ 를 이걸로 나누면 "자연 변이의 몇 배인가" 가 된다.
AB_SD = dict(HL=3.92, HC1=2.18, HC2=3.07, LC1=2.49, LC2=2.25, dc=0.268)
AB6 = ["HL", "HC1", "HC2", "LC1", "LC2", "dc"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 프레임 좌표에서 나오는 기하 — 링커 자체를 본다
# ─────────────────────────────────────────────────────────────────────────────
def ca_xyz(path: str) -> np.ndarray:
    """PDB → CA 좌표 (n, 3), Å. BioEmu 출력은 백본만이라 파싱이 단순하다."""
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path)
                     if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def frame_geometry(ca: np.ndarray, b1: int, b2: int, cut: float = 10.0) -> dict:
    """프레임 하나의 기하 관측량.

    ★ 여기 있는 것들이 **조성 채널을 나를 수 있는 유일한 후보**다.
      ABangle 6축은 도메인 배향만 보므로 링커가 어디에 붙어 있는지 모른다.
      Okazaki 의 Glue-linker 기전(링커가 VH-VL 홈에 달라붙어 closed 를 안정화)은
      배향이 아니라 **접촉**으로 나타난다.

    잔기당으로 내는 이유: 길이로 나누지 않으면 전부 길이의 다른 이름이 된다.
    """
    n = len(ca)
    if n < b2 + 5 or b2 - b1 < 3:
        return {}
    L = ca[b1:b2]                                   # 링커
    D = np.vstack([ca[:b1], ca[b2:]])               # 두 도메인
    nl = b2 - b1
    Dm = np.linalg.norm(L[:, None, :] - D[None, :, :], axis=-1)
    near = Dm.min(1)                                # 링커 잔기별 가장 가까운 도메인 거리
    Lc = L - L.mean(0)
    Dc = D - D.mean(0)
    return dict(
        # ── 링커-도메인 접촉 (조성 채널 후보) ─────────────────────────────
        링커접촉_잔기당=float((Dm < cut).sum() / nl),
        링커밀착율=float((near < cut).mean()),       # 도메인에 닿아 있는 링커 잔기 비율
        링커최근접=float(np.median(near)),
        # ── 링커 자체의 모양 ────────────────────────────────────────────
        링커Rg_잔기당=float(np.sqrt((Lc ** 2).sum(1).mean()) / nl),
        링커신장도=float(np.linalg.norm(L[-1] - L[0]) / (3.8 * (nl - 1))),  # 1.0=완전신장
        # ── 전체 ────────────────────────────────────────────────────────
        도메인Rg=float(np.sqrt((Dc ** 2).sum(1).mean())),
        말단간=float(np.linalg.norm(ca[-1] - ca[0])),
    )


def build_geom(CONS: pd.DataFrame, flow_dir, gen: str = "BioEmu",
               max_n: int | None = None, cut: float = 10.0,
               verbose: bool = True) -> pd.DataFrame:
    """모든 구성체의 모든 프레임에 frame_geometry 를 건다 → geom.csv 의 내용.

    flow_dir(r) 는 구성체 r 의 프레임이 든 디렉터리를 주는 함수다.
    태그(파일명 밑동)를 키로 두어 frames.csv 와 나중에 붙일 수 있게 한다.
    """
    rows = []
    for _, r in CONS.iterrows():
        d = flow_dir(r)
        fr = sorted(glob.glob(f"{d}/{r.링커}__{gen}_s*.pdb"))
        if max_n:
            fr = fr[:max_n]
        if not fr:
            continue
        ok = 0
        for p in fr:
            try:
                g = frame_geometry(ca_xyz(p), int(r.b1), int(r.b2), cut=cut)
            except Exception:
                continue
            if not g:
                continue
            rows.append(dict(항체=r.항체, 링커=r.링커, 생성기=gen,
                             태그=os.path.basename(p).rsplit(".", 1)[0], **g))
            ok += 1
        if verbose:
            print(f"  {r.항체:<20} {r.링커:<10} 기하 {ok}/{len(fr)}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 분포 요약자 — "분포가 바뀐다"는 주장을 숫자로 만든다
# ─────────────────────────────────────────────────────────────────────────────
def mad(v) -> float:
    """중앙값 절대편차 × 1.4826 (정규분포에서 σ 와 같아지게 맞춘 상수).
    표준편차보다 꼬리에 강해서 프레임 몇 개가 튀어도 안 흔들린다."""
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def outside_2sigma(g: pd.DataFrame, ax=("ΔHL", "Δdc")) -> float:
    """SAbDab 자연 2σ 타원 **밖**의 프레임 비율.

    "많이 움직였다"를 절대각이 아니라 **실제 항체들이 차지하는 폭** 기준으로 잰다.
    타원식: (x/2σx)² + (y/2σy)² > 1
    """
    sx = AB_SD[ax[0][1:]] * 2
    sy = AB_SD[ax[1][1:]] * 2
    q = (g[ax[0]] / sx) ** 2 + (g[ax[1]] / sy) ** 2
    return float((q > 1).mean())


def bimodality(v) -> float:
    """Sarle 의 bimodality coefficient = (skew² + 1) / kurtosis.

    0.555(균등분포) 를 넘으면 봉우리가 둘일 가능성이 있다는 널리 쓰이는 눈금이다.
    링커가 closed ⇌ open 평형을 옮긴다면 **단봉이 쌍봉이 되는 것**이 그 신호다.
    ※ n=150 에서 4차 적률은 시끄럽다. 참고치로만 쓰고 단독 판정에 쓰지 마라.
    """
    v = np.asarray(v, float)
    if len(v) < 20:
        return np.nan
    g1 = float(st.skew(v))
    g2 = float(st.kurtosis(v, fisher=False))
    return float((g1 ** 2 + 1) / g2) if g2 > 1e-9 else np.nan


def w1(a, b) -> float:
    """1차 Wasserstein 거리 — 두 분포가 얼마나 다른가를 **원래 단위로** 준다.

    KL 이나 2D 엔트로피는 n=150 에서 편향이 크다 (2D 22×22 격자면 칸당 0.3개다).
    1D Wasserstein 은 경험분포만으로 일치추정량이라 이 표본 크기에서 믿을 수 있다.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return float(st.wasserstein_distance(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 프레임 표 → 구성체 표
# ─────────────────────────────────────────────────────────────────────────────
#: 사전 등록 특징. **결과를 보기 전에 박는다.**
#:   확증(confirmatory) 1개  ·  탐색(exploratory) 3개
#: n≈20, 독립 블록 5개에서 4개가 한계다. 더 넣으면 무엇이 나와도 우연이다.
PRIMARY = "열림분율"
EXPLORATORY = ["OCD6중앙", "OCD6_MAD", "링커접촉_잔기당"]
BASELINE = ["길이"]

#: frame_geometry 가 내는 열. build_features 는 geom 이 없어도 이 열들을 NaN 으로 만든다.
GEOM_FEATS = ["링커접촉_잔기당", "링커밀착율", "링커최근접",
              "링커Rg_잔기당", "링커신장도", "도메인Rg", "말단간"]


def build_features(frames: pd.DataFrame, geom: pd.DataFrame | None,
                   gen: str = "BioEmu", pair_dc: float = 2.0,
                   ref_gen: str = "BBFlow-ref",
                   meta_cols=("블록", "항체", "링커", "길이", "배향")) -> pd.DataFrame:
    """프레임 표 → **구성체당 한 줄**.

    짝지음 판정(|Δdc| < pair_dc)은 여기서 두 가지로 쓰인다.
      · 인구분율로:  열림분율 = 안 붙은 프레임의 비율          ← 이게 확증 특징이다
      · 거르개로:    각도 요약(OCD6 등)은 **붙은 프레임만**으로 낸다

    ★ 두 번째가 중요하다. 떨어진 Fv 에서는 ABangle 의 각도 4개가 정의를 잃는다
      (붙어 있는 계면을 전제로 정의된 양이다). 섞어서 중앙값을 내면 그 중앙값은
      각도가 아니라 '떨어진 프레임이 몇 개냐'를 재게 된다 — 즉 열림분율의
      잡음 섞인 복제본이 되고, 두 특징이 같은 것이 되어 버린다.
    """
    G = frames[frames.생성기 == gen].copy()
    if not len(G):
        raise ValueError(f"생성기 '{gen}' 프레임이 없다. frames.csv 의 생성기 열: "
                         f"{sorted(frames.생성기.unique())}")
    G["짝지음"] = G["Δdc"].abs() < pair_dc

    # 기준점 앙상블이 있으면 분포 거리를 잰다. 없으면(BioEmu 는 못 만든다) 건너뛴다.
    REF = frames[(frames.생성기 == ref_gen)]
    ref_ocd = {ab: g.OCD6.values for ab, g in REF.groupby("항체")} if len(REF) else {}

    if geom is not None and len(geom):
        G = G.merge(geom.drop(columns=["항체", "링커", "생성기"], errors="ignore"),
                    on="태그", how="left")

    rows = []
    for (ab, lk), g in G.groupby(["항체", "링커"], sort=False):
        c = g[g.짝지음]                                  # 붙은 프레임만
        d = {k: g[k].iloc[0] for k in meta_cols if k in g.columns}
        d.update(항체=ab, 링커=lk, n프레임=len(g), n붙음=len(c))

        # ── 확증 특징: 인구 분율 ────────────────────────────────────────
        # HMW:Monomer 가 인구 비율이므로 계산도 인구 비율이어야 한다.
        d[PRIMARY] = float(1 - g.짝지음.mean())

        # ── 각도 요약은 붙은 프레임에서만 ───────────────────────────────
        src = c if len(c) >= 20 else g                  # 붙은 게 너무 적으면 전체로
        d["붙음부족"] = len(c) < 20
        d["OCD6중앙"] = float(np.median(src.OCD6))
        d["OCD6_MAD"] = mad(src.OCD6)
        d["OCD6_IQR"] = float(np.subtract(*np.percentile(src.OCD6, [75, 25])))
        d["자연2σ밖"] = outside_2sigma(src)
        d["OCD6_쌍봉성"] = bimodality(src.OCD6)
        for k in AB6:
            d[f"Δ{k}중앙"] = float(np.median(src[f"Δ{k}"]))
            d[f"Δ{k}_MAD"] = mad(src[f"Δ{k}"])

        # ── 기준점 앙상블과의 분포 거리 ─────────────────────────────────
        d["W1_기준점"] = w1(src.OCD6, ref_ocd[ab]) if ab in ref_ocd else np.nan

        # ── 링커 기하 (조성 채널 후보) ──────────────────────────────────
        # ★ geom 이 없어도 **열은 반드시 만든다** (NaN 으로). 7절이 열 이름으로
        #   특징을 고르므로, 열이 통째로 없으면 KeyError 로 죽는 대신 조용히
        #   다른 특징 집합이 쓰이는 사고가 난다.
        for k in GEOM_FEATS:
            d[k] = (float(np.median(g[k].dropna()))
                    if k in g.columns and g[k].notna().any() else np.nan)

        rows.append(d)
    F = pd.DataFrame(rows)
    return F.sort_values([c for c in ("블록", "항체", "길이") if c in F.columns]) \
            .reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 링커 서열 기술자 — 8절 대리모형(Stage A)의 입력
# ─────────────────────────────────────────────────────────────────────────────
#: Kyte-Doolittle 소수성
_KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
           I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
           Y=-1.3, V=4.2)
#: Chou-Fasman 나선 성향
_HEL = dict(A=1.42, R=0.98, N=0.67, D=1.01, C=0.70, Q=1.11, E=1.51, G=0.57, H=1.00,
            I=1.08, L=1.21, K=1.16, M=1.45, F=1.13, P=0.57, S=0.77, T=0.83, W=1.08,
            Y=0.69, V=1.06)


def linker_descriptors(seq: str) -> dict:
    """링커 서열만으로 나오는 기술자. 구조가 필요 없다.

    ★ 길이와 조성을 반드시 **분리**해서 낸다. 전부 '분율'로 내면 길이에 대해
      불변이고, 길이는 별도 열 하나로만 들어간다. 실제 패널에서는 링커마다
      길이가 하나씩이라 둘이 완전히 얽혀 있는데, 4-B절의 합성 요인설계가
      그 얽힘을 푸는 유일한 장치다.
    """
    s = str(seq).upper()
    n = max(len(s), 1)
    f = lambda aa: sum(s.count(a) for a in aa) / n
    return dict(
        길이=len(s),
        G분율=f("G"), S분율=f("S"), P분율=f("P"), A분율=f("A"),
        양전하분율=f("KR"), 음전하분율=f("DE"),
        순전하_잔기당=(f("KR") - f("DE")),
        전하밀도=f("KRDE"),
        극성분율=f("STNQY"), 소수성분율=f("AVLIMFW"),
        소수성=float(np.mean([_KD.get(a, 0.0) for a in s])) if s else 0.0,
        나선성향=float(np.mean([_HEL.get(a, 1.0) for a in s])) if s else 1.0,
    )


SEQ_FEATS = ["G분율", "S분율", "P분율", "A분율", "양전하분율", "음전하분율",
             "순전하_잔기당", "전하밀도", "극성분율", "소수성분율",
             "소수성", "나선성향"]


def residualize_on_length(F: pd.DataFrame, cols, length_col="길이") -> pd.DataFrame:
    """각 조성 기술자에서 **길이로 설명되는 몫을 뺀다.**

    실제 패널은 길이와 조성이 얽혀 있어서, 뺀 뒤 남는 것이 '길이로는 설명 안 되는
    조성 성분'이다. 뺀 값의 분산이 거의 0 이면 그 기술자는 이 패널에서
    길이와 구분 불가능하다 — 그걸 표로 보여주는 것이 목적이다.
    """
    out = F.copy()
    x = F[length_col].values.astype(float)
    X = np.column_stack([np.ones_like(x), x])
    for c in cols:
        y = F[c].values.astype(float)
        m = np.isfinite(y)
        if m.sum() < 3 or np.std(x[m]) < 1e-12:
            out[f"{c}_길이제거"] = np.nan
            continue
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        out[f"{c}_길이제거"] = y - X @ beta
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. 합성 링커 요인설계 — "가상 데이터셋"의 정직한 형태
# ─────────────────────────────────────────────────────────────────────────────
#: 조성 × 길이 완전요인. 전부 5-mer 라 10/15/20/25 로 깨끗이 타일링된다.
#: ★ 이 패널의 값어치는 **표본 수가 아니라 직교성**이다. 실제 26건은 링커마다
#:   길이가 하나씩이라 길이와 조성을 영원히 못 가른다. 이 격자는 같은 길이에서
#:   조성만 바꾼 칸을 주므로, "BioEmu 가 조성을 보는가"에 직접 답한다.
MOTIF = {
    "G4S":  "GGGGS",   # 표준 유연 링커. 이 패널의 기준
    "GS":   "GGSGG",   # 유연하되 세린이 덜함 — G/S 비만 다르다
    "EAAAK": "EAAAK",  # 강직 α-나선 (Arai 2001). 유연↔강직 축의 반대편
    "PAS":  "APAPA",   # 프롤린-알라닌. 신장형, 나선 아님
    "ED":   "GEGGS",   # 음전하. G4S 에서 G 하나를 E 로
    "KR":   "GKGGS",   # 양전하. ED 의 부호 대칭 — 전하 **부호**를 가른다
}
LENGTHS = (10, 15, 20, 25)


def panel_linkers(motifs=None, lengths=LENGTHS) -> pd.DataFrame:
    """완전요인 링커 패널 (조성 6 × 길이 4 = 24). 각 칸이 무엇의 대조인지 붙여서."""
    motifs = motifs or MOTIF
    rows = []
    for name, m in motifs.items():
        for L in lengths:
            s = (m * (L // len(m) + 1))[:L]
            rows.append(dict(패널링커=f"{name}{L}", 조성=name, 설계길이=L,
                             링커서열=s, **linker_descriptors(s)))
    return pd.DataFrame(rows)


def split_replicates(frames: pd.DataFrame, statfn, value_col: str,
                     key_cols=("항체", "링커"), k: int = 3,
                     seed: int = 0) -> pd.DataFrame:
    """구성체마다 프레임을 k 덩이로 쪼개 **칸 안의 반복**을 만든다.

    ★ 왜 필요한가. 합성 패널은 칸(조성×길이)당 구성체가 하나뿐이라, 2원 분산분석의
      잔차 자유도가 0 이 된다 — 상호작용과 오차를 분리할 수 없고 잔차 SS 가
      기계적으로 0 이 나온다. 그 표에서 읽은 η² 는 해석이 불가능하다.

    프레임을 무작위로 k 등분해 각 덩이에서 기술자를 따로 내면, 칸 안의 흩어짐이
    **BioEmu 표집 잡음** 그 자체가 된다. 그리고 그것이 "이 칸과 저 칸의 차이가
    표집 잡음보다 큰가"를 물을 때 쓸 올바른 오차항이다.

    ※ 생물학적 반복이 아니다. 이 오차항으로 말할 수 있는 것은 "BioEmu 의 앙상블이
      조성에 따라 다르다" 까지이고, "실제 단백질이 다르다" 는 아니다.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in frames.groupby(list(key_cols), sort=False):
        v = g[value_col].dropna().values.astype(float)
        if len(v) < 3 * k:
            continue
        idx = rng.permutation(len(v))
        for j, part in enumerate(np.array_split(idx, k)):
            d = dict(zip(key_cols, key if isinstance(key, tuple) else (key,)))
            d.update({value_col: float(statfn(v[part])), "반복": j,
                      "n프레임": len(part)})
            rows.append(d)
    return pd.DataFrame(rows)


def two_way_anova(df: pd.DataFrame, value: str, a: str = "조성", b: str = "설계길이"):
    """조성 / 길이 / 상호작용이 각각 분산의 몇 %를 설명하는가 (η²) + F 검정.

    이 표 한 장이 4-B절의 판정이다:
      η²(길이) 만 크다        → BioEmu 는 사실상 |i−j| 만 본다. 조성은 못 본다.
      η²(조성) 도 크다        → 조성 채널이 있다. 그 채널을 나르는 특징을 본다.
      η²(상호작용) 이 크다    → 조성 효과가 길이마다 다르다 — 해석이 복잡해진다.

    ★ 두 가지를 반드시 지켜라.
      1) **구성체 단위**(또는 split_replicates 의 덩이 단위)로 돌려라. 프레임 150개를
         독립 표본으로 세면 자유도가 150배로 부풀어 무엇이든 p<0.001 이 된다.
      2) 칸당 관측이 **하나뿐이면 잔차 자유도가 0** 이다. 그때는 상호작용과 오차를
         가를 수 없으므로 F·p 를 내지 않고 η² 만, 그것도 "상호작용+오차" 로 묶어 낸다.
         split_replicates 로 칸 안 반복을 만들면 제대로 분리된다.
    """
    d = df[[value, a, b]].dropna()
    y = d[value].values.astype(float)
    gm = y.mean()
    sst = float(((y - gm) ** 2).sum())
    na, nb, N = d[a].nunique(), d[b].nunique(), len(d)
    cell = d.groupby([a, b])[value].agg(["mean", "size"])
    rep = int(cell["size"].min())

    ss = {}
    for k in (a, b):
        ss[k] = float(sum(len(g) * (g[value].mean() - gm) ** 2 for _, g in d.groupby(k)))
    ss_cells = float((cell["size"] * (cell["mean"] - gm) ** 2).sum())
    inter = max(ss_cells - ss[a] - ss[b], 0.0)
    resid = max(sst - ss_cells, 0.0)

    df_ = {a: na - 1, b: nb - 1, "상호작용": (na - 1) * (nb - 1)}
    df_resid = N - na * nb
    if rep < 2 or df_resid <= 0:                        # 반복이 없다 — F 를 못 낸다
        ss["상호작용+오차"] = inter + resid
        out = [dict(요인=k, SS=round(v, 4),
                    eta2=round(v / sst, 3) if sst > 1e-12 else np.nan,
                    자유도=df_.get(k, np.nan), F=np.nan, p=np.nan)
               for k, v in ss.items()]
        note = ("칸당 관측이 1개다 — 상호작용과 오차를 분리할 수 없어 F·p 를 내지 않는다. "
                "split_replicates 로 칸 안 반복을 만들어라.")
    else:
        ss["상호작용"] = inter
        ms_e = resid / df_resid
        out = []
        for k in (a, b, "상호작용"):
            F_ = (ss[k] / df_[k]) / ms_e if ms_e > 1e-12 and df_[k] else np.nan
            p_ = float(st.f.sf(F_, df_[k], df_resid)) if np.isfinite(F_) else np.nan
            out.append(dict(요인=k, SS=round(ss[k], 4),
                            eta2=round(ss[k] / sst, 3) if sst > 1e-12 else np.nan,
                            자유도=df_[k], F=round(F_, 2) if np.isfinite(F_) else np.nan,
                            p=round(p_, 5) if np.isfinite(p_) else np.nan))
        out.append(dict(요인="잔차", SS=round(resid, 4),
                        eta2=round(resid / sst, 3) if sst > 1e-12 else np.nan,
                        자유도=df_resid, F=np.nan, p=np.nan))
        note = f"칸당 반복 {rep}개 · 잔차 자유도 {df_resid}"
    T = pd.DataFrame(out)
    T.attrs["note"] = note
    return T
