"""FvLinker · 소표본 링커-물성 추론 핵심 (v12 7절이 그대로 쓰는 코드)

이 파일은 **노트북에서 떼어내 여기서 시험하기 위한** 사본이다.
v12 노트북 7절은 같은 함수를 그대로 들고 있다. 여기서 바뀌면 거기도 바꾼다.

설계 전제 — 바꾸지 마라, 이게 전부다
------------------------------------
n = 26 구성체, 그중 링커가 3종 이상인 블록은 5개뿐이다.
독립 단위는 구성체가 아니라 **블록 5개**다. 자유도를 그렇게 세야 한다.
따라서:
  · 타깃은 블록 안에서 중심화한다 (블록 절편을 고정효과로 소거).
  · 검증은 블록 단위 leave-one-block-out 뿐이다.
  · 유의성은 **블록 안 링커 라벨 순열**로만 낸다. 모형이 뱉는 p 는 여기서 의미가 없다.
  · 특징 수는 사전 등록으로 5개 이하로 묶는다.

속도에 관하여
------------
순열 2000회 × LOBO 5폴드 = 1만 번 적합이고, 검정력 곡선은 그 200배다.
그래서 **안쪽 고리에 pandas 를 절대 두지 않는다** — 블록 색인과 쌍 목록을 한 번만
만들어 두고(`Design`), 그 뒤로는 numpy 만 돈다. 순수 pandas 판보다 100배쯤 빠르다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as st


# ─────────────────────────────────────────────────────────────────────────────
# 0. 타깃
# ─────────────────────────────────────────────────────────────────────────────
def logit(p, eps: float = 1e-3) -> np.ndarray:
    """분율 → 로짓. 0/1 에서 눌리는 것을 편다.

    HMW·단량체는 **분율**이다. 0.02 → 0.04 와 0.50 → 0.52 는 같은 2%p 지만
    응집 평형에서는 전자가 훨씬 큰 변화다. 로짓이 그걸 선형으로 편다.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def as_fraction(v) -> np.ndarray:
    """퍼센트로 들어왔으면 분율로 내린다. 1.5 를 넘는 값이 있으면 퍼센트로 본다."""
    v = np.asarray(v, dtype=float)
    return v / 100.0 if np.nanmax(np.abs(v)) > 1.5 else v


def make_target(F: pd.DataFrame, col: str, block: str = "블록",
                higher_is_worse: bool = True) -> pd.Series:
    """블록 안에서 중심화한 로짓 타깃.

        y_ij = logit(p_ij) − mean_j( logit(p_·j) )

    블록마다 항체도 정제 공정도 달라서 **절대값은 비교할 수 없다.**
    블록 평균을 빼면 남는 것은 "같은 항체에서 링커를 바꿨을 때의 차이" 뿐이고,
    그게 이 실험이 답할 수 있는 유일한 질문이다.
    (계량경제의 within-transformation = 블록 고정효과와 정확히 같다.)

    higher_is_worse=True 면 부호를 뒤집어 **클수록 좋음**으로 맞춘다.
    HMW(응집체 %)가 그렇다 — 낮을수록 좋은 물성이다.
    """
    y = pd.Series(logit(as_fraction(F[col].values)), index=F.index)
    if higher_is_worse:
        y = -y
    return y.groupby(F[block]).transform(lambda v: v - v.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 1. 설계 — 블록 색인과 쌍 목록을 **한 번만** 만든다
# ─────────────────────────────────────────────────────────────────────────────
class Design:
    """블록 구조를 numpy 로 굳혀 둔 것. 순열·검정력의 안쪽 고리가 이것만 본다.

    blocks : 블록마다 그 블록의 행 번호 배열
    pairs  : (i, j) 쌍 — 블록 **안**의 모든 조합. 블록을 가로지르는 쌍은 없다.
    """

    def __init__(self, F: pd.DataFrame, block: str = "블록"):
        self.n = len(F)
        codes = pd.factorize(F[block].values)[0]
        self.codes = codes
        self.blocks = [np.where(codes == b)[0] for b in range(codes.max() + 1)]
        self.blocks = [b for b in self.blocks if len(b) >= 2]
        self.pairs = np.array([(i, j) for b in self.blocks
                               for i, j in combinations(b, 2)], dtype=int)
        if not len(self.pairs):
            raise ValueError("블록 안 쌍이 하나도 없다 — 블록마다 구성체가 1개뿐인가?")

    def center(self, y: np.ndarray) -> np.ndarray:
        """블록 안 중심화 (numpy 판)."""
        out = np.asarray(y, float).copy()
        for b in self.blocks:
            out[b] -= out[b].mean()
        return out

    def permute(self, y: np.ndarray, rng) -> np.ndarray:
        """블록 **안에서만** 섞는다. 블록 구조와 주변 분포는 그대로 둔다."""
        out = np.asarray(y, float).copy()
        for b in self.blocks:
            out[b] = out[b][rng.permutation(len(b))]
        return out

    def permute_rows(self, X: np.ndarray, rng) -> np.ndarray:
        """특징 **행**을 블록 안에서 섞는다 (증분 검정용). 행 안의 열 구성은 유지."""
        out = np.asarray(X, float).copy()
        for b in self.blocks:
            out[b] = out[b][rng.permutation(len(b))]
        return out


def concordance(pred: np.ndarray, y: np.ndarray, D: Design):
    """블록 안의 **모든 링커 쌍**에 대해 예측 순서가 실측 순서와 맞는가.

    블록당 링커가 4종이면 Spearman 은 {±1, ±0.8, ±0.4, ±0.2, 0} 밖에 못 낸다.
    대신 블록 안 쌍(4종이면 6쌍)을 전부 모으면 5블록 × 6쌍 = 30쌍이 되고,
    귀무가설에서 각 쌍이 1/2 로 맞으므로 **이항분포**라는 해석이 붙는다.

    반환: (맞은 쌍, 센 쌍, 일치도). NaN 예측이 낀 쌍과 동점은 세지 않는다.
    """
    i, j = D.pairs[:, 0], D.pairs[:, 1]
    dp, dy = pred[i] - pred[j], y[i] - y[j]
    ok = np.isfinite(dp) & np.isfinite(dy) & (np.abs(dp) > 1e-12) & (np.abs(dy) > 1e-12)
    n = int(ok.sum())
    if not n:
        return 0, 0, np.nan
    a = int((np.sign(dp[ok]) == np.sign(dy[ok])).sum())
    return a, n, a / n


def binom_p(agree: int, n: int) -> float:
    """일치도의 단측 이항 p — '우연보다 잘 맞나'. 순열 p 의 정합성 확인용이다."""
    if not n:
        return np.nan
    return float(st.binomtest(agree, n, 0.5, alternative="greater").pvalue)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Leave-One-Block-Out (numpy 판)
# ─────────────────────────────────────────────────────────────────────────────
def _ridge_fit_predict(Xtr, ytr, Xte, alpha):
    """학습 폴드에서 **표준화까지 적합**하고 시험 폴드를 예측한다.

    ★ 전체로 표준화하면 시험 블록의 평균·분산이 학습에 새어 들어간다.
      n=26 에서는 그것만으로 일치도가 눈에 띄게 올라간다 — 반드시 폴드 안에서.
    """
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Xtr - mu) / sd
    yc = ytr - ytr.mean()
    k = Z.shape[1]
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ yc)
    return ((Xte - mu) / sd) @ coef + ytr.mean(), coef


def lobo_predict(X: np.ndarray, y: np.ndarray, D: Design, alpha: float = 1.0):
    """블록 하나를 빼고 학습해 그 블록을 예측한다. 모든 블록에 대해 반복.

    절편은 블록 안 쌍 비교에서 상쇄되므로 블록 절편을 따로 추정하지 않는다.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    coefs = []
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    for b in D.blocks:
        te = b[ok[b]]
        tr = np.array([i for i in range(len(y)) if ok[i] and i not in set(b.tolist())])
        if len(te) < 2 or len(tr) < X.shape[1] + 2:
            continue
        p, c = _ridge_fit_predict(X[tr], y[tr], X[te], alpha)
        pred[te] = p
        coefs.append(c)
    return pred, (np.array(coefs) if coefs else np.zeros((0, X.shape[1])))


def lobo_score(X, y, D: Design, alpha: float = 1.0):
    """LOBO 예측 → 블록 안 쌍 일치도. 이 한 숫자가 모든 검정의 통계량이다."""
    pred, _ = lobo_predict(X, y, D, alpha)
    return concordance(pred, np.asarray(y, float), D)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 순열 검정 — 이 n 에서는 이것 말고 유의성을 말할 방법이 없다
# ─────────────────────────────────────────────────────────────────────────────
def perm_test_signal(X, y, D: Design, alpha=1.0, n_perm=2000, seed=0):
    """귀무가설 A: **링커와 물성 사이에 아무 관계도 없다.**

    블록 **안에서** y 를 섞는다. 블록 구조·블록별 분포·특징행렬은 그대로 두고
    링커↔물성 대응만 끊는다. 그리고 LOBO 전체를 다시 돈다 — 표준화도 폴드마다
    다시 하므로 검정이 파이프라인 전체를 포함한다.

    블록당 링커 4종이면 블록 안 배열이 4! = 24, 5블록이면 24⁵ ≈ 800만 가지라
    2000회 무작위 추출로 충분하다 (정확검정을 근사한다).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    a, n, obs = lobo_score(X, y, D, alpha)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = lobo_score(X, D.permute(y, rng), D, alpha)[2]
    fin = np.isfinite(null)
    p = float((np.sum(null[fin] >= obs) + 1) / (fin.sum() + 1))
    return dict(일치=f"{a}/{n}", 일치도=round(float(obs), 3),
                귀무평균=round(float(np.nanmean(null)), 3),
                귀무95=round(float(np.nanpercentile(null[fin], 95)), 3),
                순열p=round(p, 4), 이항p=round(binom_p(a, n), 4),
                _null=null, _obs=obs)


def perm_test_increment(Xbase, Xadd, y, D: Design, alpha=1.0, n_perm=2000, seed=0):
    """귀무가설 B: **앙상블 특징은 링커 길이 위에 아무것도 더하지 않는다.**

    이게 이 연구의 진짜 질문이다. "상관이 있다"는 대개 "긴 링커가 나쁘다"를
    다시 발견한 것일 뿐이기 때문이다.

    그래서 y 를 섞지 않는다. **앙상블 특징 행만** 블록 안에서 섞는다 —
    길이와 y 의 관계는 그대로 둔 채 앙상블 특징이 링커에 붙어 있다는 사실만 끊는다.
    통계량은 Δ일치도 = (길이+앙상블) − (길이만) 이다.
    """
    Xb = np.atleast_2d(np.asarray(Xbase, float).T).T
    Xa = np.atleast_2d(np.asarray(Xadd, float).T).T
    y = np.asarray(y, float)
    b = lobo_score(Xb, y, D, alpha)[2]
    f = lobo_score(np.hstack([Xb, Xa]), y, D, alpha)[2]
    obs = f - b
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        Xp = D.permute_rows(Xa, rng)
        null[t] = lobo_score(np.hstack([Xb, Xp]), y, D, alpha)[2] - b
    fin = np.isfinite(null)
    p = float((np.sum(null[fin] >= obs) + 1) / (fin.sum() + 1))
    return dict(길이만=round(float(b), 3), 길이_앙상블=round(float(f), 3),
                증분=round(float(obs), 3),
                귀무평균=round(float(np.nanmean(null)), 3),
                순열p=round(p, 4), _null=null, _obs=obs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 검정력 — 귀무 결과가 정보인지 그냥 표본이 모자란 것인지 가른다
# ─────────────────────────────────────────────────────────────────────────────
def power_curve(x, D: Design, betas=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0), n_sim=200,
                n_perm=300, alpha=1.0, seed=0, sigma=1.0) -> pd.DataFrame:
    """실제 특징값은 그대로 두고 **라벨만** 알려진 효과크기로 심어 만든다.

        y = β·z(특징) + ε,   ε ~ N(0, σ²),   그다음 블록 안 중심화

    그리고 진짜 분석(LOBO + 순열)을 통째로 돌려 **검출률**을 센다.
    β=0 행이 거짓양성률이다 — 0.05 근처여야 검정이 정직한 것이다.

    이 표가 답하는 것: "상관이 안 나왔다"가 **없다는 증거**인지,
    아니면 블록 5개로는 애초에 못 보는 크기였는지.

    β 의 단위: 특징 1 표준편차당 타깃 몇 σ. β=1 은 잡음과 같은 크기의 효과다.
    """
    x = np.asarray(x, float)
    z = (x - np.nanmean(x)) / (np.nanstd(x) if np.nanstd(x) > 1e-12 else 1.0)
    X = z[:, None]
    rng = np.random.default_rng(seed)
    rows = []
    for b in betas:
        hit = 0
        for _ in range(n_sim):
            y = D.center(b * z + rng.normal(0, sigma, len(z)))
            r = perm_test_signal(X, y, D, alpha=alpha, n_perm=n_perm,
                                 seed=int(rng.integers(1 << 30)))
            hit += int(r["순열p"] < 0.05)
        rows.append(dict(효과크기=b, 검출률=round(hit / n_sim, 3), 시행=n_sim))
    return pd.DataFrame(rows)


def min_detectable_effect(P: pd.DataFrame, target: float = 0.8):
    """검정력 곡선에서 목표 검출률에 닿는 최소 효과크기 (선형 보간)."""
    P = P.sort_values("효과크기")
    b, d = P.효과크기.values, P.검출률.values
    hit = np.where(d >= target)[0]
    if not len(hit):
        return np.inf
    k = hit[0]
    if k == 0:
        return float(b[0])
    x0, x1, y0, y1 = b[k - 1], b[k], d[k - 1], d[k]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(b[k])


# ─────────────────────────────────────────────────────────────────────────────
# 5. 교란 원장 — 특징이 길이의 변장인지 먼저 본다
# ─────────────────────────────────────────────────────────────────────────────
def confound_ledger(F: pd.DataFrame, feats, length_col="길이", y=None,
                    block="블록") -> pd.DataFrame:
    """각 특징이 링커 **길이**와 얼마나 같은 것인지 표로 낸다.

    |r| > 0.9 면 그 특징은 길이의 다른 이름이다 — n=20 에서 둘을 가를 수 없다.
    블록 안 상관(블록 효과를 뺀 뒤)을 같이 내는 이유는, 블록 간 차이가 전체 상관을
    부풀리기 때문이다. 판정은 **블록 안** 상관으로 한다.
    """
    rows = []
    cw = F.groupby(block)[length_col].transform(lambda v: v - v.mean()).values
    for c in feats:
        v = F[c].astype(float)
        vw = v.groupby(F[block]).transform(lambda s: s - s.mean()).values
        m = np.isfinite(v.values) & np.isfinite(F[length_col].values.astype(float))
        mw = np.isfinite(vw) & np.isfinite(cw)
        rw = (np.corrcoef(vw[mw], cw[mw])[0, 1]
              if mw.sum() > 2 and np.std(vw[mw]) > 1e-12 and np.std(cw[mw]) > 1e-12
              else np.nan)
        d = dict(특징=c,
                 r_길이=round(float(np.corrcoef(v[m], F[length_col].values[m])[0, 1]), 2)
                 if m.sum() > 2 else np.nan,
                 rho_길이=round(float(st.spearmanr(v[m], F[length_col].values[m])[0]), 2)
                 if m.sum() > 2 else np.nan,
                 블록내r_길이=round(float(rw), 2) if np.isfinite(rw) else np.nan)
        if y is not None:
            yy = np.asarray(y, float)
            mm = m & np.isfinite(yy)
            d["rho_실측"] = (round(float(st.spearmanr(v[mm], yy[mm])[0]), 2)
                            if mm.sum() > 2 else np.nan)
        rw_ = abs(d["블록내r_길이"]) if np.isfinite(d["블록내r_길이"]) else 0.0
        d["판정"] = ("길이의 변장" if rw_ > 0.9 else
                     "길이와 강하게 얽힘" if rw_ > 0.7 else "독립적")
        rows.append(d)
    return pd.DataFrame(rows)


def partial_spearman(x, y, z) -> float:
    """z 를 통제한 x-y 편상관 (Spearman).

    ★ v11 의 `차 = rho_실측 − rho_길이 × rho(길이,실측)` 는 편상관이 **아니다** —
      분모가 빠져 있다. 올바른 식은 이것이다:

          r_xy·z = (r_xy − r_xz·r_yz) / sqrt((1 − r_xz²)(1 − r_yz²))

      분모는 항상 1 이하라 나누면 값이 **커진다**. 즉 v11 의 '차' 는 편상관을
      체계적으로 **과소평가**하고, 길이 상관이 강할수록 더 그렇다.
      (0 인지 아닌지는 두 식이 같이 판정하지만, 크기는 비교할 수 없다.)
    """
    rx = st.spearmanr(x, y)[0]
    rz1 = st.spearmanr(x, z)[0]
    rz2 = st.spearmanr(y, z)[0]
    den = np.sqrt(max(1 - rz1 ** 2, 0) * max(1 - rz2 ** 2, 0))
    return float((rx - rz1 * rz2) / den) if den > 1e-12 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 6. 특징 불확실성 — 특징 하나하나가 150 프레임의 통계량이다
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_feature_se(frames: pd.DataFrame, key_cols, value_col, statfn,
                         n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """프레임 재표집으로 구성체별 특징의 표준오차.

    특징은 150개 표본의 통계량이라 오차가 있다. 그 오차가 크면 회귀계수가 0 쪽으로
    눌린다 (errors-in-variables 감쇠) — 즉 **상관이 없어 보이는 것이 진짜 없어서가
    아니라 특징이 시끄러워서**일 수 있다. 이 표가 그 구분을 준다.

    읽는 법: SE 가 구성체 간 특징 흩어짐(SD)보다 크면 그 특징은 이 표본 크기에서
    쓸 수 없다. 프레임을 더 뽑아야 한다.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k, g in frames.groupby(list(key_cols), sort=False):
        v = g[value_col].dropna().values.astype(float)
        if len(v) < 5:
            continue
        bs = np.array([statfn(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
        d = dict(zip(key_cols, k if isinstance(k, tuple) else (k,)))
        d.update({value_col: float(statfn(v)), "SE": float(np.std(bs, ddof=1)),
                  "n프레임": len(v)})
        rows.append(d)
    T = pd.DataFrame(rows)
    if len(T):
        sd = T[value_col].std(ddof=1)
        T["SE대SD"] = (T.SE / sd).round(2) if sd > 1e-12 else np.nan
        T["판정"] = np.where(T.SE대SD > 0.5, "★ 잡음이 신호만큼 크다", "쓸 만하다")
    return T


def attenuation_note(se: float, sd_between: float) -> str:
    """측정오차에 의한 상관 감쇠 배율을 말로 돌려준다.

    관측 상관 ≈ 참 상관 × sqrt(신뢰도),  신뢰도 = 1 − (SE²/SD_between²).
    """
    if not np.isfinite(se) or not np.isfinite(sd_between) or sd_between <= 1e-12:
        return "판정 불가"
    rel = max(1 - (se ** 2) / (sd_between ** 2), 0.0)
    return (f"신뢰도 {rel:.2f} → 관측 상관이 참값의 {np.sqrt(rel):.2f}배로 눌린다"
            + ("  ★ 프레임을 더 뽑아라" if rel < 0.7 else ""))
