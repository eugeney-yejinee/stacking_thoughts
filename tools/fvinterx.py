"""fvinterx — 7절-C 계면강도 × 열림 상호작용 검정 (FvFlow v12).

fvml 과 같은 규약: 타깃은 항체 안 중심화, 독립 단위는 항체, 유의성은 순열만.
fvml 과 다른 점: 통계량이 **항체별 기울기 대 계면강도** 이지 Δ일치도가 아니다.
Δ일치도는 이 대립가설에 대해 원리적으로 거의 무력하다 (아래 주석).

모형 (항체 j, 링커 i):
    y_ij = α_j + β·o_ij + γ·(S_j·o_ij) + ε_ij
항체 안 중심화 뒤:
    ỹ_ij = (β + γ·S_j)·õ_ij + ε̃_ij
→ 자료가 말할 수 있는 것은 **항체별 기울기 b_j = β + γ·S_j** 뿐이다.
   S_j 의 주효과는 정의상 소거된다. γ 는 b_j 를 S_j 에 회귀해서만 나온다.

정확 분산 (S 와 o 를 조건부로):
    Var(γ̂) = Σ_j w_j²d_j²(σ²/Sxx_j + τ²) / (Σ_j w_j d_j²)²,
    w_j = Sxx_j/σ², d_j = S_j − S̄_w, Sxx_j = Σ_i õ_ij².
    τ² = 0 이면 Var(γ̂) = 1/Σ_j w_j d_j².  모의로 비율 1.007 확인.
"""
from __future__ import annotations
import numpy as np, pandas as pd, math
from itertools import permutations
from scipy import stats as st


def _rho(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(st.spearmanr(a, b).statistic)

__all__ = ["antibody_slopes", "interaction_test", "slope_heterogeneity",
           "identifiability_report", "interaction_power"]


# ─────────────────────────────────────────────────────────────── 1. 항체별 기울기
def antibody_slopes(y, x, groups):
    """항체별 기울기 b_j 와 그 정밀도. y 는 이미 항체 안 중심화된 타깃.

    반환 DataFrame: 항체 · k · b · Sxx · se · r_within · mean_x · sd_x
    σ² 는 **항체끼리 풀링**한다. 항체 하나에 링커가 3종이면 잔차 자유도가 1 뿐이라
    항체별 SE 는 못 믿는다 — 풀링이 유일하게 안정적인 선택이다.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    g = pd.Series(groups).astype(str).values
    rows, rss, dof = [], 0.0, 0
    for ab in pd.unique(g):
        m = (g == ab) & np.isfinite(y) & np.isfinite(x)
        if m.sum() < 3:                      # 기울기 + 절편 + 잔차 1 개는 있어야 한다
            continue
        yy = y[m] - y[m].mean(); xx = x[m] - x[m].mean()
        sxx = float(xx @ xx)
        if sxx < 1e-12:                      # 이 항체 안에서 특징이 상수 → 기울기 없음
            continue
        b = float(xx @ yy / sxx)
        r = yy - b * xx
        rss += float(r @ r); dof += m.sum() - 2
        sy = yy.std(ddof=0)
        rows.append(dict(항체=ab, k=int(m.sum()), b=b, Sxx=sxx,
                         r_within=float(b * xx.std(ddof=0) / sy) if sy > 1e-12 else np.nan,
                         mean_x=float(x[m].mean()), sd_x=float(x[m].std(ddof=1)),
                         mean_y=float(np.asarray(y)[m].mean())))
    T = pd.DataFrame(rows)
    if not len(T):
        raise ValueError("기울기를 낼 수 있는 항체가 없다 (항체당 링커 3종 이상 필요)")
    sig2 = rss / dof if dof > 0 else np.nan
    T["se"] = np.sqrt(sig2 / T.Sxx.values)
    T.attrs["sigma2"] = sig2; T.attrs["dof"] = dof
    return T


# ─────────────────────────────────────────────────────────────── 2. 통계량
def _wls_t(b, S, w):
    sw = w.sum()
    d = S - (w * S).sum() / sw
    den = float((w * d * d).sum())
    if den <= 1e-300:
        return 0.0, 0.0
    bw = (w * b).sum() / sw
    gam = float((w * d * (b - bw)).sum() / den)
    return gam, gam * math.sqrt(den)          # (γ̂, 스튜던트화 통계량)


def _residualise(v, C):
    C = C - C.mean(); v = v - v.mean()
    s = float(C @ C)
    return v - (float(C @ v) / s) * C if s > 1e-12 else v


# ─────────────────────────────────────────────────────────────── 3. 상호작용 검정
def interaction_test(slopes: pd.DataFrame, S, *, covariate=None, sided="greater",
                     max_exact=5040, n_rand=20000, seed=0):
    """항체 수준 S 라벨만 재배치하는 정확 순열 검정.

    귀무가설 H0 : 항체별 기울기 벡터 (b_1..b_m) 의 결합분포가 **S 라벨의 치환에
    대해 불변**이다. 즉 어떤 항체가 어느 S 를 갖는지가 기울기와 무관하다.
    (γ=0 이고, 게다가 S 가 항체의 정밀도·링커수·기준선과도 무관해야 성립한다.
     그래서 identifiability_report 를 반드시 같이 읽어야 한다.)

    ★ 항체 **안** 링커 라벨은 절대 섞지 않는다. 섞으면 β 까지 지워져
      귀무가 γ=0 이 아니라 β=γ=0 이 되고, 주효과만 있어도 기각한다.

    covariate 를 주면 Freedman–Lane 로 그 항체 수준 교란(예: 항체별 평균 열림분율)
    을 통제한다. S 의 잔차만 섞고 적합값은 되돌려 붙이므로 S–covariate 관계가
    순열 안에서 보존된다. 모의: 교란 상관 1.0 에서도 거짓양성 0.052 (통제 안 하면 0.412).
    """
    b = slopes.b.values.astype(float)
    S = np.asarray(S, float)
    m = len(b)
    assert len(S) == m, "S 의 길이가 항체 수와 다르다"
    w = 1.0 / (slopes.se.values ** 2)          # = Sxx_j/σ̂², y-정보에 의존하지 않는 가중
    C = None if covariate is None else np.asarray(covariate, float)

    exact = math.factorial(m) <= max_exact
    if exact:
        P = np.array(list(permutations(range(m))))
    else:
        rng = np.random.default_rng(seed)
        P = np.vstack([np.arange(m), [rng.permutation(m) for _ in range(n_rand - 1)]])

    if C is None:
        bb = b
        def make(p): return S[p]
    else:                                       # Freedman–Lane
        Cc = C - C.mean()
        fit = (float(Cc @ (S - S.mean())) / float(Cc @ Cc)) * Cc
        Sr = (S - S.mean()) - fit
        bb = _residualise(b, C)
        def make(p): return Sr[p] + fit

    T = np.empty(len(P)); G = np.empty(len(P))
    for t, p in enumerate(P):
        Sp = make(p)
        if C is not None:
            Sp = _residualise(Sp, C)
        G[t], T[t] = _wls_t(bb, Sp, w)
    obs_g, obs_t = G[0], T[0]
    if sided == "greater":   p = float((T >= obs_t - 1e-12).mean())
    elif sided == "less":    p = float((T <= obs_t + 1e-12).mean())
    else:                    p = float((np.abs(T) >= abs(obs_t) - 1e-12).mean())

    rho = _rho(b, S) if m > 2 else np.nan
    se_g = abs(obs_g / obs_t) if abs(obs_t) > 1e-12 else np.nan
    return dict(항체수=m, gamma=round(obs_g, 4), se_gamma=round(se_g, 4),
                t=round(obs_t, 3), spearman_rho=round(rho, 3),
                순열p=round(p, 5), 정확=bool(exact),
                최소가능p=round(1.0 / len(P), 5), 순열수=len(P),
                통제=None if covariate is None else "Freedman–Lane",
                _null=T, _obs=obs_t)


# ─────────────────────────────────────────────────────────────── 4. 사전 점검
def slope_heterogeneity(y, x, groups):
    """항체별 기울기가 애초에 다른가 — 공통기울기 대 개별기울기 F 검정.
    (항체 절편은 양쪽 모두에 들어 있다. df = (m−1, n−2m).)

    ★ 이 결과로 상호작용 검정을 **막지 마라.** 모의에서 γ=1 일 때
      상호작용 검정 0.523 vs 이 F 검정 0.386 이고, 게이트를 걸면 0.265 로 떨어진다.
      덜 예민한 전방위 검정으로 더 예민한 방향성 검정을 막는 셈이다.
      이건 보고용 맥락이지 관문이 아니다.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    g = pd.Series(groups).astype(str).values
    parts = []
    for ab in pd.unique(g):
        msk = (g == ab) & np.isfinite(y) & np.isfinite(x)
        if msk.sum() < 3: continue
        yy = y[msk] - y[msk].mean(); xx = x[msk] - x[msk].mean()
        if float(xx @ xx) < 1e-12: continue
        parts.append((yy, xx))
    m = len(parts); n = sum(len(p[0]) for p in parts)
    if m < 2 or n - 2 * m <= 0:
        return dict(F=np.nan, p=np.nan, 메모="자유도가 없다")
    sxx = sum(float(xx @ xx) for _, xx in parts)
    sxy = sum(float(xx @ yy) for yy, xx in parts)
    bc = sxy / sxx
    rss_r = sum(float((yy - bc * xx) @ (yy - bc * xx)) for yy, xx in parts)
    rss_f = sum(float((yy - (xx @ yy / (xx @ xx)) * xx) @ (yy - (xx @ yy / (xx @ xx)) * xx))
                for yy, xx in parts)
    df1, df2 = m - 1, n - 2 * m
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return dict(F=round(float(F), 3), df=(df1, df2), p=round(float(st.f.sf(F, df1, df2)), 4))


def identifiability_report(slopes: pd.DataFrame, S, covariates: dict | None = None):
    """검정을 읽기 **전에** 찍는 y-무관 진단. 전부 y 를 안 쓰므로 관문으로 써도 된다.

    · S 가 항체 사이에서 실제로 변하는가 (CV, 범위, 레버리지)
    · S 가 항체별 정밀도·링커수와 얽혀 있는가 → 순열 교환가능성이 깨진다
    · S 가 다른 항체 수준 변수(평균 열림분율·CDR-H3 길이·순전하…)와 구별되는가
      → 항체 5개면 순서가 120가지뿐이라 **우연히 같은 순서인 변수가 반드시 있다**
    """
    S = np.asarray(S, float); m = len(S)
    d = S - S.mean(); Sxx = float(d @ d)
    lev = (d ** 2) / Sxx if Sxx > 1e-12 else np.full(m, np.nan)
    w = 1.0 / slopes.se.values ** 2
    rows = [dict(항목="S 변동계수", 값=round(float(np.std(S, ddof=1) / abs(np.mean(S))), 4)
                 if abs(np.mean(S)) > 1e-12 else np.nan,
                 판정="S 가 사실상 상수다 — 상호작용을 물을 수 없다"
                 if np.std(S, ddof=1) < 1e-9 else ""),
            dict(항목="γ 유효정보 Σw_j d_j²", 값=round(float((w * d * d).sum()), 3),
                 판정=f"→ se(γ) = {1/math.sqrt(max((w*d*d).sum(),1e-300)):.3f}"),
            dict(항목="최대 레버리지 항체", 값=f"{slopes.항체.values[int(np.argmax(lev))]} "
                 f"({lev.max():.2f})",
                 판정="★ 한 항체가 S축을 지배한다 — 5점 회귀가 그 점 하나다"
                 if lev.max() > 0.6 else ""),
            dict(항목="ρ(S, 항체별 SE)", 값=round(_rho(S, slopes.se), 3),
                 판정="★ 정밀도가 S 를 따라간다 — 순열 교환가능성이 의심스럽다"
                 if abs(_rho(S, slopes.se) or 0) > 0.8 else ""),
            dict(항목="ρ(S, 링커수 k)", 값=round(_rho(S, slopes.k), 3),
                 판정="★ 설계 불균형이 S 와 얽혔다" if (abs(_rho(S, slopes.k)) > 0.8 if np.isfinite(_rho(S, slopes.k)) else False) else "")]
    for nm, v in (covariates or {}).items():
        r = _rho(S, v)
        rows.append(dict(항목=f"ρ(S, {nm})", 값=round(r, 3),
                         판정="★★ 이 변수와 S 를 가를 수 없다 — 유의해도 '계면 때문' 이라 못 쓴다"
                         if np.isfinite(r) and abs(r) > 0.9 else ("얽힘" if np.isfinite(r) and abs(r) > 0.7 else "")))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────── 5. 검정력
def interaction_power(m, k, gammas=(0, 0.5, 1, 1.5, 2, 3), n_sim=2000, tau=0.0,
                      lamS=1.0, lamO=1.0, alpha=0.05, seed=0, sigma=1.0):
    """γ 단위: 타깃 σ / (특징 1 SD × S 1 SD).  lamS·lamO 는 신뢰도(측정오차).
    감쇠는 정확히 γ_obs = γ·lamS·lamO 다 (모의 확인, 아래 표).
    """
    rng = np.random.default_rng(seed)
    P = np.array(list(permutations(range(m)))) if math.factorial(m) <= 5040 else \
        np.vstack([np.arange(m), [rng.permutation(m) for _ in range(1999)]])
    rows = []
    for gma in gammas:
        hit = 0
        for _ in range(n_sim):
            S = rng.normal(size=m); S -= S.mean()
            u = rng.normal(0, tau, m)
            o = rng.normal(size=(m, k))
            Y = (1.0 + gma * S + u)[:, None] * o + rng.normal(0, sigma, (m, k))
            Y -= Y.mean(1, keepdims=True)
            oo = o + (rng.normal(0, math.sqrt((1 - lamO) / lamO), (m, k)) if lamO < 1 else 0)
            So = S + (rng.normal(0, math.sqrt((1 - lamS) / lamS), m) if lamS < 1 else 0)
            gidx = np.repeat(np.arange(m), k)
            T = antibody_slopes(Y.ravel(), oo.ravel(), gidx)
            hit += int(interaction_test(T, So)["순열p"] <= alpha)
        rows.append(dict(gamma=gma, 검출률=round(hit / n_sim, 3),
                         유효gamma=round(gma * lamS * lamO, 3)))
    return pd.DataFrame(rows)
