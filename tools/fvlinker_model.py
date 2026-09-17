#!/usr/bin/env python3
"""FvLinker 응집 모델 v1 — 저pH 용출에서의 **cis/trans 재짝짓기 경쟁**.

────────────────────────────────────────────────────────────────────────────
이론
────────────────────────────────────────────────────────────────────────────
이 분자에서 scFv 의 VH–VL 짝만 사슬간 이황화로 안 잡혀 있다 (Fab 은 CH1–CL +
사슬간 S–S 가, Fc 는 힌지가 받쳐준다). scFv 를 붙잡는 건 **링커뿐이다.**
그래서 Protein A 저pH 용출(pH 3.0~3.6)에서 scFv 가 제일 먼저 열린다.

열린 VH·VL 은 다시 짝을 찾는데 갈래가 둘뿐이다:
  cis   — 제 짝과 다시.      속도 ∝ c_eff    (링커가 정한다)
  trans — 다른 분자의 짝과.  속도 ∝ c_bulk   (용출액 농도가 정한다)
중화하면 계면이 다시 묻혀 **되돌릴 수 없다.** trans 로 간 것이 HMW 다.

    HMW = HMW₀ᵢ + fᵢ · c_bulk / (c_eff + c_bulk)

    fᵢ, HMW₀ᵢ  = 항체 성질 (계면 강도)      → **항체 간**
    c_eff       = 링커 성질                  → **항체 안**  ← 검정 대상
    c_bulk      = 공정 성질

평형이 아니라 **동역학적 분배**라는 게 요점이다. 평형 f_open ≈ 7e-4 로는
HMW 4~43% 가 안 나오지만, 분배비는 0~1 아무 값이나 된다.

    ⟨r²⟩ = n·l·b            (Gaussian 사슬, 조성을 못 본다)
    c_eff ∝ (3/2π⟨r²⟩)^1.5 · exp(−3·r_ab²/2⟨r²⟩)
    n* = r_ab²/(l·b)        (c_eff 가 최대가 되는 길이)

n < n* 이면 **길수록 c_eff 가 커지고 → HMW 가 낮아진다.**
n > n* 이면 부호가 뒤집힌다. 이게 이 모델의 유일한 정성 예측이고,
한 항체만 반대 방향인 실측을 설명할 수 있는 **유일한 손잡이**다.

────────────────────────────────────────────────────────────────────────────
이 스크립트가 하는 일 — 전부 자동, 물어보는 것 없음
────────────────────────────────────────────────────────────────────────────
  [0] 환경 점검        무엇이 있고 무엇이 없는지 **먼저 찍는다**
  [1] 데이터           엑셀 + anchor_span.csv (없으면 명시하고 진행)
  [2] 사슬 치수        ALBATROSS 가 깔려 있으면 **서열별** Re, 없으면 길이만
  [3] c_eff
  [4] 반전 판정        길이 추세가 반대인 항체의 r_ab 가 조건을 만족하나
  [5] 순위 검정        정확 순열 (항체 안에서만 섞는다)
  [6] ★ 진폭 진단      실측 폭을 내려면 f 가 몇 %여야 하나. >100%면 **불가능**
  [7] b 프로파일       자유변수는 b 하나뿐. 27쌍으로 잡는다
  [8] 예측 곡선        n=10~45. **다음 라운드 설계가 여기서 나온다**
  [9] 판정문

실행:
    python3 fvlinker_model.py                 # Drive 기본 경로
    python3 fvlinker_model.py <엑셀> [앵커csv]
"""
from __future__ import annotations

import glob
import itertools
import os
import re
import sys
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd

# ── 상수 ────────────────────────────────────────────────────────────────────
DRIVE = "/content/drive/MyDrive/FvTwist"
IN = f"{DRIVE}/input"
OUT = f"{DRIVE}/fvflow"          # ★ build_v12.py 와 같은 경로. 이름이 FvLinker 여도 안 바꾼다
RES_L = 3.5                      # Å / 잔기 (contour rise)
A3_TO_M = 1661.0                 # 1 Å⁻³ = 1661 M
B_GRID = np.arange(6.0, 22.1, 0.5)   # Kuhn 길이 탐색 격자 (Å)
OBS_KEYS = ("hmw", "monomer", "단량체", "purity", "순도", "수율", "생산",
            "titer", "sec", "quantific", "supernatant")

W = 78
bar = lambda c="─": c * W
def head(t, c="="):
    print("\n" + c * W); print(t); print(c * W)


# ── [0] 환경 ────────────────────────────────────────────────────────────────
def check_albatross():
    """ALBATROSS(sparrow)가 쓸 수 있나. **없어도 계속 간다.**

    PyPI 의 `sparrow` 는 전혀 다른 패키지(RDF/SPARQL)다. 진짜는 GitHub 이다:
        pip install git+https://github.com/idptools/sparrow.git
    API 이름이 판마다 다를 수 있어 여러 모양을 다 시도하고, 실패하면
    **조용히 넘어가지 않고 무엇이 왜 안 됐는지 찍는다.**
    """
    try:
        import sparrow
    except Exception as e:
        return None, f"sparrow 없음 ({type(e).__name__}) — 길이 기반 Gaussian 으로 간다"
    if not hasattr(sparrow, "Protein"):
        return None, ("PyPI 의 다른 `sparrow`(RDF/SPARQL)가 깔려 있다. "
                      "pip uninstall sparrow 후 GitHub 판을 깔아라")
    probe = "GGGGSGGGGSGGGGS"
    try:
        p = sparrow.Protein(probe)
    except Exception as e:
        return None, f"sparrow.Protein() 실패 ({type(e).__name__}: {e})"

    def _try(fn):
        try:
            v = fn(p)
            v = float(v() if callable(v) else v)
            return v if np.isfinite(v) and v > 0 else None
        except Exception:
            return None

    cands = [
        ("predictor.end_to_end_distance", lambda q: q.predictor.end_to_end_distance),
        ("predictor.end_to_end",          lambda q: q.predictor.end_to_end),
        ("predictor.re",                  lambda q: q.predictor.re),
        ("predictor.radius_of_gyration",  lambda q: q.predictor.radius_of_gyration),
        ("predictor.rg",                  lambda q: q.predictor.rg),
    ]
    for name, f in cands:
        v = _try(f)
        if v is not None:
            is_rg = ("gyration" in name) or name.endswith(".rg")
            return (name, is_rg), f"ALBATROSS 사용 — {name}  (시험 {probe[:9]}… → {v:.1f} Å)"
    avail = [a for a in dir(getattr(p, "predictor", p)) if not a.startswith("_")]
    return None, f"sparrow 는 있는데 치수 예측 API 를 못 찾았다. 후보: {avail[:14]}"


def albatross_re(seqs, how):
    """서열 목록 → 말단간 거리 Re (Å). Rg 만 나오면 Re ≈ √6·Rg 로 바꾼다."""
    import sparrow
    name, is_rg = how
    path = name.split(".")
    out = []
    for s in seqs:
        if not s:
            out.append(np.nan); continue
        try:
            o = sparrow.Protein(s)
            for a in path:
                o = getattr(o, a)
            v = float(o() if callable(o) else o)
            out.append(v * np.sqrt(6.0) if is_rg else v)
        except Exception:
            out.append(np.nan)
    return np.array(out, float)


# ── [1] 데이터 ──────────────────────────────────────────────────────────────
def load_excel(path):
    X = pd.read_excel(path)
    X.columns = [str(c).strip() for c in X.columns]
    col = lambda p: next((c for c in X.columns if c.lower().replace(" ", "") == p), None)
    nm = col("이름") or col("name") or col("sample") or X.columns[0]
    blk = col("block") or col("블록")
    dc = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("domain")],
                key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))
    lc = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("linker")],
                key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))
    obs = [c for c in X.columns if any(k in c.lower() or k in c for k in OBS_KEYS)]
    hm = [c for c in obs if "hmw" in str(c).lower()]
    if not dc:
        raise SystemExit("★ 'Domain' 로 시작하는 열이 없다 — 엑셀 열 이름을 확인해라")
    if not hm:
        raise SystemExit(f"★ HMW 열이 없다. 찾은 실측열: {obs}")
    HC = hm[0]
    print(f"  파일     {os.path.basename(path)} ({len(X)}행)")
    print(f"  도메인열 {dc}   링커열 {lc}")
    print(f"  판정열   **{HC}**")

    rows = []
    for _, r in X.iterrows():
        # ★ 원래 열 위치를 유지한다. 빈 칸을 걸러 내며 압축하면 Domain_1 과 Domain_3 이
        #   이웃이 되어 엉뚱한 링커와 짝지어진다.
        D = {i: str(r[c]).strip().upper() for i, c in enumerate(dc)
             if isinstance(r[c], str) and len(str(r[c])) > 50}
        if not D:
            continue
        for i in sorted(D):
            if (i + 1) not in D:
                continue
            L = (str(r[lc[i]]).strip().upper()
                 if i < len(lc) and isinstance(r[lc[i]], str) else "")
            b = str(r[blk]) if blk and pd.notna(r.get(blk)) else "1"
            rows.append(dict(
                링커=str(r[nm]).strip().rsplit("_", 1)[-1],
                링커서열=L, n=len(L), D1=D[i], D2=D[i + 1],
                블록=re.sub(r"\.0$", "", b),
                HMW=pd.to_numeric(r[HC], errors="coerce")))
    d = pd.DataFrame(rows).dropna(subset=["HMW"])
    if not len(d):
        raise SystemExit("★ 구성체를 하나도 못 뽑았다")
    return d, HC


def attach_ab(d, keymap):
    if keymap and os.path.isfile(keymap):
        km = pd.read_csv(keymap, dtype={"블록": str})
        need = [c for c in ("D1", "D2", "블록") if c in km.columns]
        o = d.merge(km[need + ["항체"]].drop_duplicates(), on=need, how="left")
        miss = int(o.항체.isna().sum())
        print(f"  항체키   {os.path.basename(keymap)} (노트북과 같은 이름)"
              + (f"  ★ {miss}행은 못 찾아 직접 묶는다" if miss else ""))
        if miss:
            fb = "Ab" + (o.groupby(o.D1 + o.D2).ngroup() + 1).astype(str) + "_B" + o.블록
            o.loc[o.항체.isna(), "항체"] = fb[o.항체.isna()]
        return o
    print("  항체키   antibody_keys.csv 없음 — 도메인쌍으로 직접 묶는다")
    d = d.copy()
    d["항체"] = "Ab" + (d.groupby(d.D1 + d.D2).ngroup() + 1).astype(str) + "_B" + d.블록
    return d


def load_span(path, abs_):
    """anchor_span.csv → 항체별 r_ab. 없으면 NaN 으로 두고 [7] 에서 격자 탐색한다."""
    if path and os.path.isfile(path):
        s = pd.read_csv(path)
        if "항체" in s.columns and "r_ab" in s.columns:
            m = s.set_index("항체").r_ab.to_dict()
            sd = s.set_index("항체").get("r_ab_SD", pd.Series(dtype=float)).to_dict()
            got = sum(a in m for a in abs_)
            print(f"  앵커스팬 {os.path.basename(path)} — {got}/{len(abs_)}개 항체")
            big = [f"{a}({sd[a]:.1f})" for a in abs_ if sd.get(a, 0) and sd[a] > 2.0]
            if big:
                print(f"    ★ r_ab_SD > 2 Å 인 항체: {big} — 기준 구조가 이 값을 잘 못 정한다")
            return m, sd
    print(f"  앵커스팬 ★없음 ({path}) — r_ab 를 격자로 훑는다")
    return {}, {}


# ── [3] c_eff ───────────────────────────────────────────────────────────────
def c_eff_from_r2(r2, r_ab):
    """⟨r²⟩ 와 앵커 스팬에서 실효농도 (M). ⟨r²⟩ 출처가 Gaussian 이든 ALBATROSS 든 같다."""
    r2 = np.asarray(r2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (3.0 / (2 * np.pi * r2)) ** 1.5 * np.exp(-3.0 * r_ab ** 2 / (2 * r2)) * A3_TO_M
    return v


def ln_ceff(n, r_ab, b, re2=None):
    r2 = (n * RES_L * b) if re2 is None else re2
    r2 = np.asarray(r2, float)
    return -1.5 * np.log(r2) - 3.0 * r_ab ** 2 / (2 * r2)


# ── [5] 순위 검정 — 항체 안에서만 섞는 정확 순열 ────────────────────────────
def conc(x, h):
    """x(=ln c_eff) 가 클수록 h(=HMW) 가 작다는 쌍의 수. **동률은 양쪽 다 0.5점.**

    ★ x 동률을 행 순서로 흘리면 정보가 0인 축이 만점을 받는다. Gaussian 가지에서는
      ln c_eff 가 길이에만 의존하므로 **같은 길이 다른 조성** 링커가 정확히 이 경우다 —
      이 프로젝트가 가르려는 바로 그 쌍이다.
    """
    c = 0.0
    for i, j in itertools.combinations(range(len(x)), 2):
        if x[i] == x[j]:
            c += 0.5
            continue
        a, b_ = (i, j) if x[i] < x[j] else (j, i)     # a = c_eff 작은 쪽
        c += 1.0 if h[b_] < h[a] else (0.5 if h[b_] == h[a] else 0.0)
    return c


def exact_p(groups):
    """groups = [(x배열, h배열), ...]. 항체 안에서 h 를 모든 순서로 재배치한
    정확 귀무분포를 합성곱으로 만든다. MC 가 아니라 **정확 p** 다."""
    obs = sum(conc(x, list(h)) for x, h in groups)
    tot = Counter({0.0: 1.0})
    for x, h in groups:
        d = Counter(conc(x, list(p)) for p in permutations(h))
        s = sum(d.values()); nt = Counter()
        for k0, v0 in tot.items():
            for k1, v1 in d.items():
                nt[k0 + k1] += v0 * v1 / s
        tot = nt
    k = np.array(sorted(tot)); pr = np.array([tot[v] for v in sorted(tot)])
    mu = float((k * pr).sum()); sd = float(np.sqrt((pr * (k - mu) ** 2).sum()))
    return obs, float(pr[k >= obs].sum()), mu, sd, int(k.max())


# ── [6] 진폭 — 실측 폭을 내려면 f 가 몇 %여야 하나 ──────────────────────────
def f_required(spread, dln):
    """분배식의 최대 진폭은 c_bulk 를 최적(=c_eff 기하평균)에 놓았을 때
        ΔHMW_max = f · tanh(Δln c_eff / 4)
    이다. 그러니 실측 폭을 내는 데 필요한 f 는 spread / tanh(Δ/4).
    **f > 100 이면 그 b 에서 모델이 그 항체를 원리적으로 못 만든다.**"""
    t = np.tanh(np.asarray(dln, float) / 4.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, np.asarray(spread, float) / t, np.inf)


# ── 본체 ────────────────────────────────────────────────────────────────────
def main(argv):
    head("FvLinker 응집 모델 v1 · cis/trans 재짝짓기 경쟁", "=")

    head("[0] 환경", "─")
    how, msg = check_albatross()
    print(f"  {msg}")

    head("[1] 데이터", "─")
    xl = argv[0] if argv and os.path.isfile(argv[0]) else None
    if not xl:
        c = (sorted(glob.glob(f"{IN}/*test_result*.xls*"))
             or sorted(glob.glob(f"{IN}/*.xls*")) or sorted(glob.glob("*.xls*")))
        xl = c[-1] if c else None
    if not xl:
        raise SystemExit(f"★ 엑셀을 못 찾았다. 경로를 인자로 줘라 (찾아본 곳: {IN})")
    d, HC = load_excel(xl)
    d = attach_ab(d, f"{OUT}/antibody_keys.csv")
    span, span_sd = load_span(argv[1] if len(argv) > 1 else f"{OUT}/anchor_span.csv",
                              sorted(d.항체.unique()))

    multi = [a for a, g in d.groupby("항체") if len(g) >= 2]
    D = d[d.항체.isin(multi)].copy().sort_values(["항체", "n"]).reset_index(drop=True)
    print(f"\n  구성체 {len(d)} · 항체 {d.항체.nunique()}"
          f" · **링커 2종 이상 항체 {len(multi)}개 / 구성체 {len(D)}개**가 검정 대상")
    npair = sum(len(g) * (len(g) - 1) // 2 for _, g in D.groupby("항체"))
    print(f"  항체내 쌍 {npair}개 — 이게 실제 표본 수다 (구성체 26개가 아니다)")
    if not len(D):
        raise SystemExit("★ 링커가 2종 이상인 항체가 없다 — 항체내 비교가 불가능하다")

    head("[2] 사슬 치수 ⟨r²⟩", "─")
    if how:
        re_ = albatross_re(list(D.링커서열), how)
        ok = int(np.isfinite(re_).sum())
        print(f"  ALBATROSS 로 {ok}/{len(D)}개 링커의 Re 를 얻었다")
        if ok < len(D):
            print(f"  ★ {len(D)-ok}개는 실패 — 그 칸만 Gaussian 으로 메운다")
        D["Re_A"] = re_
        # 같은 길이에서 조성이 치수를 얼마나 바꾸나 — **교란을 깨는 칸**
        for n0, g in D.dropna(subset=["Re_A"]).groupby("n"):
            if g.링커.nunique() >= 2:
                print(f"    길이 {n0}: " + " · ".join(
                    f"{r.링커} {r.Re_A:.1f}Å" for _, r in g.iterrows()))
    else:
        D["Re_A"] = np.nan
        print("  Gaussian 사슬만 쓴다 — ⟨r²⟩ = n·l·b. **조성을 못 본다.**")
        print("  (Colab 에서 `pip install git+https://github.com/idptools/sparrow.git`")
        print("   하면 서열별 치수가 들어온다. 없어도 아래는 전부 돈다.)")

    # r_ab 가 없는 항체는 격자로 훑는다
    R_GRID = np.arange(24.0, 44.1, 0.5)
    have_r = {a: span[a] for a in multi if a in span and np.isfinite(span.get(a, np.nan))}
    print(f"\n  r_ab 실측 {len(have_r)}/{len(multi)}개 항체")
    for a in multi:
        print(f"    {a:10s} r_ab = " + (f"{have_r[a]:.1f} Å" if a in have_r else "★없음 → 격자"))

    head("[7] 자유변수는 b 하나 — 27쌍으로 잡는다", "=")
    print("  b 를 훑으며 (a) 순위 일치도와 (b) 필요한 f 를 동시에 본다.")
    print("  r_ab 가 없는 항체는 그 b 에서 일치도를 최대로 만드는 r_ab 를 고른다")
    print("  (그 항체에 대해서는 사후 적합이므로 p 를 그만큼 낮춰 읽어야 한다).\n")

    rows = []
    for b in B_GRID:
        groups, dln, sprd, rused = [], [], [], {}
        for ab, g in D.groupby("항체"):
            n = g.n.values.astype(float)
            r2 = np.where(np.isfinite(g.Re_A.values),
                          g.Re_A.values ** 2, n * RES_L * b)
            h = list(g.HMW.values)
            if ab in have_r:
                r = have_r[ab]
            else:                       # 일치도를 최대로 만드는 r_ab
                best, r = -1, R_GRID[0]
                for rr in R_GRID:
                    c = conc(ln_ceff(n, rr, b, r2), h)
                    if c > best:
                        best, r = c, rr
            rused[ab] = r
            x = ln_ceff(n, r, b, r2)
            groups.append((x, h))
            dln.append(float(np.max(x) - np.min(x)))
            sprd.append(float(np.max(h) - np.min(h)))
        obs, p, mu, sd, kmax = exact_p(groups)
        freq = f_required(sprd, dln)
        rows.append(dict(b=b, 일치=obs, 최대=kmax, p=p, 귀무=mu,
                         Δln중앙=float(np.median(dln)),
                         f필요중앙=float(np.median(freq)),
                         f필요최대=float(np.max(freq)),
                         불가능항체=int((freq > 100).sum()),
                         **{f"r_{a}": rused[a] for a in multi}))
    PR = pd.DataFrame(rows)

    show = ["b", "일치", "p", "Δln중앙", "f필요중앙", "f필요최대", "불가능항체"]
    print(PR[show].round(3).to_string(index=False))

    ok = PR[PR.불가능항체 == 0]
    best = PR.loc[PR.p.idxmin()]
    head("[9] 판정", "=")
    print(f"  최대 일치도: b = {best.b:.1f} Å 에서 {best.일치:.1f}/{int(best.최대)} "
          f"(귀무 {best.귀무:.1f}) · **정확 순열 p = {best.p:.4f}**")
    print(f"    그 b 에서 실측 폭을 내는 데 필요한 f: 중앙 {best.f필요중앙:.0f}% · "
          f"최대 {best.f필요최대:.0f}%")
    print()
    if not len(ok):
        print("  ★★ 어느 b 에서도 **모든 항체를 만들 수 있는 조합이 없다** (f > 100% 필요).")
        print("     Gaussian 사슬의 지렛대가 실측 진폭에 못 미친다는 뜻이다.")
        print("     → 링커 **조성**이 사슬 치수를 바꾸는 항이 있어야 한다 (ALBATROSS),")
        print("       또는 길이가 주 기전이 아니다 → pH 3 MD + 노출 패치로 간다.")
    else:
        lo, hi = ok.b.min(), ok.b.max()
        pm = float(ok.p.min())
        print(f"  ✔ f ≤ 100% 로 모든 항체를 만들 수 있는 b 구간: **{lo:.1f} ~ {hi:.1f} Å** "
              f"(그 안 최소 p = {pm:.4f})")
        print("    폴리펩타이드 Kuhn 길이의 통상 범위(8~20 Å)와 겹치면 물리적으로 말이 된다.")
    if float(best.p) < 0.05:
        print("\n  ✔ 순위 예측이 유의하다. 다만 b 를 훑어 최소 p 를 고른 것이므로")
        print("    다중비교가 있다 — 사전등록된 값이 아니다.")
    else:
        print(f"\n  ✗ 순위 예측이 유의하지 않다 (최선의 b 에서도 p = {best.p:.3f}).")
        print("    방향은 맞는데 표본이 못 따라오거나, 한 항체가 반대로 간다.")

    head("[4] 반전 — 길이 추세가 반대인 항체", "─")
    print("  모델은 n > n* 이면 부호가 뒤집힌다고 말한다. n* = r_ab²/(l·b).")
    print("  추세가 반대인 항체는 **n* 가 그 항체 링커 길이 창 아래**여야 한다.\n")
    b0 = float(best.b)
    for ab, g in D.groupby("항체"):
        g = g.sort_values("n")
        rho = float(pd.Series(g.n.values).corr(pd.Series(g.HMW.values), method="spearman"))
        r = have_r.get(ab, float(best[f"r_{ab}"]))
        ns = r * r / (RES_L * b0)
        need = np.sqrt(g.n.min() * RES_L * b0)
        tag = "★ 반대 방향" if rho > 0 else ""
        line = (f"  {ab:10s} 길이-HMW rho {rho:+.2f}  r_ab {r:.1f} Å  "
                f"n*(b={b0:.1f}) = {ns:.0f}  링커 {g.n.min()}~{g.n.max()}  {tag}")
        print(line)
        if rho > 0:
            src = "실측" if ab in have_r else "격자에서 고른 값이라 검정이 아니다"
            print(f"      → 반전하려면 r_ab ≤ {need:.1f} Å 여야 한다. "
                  f"{r:.1f} Å ({src}) → "
                  + ("**조건 만족**" if r <= need else "**조건 불만족 — 이 항체는 길이로 설명 안 된다**"))

    head("[6] 진폭 — 항체별", "─")
    print(f"  b = {b0:.1f} Å 기준. f필요 > 100% 면 그 항체는 이 기전으로 못 만든다.\n")
    amp = []
    for ab, g in D.groupby("항체"):
        n = g.n.values.astype(float)
        r2 = np.where(np.isfinite(g.Re_A.values), g.Re_A.values ** 2, n * RES_L * b0)
        r = have_r.get(ab, float(best[f"r_{ab}"]))
        x = ln_ceff(n, r, b0, r2)
        dl = float(x.max() - x.min()); sp = float(g.HMW.max() - g.HMW.min())
        fr = float(f_required(sp, dl))
        amp.append(dict(항체=ab, n링커=len(g), 실측폭=round(sp, 1),
                        Δln_c_eff=round(dl, 3), f필요=round(fr, 0),
                        가능=("예" if fr <= 100 else "★아니오")))
    AMP = pd.DataFrame(amp)
    print(AMP.to_string(index=False))

    head("[8] 예측 곡선 — 다음 라운드 설계는 여기서 나온다", "─")
    print(f"  b = {b0:.1f} Å · f = 실측에서 필요한 값 · c_bulk = c_eff 기하평균.")
    print("  **최소를 주는 길이(n*)가 어디인지**가 다음에 만들 링커를 정한다.\n")
    curves = []
    for ab, g in D.groupby("항체"):
        r = have_r.get(ab, float(best[f"r_{ab}"]))
        ns = r * r / (RES_L * b0)
        fr = float(AMP[AMP.항체 == ab].f필요.iloc[0])
        nn = np.arange(10, 46)
        x = ln_ceff(nn.astype(float), r, b0)
        x0 = np.median(ln_ceff(g.n.values.astype(float), r, b0))
        hmw = float(g.HMW.min()) + min(fr, 100.0) * (1 / (1 + np.exp(x - x0))
                                                     - 1 / (1 + np.exp(x.max() - x0)))
        for k, v in zip(nn, hmw):
            curves.append(dict(항체=ab, n=int(k), 예측HMW=round(float(v), 2)))
        print(f"  {ab:10s} r_ab {r:.1f} Å → **n* = {ns:.0f} 잔기에서 HMW 최소**  "
              f"(실측 창 {g.n.min()}~{g.n.max()})")
        if ns > g.n.max():
            print(f"      아직 꼭짓점을 안 지났다 → **{ns:.0f} 잔기 링커를 만들면 더 내려가야 한다**")
        elif ns < g.n.min():
            print(f"      이미 지났다 → **더 짧은 링커({ns:.0f})가 더 낮아야 한다**")
        else:
            print(f"      창 안에 꼭짓점이 있다 → 양쪽으로 갈수록 올라가야 한다")

    # ── 저장 ────────────────────────────────────────────────────────────────
    od = OUT if os.path.isdir(OUT) else "."
    for nm2, df2 in (("fvlinker_profile.csv", PR), ("fvlinker_amplitude.csv", AMP),
                     ("fvlinker_curves.csv", pd.DataFrame(curves)),
                     ("fvlinker_data.csv", D.drop(columns=["D1", "D2"]))):
        try:
            df2.to_csv(f"{od}/{nm2}", index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"  ★ {nm2} 저장 실패: {e}")
    print(f"\n  저장: {od}/fvlinker_{{profile,amplitude,curves,data}}.csv")
    head("끝", "=")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
