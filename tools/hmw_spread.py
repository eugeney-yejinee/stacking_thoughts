#!/usr/bin/env python3
"""항체 **안에서** HMW 가 몇 %p 벌어지는가 — 이 숫자 하나만 본다.

왜 이것만 따로 뽑나:
  전체 HMW 범위 4~43% 는 대부분 **항체 간** 차이다. 모델이 설명해야 하는 것은
  *같은 항체에서 링커만 바꿨을 때* 벌어지는 폭이다. 그 폭이

    ~5 %p   → Gaussian 사슬 c_eff 가 낼 수 있는 진폭(최대 4.9 %p)과 같은 자리다.
              방향성(순위) 주장까지만 가능하고 기울기는 인용하면 안 된다.
    15~20 %p → **c_eff 로는 원리적으로 못 만든다.** 링커 길이가 기전이 아니라는 뜻이고,
              진폭이 큰 다른 축(pH 3 에서 드러나는 표면)을 봐야 한다.

노트북을 안 돌려도 된다. 엑셀만 읽는다. pandas + openpyxl 외에 의존성 없다.

사용:
    python3 tools/hmw_spread.py                      # Drive 기본 경로에서 찾는다
    python3 tools/hmw_spread.py <엑셀경로> [키맵경로]
"""
from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

DRIVE = "/content/drive/MyDrive/FvTwist"
IN = f"{DRIVE}/input"
OUT = f"{DRIVE}/fvflow"          # ★ build_v12.py 와 같은 경로. 이름이 FvLinker 여도 안 바꾼다.

OBS_KEYS = ("hmw", "monomer", "단량체", "purity", "순도", "수율", "생산",
            "titer", "sec", "quantific", "supernatant")


def find_excel(argv: list[str]) -> str | None:
    if argv:
        return argv[0] if os.path.isfile(argv[0]) else None
    for pat in (f"{IN}/*test_result*.xls*", f"{IN}/*.xls*", "input/*test_result*.xls*"):
        c = sorted(glob.glob(pat))
        if c:
            return c[-1]
    return None


def load_pairs(path: str) -> pd.DataFrame:
    """엑셀 → (D1, D2, 블록, 링커라벨, 링커길이, 실측열들). build_v12.load() 와 같은 규칙.

    ANARCI 는 쓰지 않는다 — 항체를 **묶기만** 하면 되고, 묶는 키는 도메인 쌍이라
    H/L 판별이 필요 없다. 그래서 배향으로 걸러내지 않고 이웃 쌍을 전부 남긴다.
    """
    X = pd.read_excel(path)
    X.columns = [str(c).strip() for c in X.columns]
    col = lambda p: next((c for c in X.columns if c.lower().replace(" ", "") == p), None)

    nm = col("이름") or col("name") or col("sample") or X.columns[0]
    blk = col("block") or col("블록")
    obs = [c for c in X.columns
           if any(k in c.lower() or k in c for k in OBS_KEYS)]
    dcol = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("domain")],
                  key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))
    lcol = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("linker")],
                  key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))

    print(f"  파일      {os.path.basename(path)}  ({len(X)}행)")
    print(f"  도메인열  {dcol}")
    print(f"  링커열    {lcol}")
    print(f"  실측열    {obs}")
    if not dcol:
        print("  ★ 'Domain' 로 시작하는 열이 없다 — 엑셀 열 이름을 확인해라")
        return pd.DataFrame()

    rows = []
    for _, r in X.iterrows():
        # ★ 원래 열 위치를 유지한다. 빈 칸을 걸러 내며 압축하면 Domain_1 과 Domain_3 이
        #   이웃이 되어 엉뚱한 링커와 짝지어진다 (v11 이 그랬다).
        D = {i: str(r[c]).strip().upper() for i, c in enumerate(dcol)
             if isinstance(r[c], str) and len(str(r[c])) > 50}
        L = {i: (str(r[c]).strip().upper() if isinstance(r[c], str) else "")
             for i, c in enumerate(lcol)}
        base = str(r[nm]).strip()
        if not D:
            continue
        for i in sorted(D):
            if (i + 1) not in D:          # **이웃한** 도메인 쌍만
                continue
            tag = base if len(D) == 2 else f"{base}#{i + 1}"
            rows.append(dict(
                행=tag,
                링커=str(tag).rsplit("_", 1)[-1],
                링커서열=L.get(i, ""),
                길이=len(L.get(i, "")),
                D1=D[i], D2=D[i + 1],
                블록=str(r[blk]) if blk and pd.notna(r.get(blk)) else "1",
                **{o: r[o] for o in obs if pd.notna(r.get(o))},
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df["블록"] = df.블록.map(
            lambda v: str(int(float(v))) if re.fullmatch(r"\d+(\.0)?", str(v)) else str(v))
    return df


def attach_antibody(df: pd.DataFrame, keymap: str | None) -> pd.DataFrame:
    """항체 이름을 붙인다. Drive 의 antibody_keys.csv 가 있으면 **그것을 정답으로 삼는다**
    (노트북과 같은 이름이 나와야 나중에 대조가 된다). 없으면 도메인쌍으로 직접 묶는다."""
    if keymap and os.path.isfile(keymap):
        km = pd.read_csv(keymap, dtype={"블록": str})
        need = [c for c in ("D1", "D2", "블록") if c in km.columns]
        out = df.merge(km[need + ["항체"]].drop_duplicates(), on=need, how="left")
        miss = int(out.항체.isna().sum())
        print(f"  항체키     {os.path.basename(keymap)} 에서 읽었다"
              + (f"  ★ {miss}행은 못 찾아서 직접 묶는다" if miss else ""))
        if miss:
            fb = "직접" + (out.groupby(out.D1 + out.D2).ngroup() + 1).astype(str) \
                 + "_B" + out.블록
            out.loc[out.항체.isna(), "항체"] = fb[out.항체.isna()]
        return out
    print("  항체키     antibody_keys.csv 가 없다 — 도메인쌍으로 직접 묶는다")
    df = df.copy()
    df["항체"] = ("항체" + (df.groupby(df.D1 + df.D2).ngroup() + 1).astype(str)
                  + "_B" + df.블록)
    return df


def main(argv: list[str]) -> int:
    path = find_excel(argv)
    if not path:
        print("★ 엑셀을 못 찾았다. 경로를 인자로 줘라:")
        print("   python3 tools/hmw_spread.py /content/drive/MyDrive/FvTwist/input/....xlsx")
        return 1

    print("=" * 78)
    df = load_pairs(path)
    if not len(df):
        print("★ 구성체를 하나도 못 뽑았다.")
        return 1

    keymap = argv[1] if len(argv) > 1 else f"{OUT}/antibody_keys.csv"
    df = attach_antibody(df, keymap)

    obs = [c for c in df.columns if any(k in str(c).lower() or k in str(c) for k in OBS_KEYS)]
    hm = [c for c in obs if "hmw" in str(c).lower()]
    if not hm:
        print(f"\n★ HMW 열이 없다. 찾은 실측열: {obs}")
        return 1
    HC = hm[0]
    df[HC] = pd.to_numeric(df[HC], errors="coerce")
    n_bad = int(df[HC].isna().sum())

    print(f"\n  구성체 {len(df)} · 항체 {df.항체.nunique()} · 판정열 **{HC}**"
          + (f"  (수치가 아닌 칸 {n_bad}개는 뺀다)" if n_bad else ""))

    d = df.dropna(subset=[HC])
    g = (d.groupby("항체")[HC].agg(n="count", 최소="min", 최대="max").reset_index())
    g["폭"] = (g.최대 - g.최소).round(2)
    g["최소"] = g.최소.round(2)
    g["최대"] = g.최대.round(2)
    g = g.sort_values("폭", ascending=False)

    print("\n" + "=" * 78)
    print("★ 항체 **안에서** HMW 가 벌어지는 폭  (링커 2종 이상인 항체만 의미가 있다)")
    print("=" * 78)
    print(g.to_string(index=False))

    multi = g[g.n >= 2]
    if not len(multi):
        print("\n★ 링커가 2종 이상인 항체가 없다 — 항체내 비교가 원리적으로 불가능하다.")
        return 1

    print(f"\n  링커 2종 이상인 항체 {len(multi)}개 · 폭 중앙값 "
          f"**{float(multi.폭.median()):.1f} %p** · 최대 {float(multi.폭.max()):.1f} %p")
    print(f"  (참고) 전체 범위 {float(d[HC].min()):.1f} ~ {float(d[HC].max()):.1f} %p "
          f"— 이 중 대부분이 항체 간 차이다")

    med = float(multi.폭.median())
    print("\n" + "-" * 78)
    if med < 7:
        print("  → 판정: Gaussian 사슬 c_eff 가 낼 수 있는 진폭(최대 4.9 %p)과 같은 자리다.")
        print("    순위/방향 주장까지만 가능하다. 기울기는 인용하면 안 된다.")
    elif med < 12:
        print("  → 판정: 경계다. c_eff 만으로는 빠듯하고, 링커 조성이 Kuhn 길이를")
        print("    바꾸는 항이 있어야 설명된다.")
    else:
        print("  → 판정: **c_eff 로는 원리적으로 못 만든다.** 링커 길이가 기전이 아니다.")
        print("    진폭이 큰 다른 축을 봐야 한다 — pH 3 에서 드러나는 표면 쪽.")
    print("-" * 78)

    print("\n항체별 링커 내역 (길이 = 링커 잔기 수):")
    cols = [c for c in ("항체", "링커", "길이", HC) if c in d.columns]
    print(d[cols].sort_values(["항체", "길이"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
