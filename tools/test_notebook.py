"""FvFlow_v12.ipynb 의 **분석 셀을 실제로 실행**해 본다.

Drive·GPU·BioEmu 가 필요한 셀(0~5절 실행부)은 건너뛰고, 그 셀들이 만들었을
표(CONS · E · G)를 가짜로 만들어 넣은 뒤 6~9절을 통째로 돌린다.

이게 잡는 것: 이름 오타, 열 이름 불일치, groupby 키 실수, NaN 전파, 0 나누기,
순열 검정의 영점 붕괴 — 노트북을 Colab 에 올리고 나서야 알게 될 것들.

돌리는 법:  python3 tools/test_notebook.py
"""
import ast
import json
import os
import sys
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "FvFlow_v12.ipynb")

# 실행할 셀 — 6절 특징표부터 9절 그림까지 (Drive·GPU 를 안 쓴다)
ANALYSIS_CELLS = {
    17: "6절 특징표",
    18: "6절-B 두 원장",
    20: "6절-C 채널 판정 (2원 분산분석)",
    21: "6절-D 국소 채널 시험 (EAAAK vs G4S)",
    22: "6절-E 표본 독립성 (ICC)",
    24: "7절 ML 핵심",
    25: "7절-B 판정",
    27: "8절 대리모형",
    29: "9절 그림",
}


# ─────────────────────────────────────────────────────────────────────────────
# 가짜 입력 — 실제 데이터와 **같은 모양**으로 만든다
# ─────────────────────────────────────────────────────────────────────────────
AB_SD = dict(HL=3.92, HC1=2.18, HC2=3.07, LC1=2.49, LC2=2.25, dc=0.268)
AB6 = ["HL", "HC1", "HC2", "LC1", "LC2", "dc"]
DC0, DC_SD, PAIR_NSD = 16.206, 0.268, 6.0
MOTIF = {"G4S": "GGGGS", "GS": "GGSGG", "EAAAK": "EAAAK",
         "PAS": "APAPA", "ED": "GEGGS", "KR": "GKGGS"}
PANEL_LENGTHS = (10, 15, 20, 25)
HC = "HMW(%)"


def fake_cons(n_block=5, plant_composition=True):
    """실측 블록 5개 × 링커 4종 + 합성 패널 24칸. 실제 CONS 와 같은 열."""
    rng = np.random.default_rng(0)
    lens = [15, 18, 21, 25]
    rows = []
    for b in range(1, n_block + 1):
        base = rng.uniform(2, 20)                       # 블록마다 HMW 바닥이 다르다
        for i, L in enumerate(lens):
            d1 = "E" * 120
            d2 = "D" * 118
            s = "GGGGS" * (L // 5)
            rows.append(dict(
                블록=str(b), 항체=f"실측{b}_B{b}_HL", 포맷="순수", 배향="HL",
                링커=f"Linker{i+1}", 링커서열=s, D1=d1, D2=d2,
                seq=d1 + s + d2, b1=120, b2=120 + L, 길이=L, 합성=False,
                Fv키=f"Fv{b}", 조성=np.nan, 설계길이=np.nan,
                **{HC: base + 0.55 * (L - 15) + rng.normal(0, 1.0)}))
    # 합성 요인패널 — 블록 1 의 숙주에 붙는다
    h = rows[0]
    for nm, m in MOTIF.items():
        for L in PANEL_LENGTHS:
            s = (m * (L // len(m) + 1))[:L]
            rows.append(dict(h, 링커=f"P_{nm}{L}", 링커서열=s, b2=120 + L, 길이=L,
                             seq=h["D1"] + s + h["D2"], 합성=True,
                             조성=nm, 설계길이=L, **{HC: np.nan}))
    return pd.DataFrame(rows)


def fake_frames(CONS, plant_composition=True, seed=0):
    """구성체마다 150 프레임. **길이 효과는 꼭 심고**, 조성 효과는 스위치로."""
    rng = np.random.default_rng(seed)
    rows = []
    comp_shift = {"G4S": 0.0, "GS": 0.05, "EAAAK": 0.9, "PAS": 0.5,
                  "ED": 0.2, "KR": -0.15}
    for _, r in CONS.iterrows():
        # dc 의 퍼짐이 길이에 따라 커진다 → 열림분율이 길이를 따라간다
        sd = 0.35 + 0.055 * (r.길이 - 10)
        if plant_composition and isinstance(r.조성, str):
            sd += comp_shift.get(r.조성, 0.0)           # 조성축 신호
        for i in range(150):
            d = {f"Δ{k}": rng.normal(0, AB_SD[k] * 0.9) for k in AB6}
            d["Δdc"] = rng.normal(0.1, sd)
            d.update({k: 0.0 for k in AB6})
            d["dc"] = DC0 + d["Δdc"]                 # 원시 6값도 저장한다 (v12 형식)
            d["OCD6"] = sum(abs(d[f"Δ{k}"]) / AB_SD[k] for k in AB6)
            d["OCD5"] = sum(abs(d[f"Δ{k}"]) / AB_SD[k] for k in AB6 if k != "dc")
            rows.append(dict(블록=r.블록, 항체=r.항체, 배향=r.배향, 링커=r.링커,
                             길이=int(r.길이), 합성=bool(r.합성), 조성=r.조성,
                             설계길이=r.설계길이, 생성기="BioEmu",
                             태그=f"{r.링커}__BioEmu_s{i}",
                             **{HC: r[HC]}, **d))
    return pd.DataFrame(rows)


def fake_geom(E, plant_composition=True, seed=0):
    """좌표 기하 표. 조성 효과는 스위치로 — 끄면 **접촉도 나선도도 조성과 무관**해야 한다."""
    rng = np.random.default_rng(seed)
    contact = {"G4S": 12.0, "GS": 12.5, "EAAAK": 6.0, "PAS": 8.0,
               "ED": 11.0, "KR": 13.5}
    helix   = {"G4S": .10, "GS": .12, "EAAAK": .72, "PAS": .15,
               "ED": .11, "KR": .13}          # EAAAK 만 나선 — 6절-D 의 양성대조
    rows = []
    for _, r in E.drop_duplicates(["항체", "태그"]).iterrows():
        on = plant_composition and isinstance(r.조성, str)
        base = contact.get(r.조성, 11.5) if on else 11.5
        hx   = helix.get(r.조성, .11) if on else .11
        d4   = 6.2 if hx > .4 else 11.0
        rows.append(dict(항체=r.항체, 링커=r.링커, 태그=r.태그,
                         링커접촉_잔기당=base + rng.normal(0, 1.5),
                         링커밀착율=np.clip(base / 20 + rng.normal(0, .05), 0, 1),
                         링커최근접=6 + rng.normal(0, 1),
                         링커Rg_잔기당=0.3 + rng.normal(0, .03),
                         링커신장도=np.clip(.6 + rng.normal(0, .08), 0, 1.2),
                         링커나선도=np.clip(hx + rng.normal(0, .08), 0, 1),
                         링커i_i4=d4 + rng.normal(0, 1.0),
                         도메인Rg=18 + rng.normal(0, .4),
                         말단간=45 + rng.normal(0, 3)))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def load_cells():
    nb = json.load(open(NB, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]], \
           [c["cell_type"] for c in nb["cells"]]


def strip_magic(src):
    return "\n".join(("#MAGIC " + l) if l.lstrip().startswith("!") else l
                     for l in src.split("\n"))


def base_globals(CONS, E, G, out_dir):
    """0~5절이 만들었을 이름들을 전부 준비한다. 여기 빠진 이름이 있으면
    그것 자체가 노트북의 **셀 간 숨은 의존**이라는 뜻이다 — 잡아야 할 버그다."""
    import glob as _glob
    import time as _time
    import subprocess as _sp
    import scipy.stats as st_
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def setup_font():
        plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 70})
        return plt

    def display(x):                                   # Colab 의 display 대역
        if isinstance(x, pd.DataFrame):
            print(x.to_string(index=False)[:2400])
        else:
            print(x)

    T0 = _time.time()
    g = dict(
        np=np, pd=pd, os=os, glob=_glob, time=_time, json=json, sys=sys,
        subprocess=_sp, st_=st_, plt=plt, display=display, setup_font=setup_font,
        shutil=__import__("shutil"), re=__import__("re"),
        # 1절
        OUT=out_dir, DRIVE=out_dir, IN=out_dir, CONS=CONS, HC=HC, BLOCKS="all",
        PAIR_DC=PAIR_NSD*DC_SD, PAIR_NSD=PAIR_NSD, DC0=DC0, DC_SD=DC_SD,
        N_PERM=300, ALPHA=0.05, GEN="BioEmu", HIGHER_IS_WORSE=True,
        OBS=[HC], OBS_KEYS=("hmw",),
        RUN_ABB2=False, RUN_BIOEMU=False, RUN_PANEL=True, RUN_ML=True,
        T0=T0, elapsed=lambda: "[0분]", budget=lambda *a, **k: True,
        left=lambda: 1e9,
        FL=lambda r: f"{out_dir}/{r.항체}/flow", REFD=lambda ab: f"{out_dir}/{ab}",
        safe=lambda x: str(x),
        # 2절
        AB_SD=AB_SD, AB6=AB6, ca_xyz=lambda p: np.zeros((240, 3)),
        kabsch=lambda P, Q: np.eye(3),
        # 4-B절
        MOTIF=MOTIF, PANEL_LENGTHS=PANEL_LENGTHS,
        # 5절
        E=E, G=G, CSV_GEOM=f"{out_dir}/geom.csv", REUSE_CSV=False,
        CONTACT_CUT=10.0,
        GEOM_FEATS=["링커접촉_잔기당", "링커밀착율", "링커최근접",
                    "링커Rg_잔기당", "링커신장도", "링커나선도", "링커i_i4",
                    "도메인Rg", "말단간"],
        HELIX_LO=5.0, HELIX_HI=7.5,
        _pairs_needed=lambda r: [],
    )
    g["__builtins__"] = __builtins__
    return g


def run(plant_composition=True, label=""):
    srcs, kinds = load_cells()
    out = os.path.join("/tmp", "fvflow_nbtest")
    os.makedirs(out, exist_ok=True)
    CONS = fake_cons()
    E = fake_frames(CONS, plant_composition)
    G = fake_geom(E, plant_composition)
    g = base_globals(CONS, E, G, out)

    print("=" * 78)
    print(f"노트북 분석 셀 실행  {label}")
    print("=" * 78)
    for i, name in ANALYSIS_CELLS.items():
        assert kinds[i] == "code", f"셀 {i} 가 코드가 아니다 ({kinds[i]}) — 셀 번호를 고쳐라"
        print(f"\n{'─'*78}\n▶ 셀 {i} · {name}\n{'─'*78}")
        exec(compile(strip_magic(srcs[i]), f"<cell {i}>", "exec"), g)
    return g


def main():
    fails = []

    # ── 1. 조성 효과를 심은 세계 ───────────────────────────────────────────
    g = run(plant_composition=True, label="(조성 효과를 심은 가짜 데이터)")

    # 특징표가 섰나
    F = g["FEAT"]
    assert len(F) == 44, f"특징표 행 수가 44가 아니다: {len(F)}"
    assert F.합성.sum() == 24, f"합성 구성체가 24가 아니다: {F.합성.sum()}"
    assert g["GROUP"] == "항체", f"독립 단위가 항체가 아니다: {g['GROUP']}"
    for c in [g["PRIMARY"]] + g["EXPLORATORY"]:
        assert c in F.columns, f"사전 등록 특징 {c} 가 특징표에 없다"
        assert F[c].notna().any(), f"{c} 가 전부 NaN 이다"
    print(f"\n  OK  특징표 {len(F)}행 · 사전 등록 특징 4개 전부 존재")

    # 채널 판정이 조성을 잡았나 (심었으니 잡아야 한다)
    assert "VERD" in g and g["VERD"], "채널 판정이 아무 특징도 못 냈다"
    V = pd.DataFrame(g["VERD"].values())
    hit = V[(V.조성p.notna()) & (V.조성p < 0.05) & (V.조성eta2 > 0.15)]
    if not len(hit):
        fails.append(f"심은 조성 효과를 채널 판정이 못 잡았다:\n{V.to_string(index=False)}")
    else:
        print(f"  OK  채널 판정 — 조성 효과를 잡았다: "
              f"{list(hit.특징)} (최대 η²조성 {hit.조성eta2.max():.2f})")

    # ML 이 돌았나
    assert "ML_RESULT" in g, "7절이 ML_RESULT 를 안 만들었다"
    R = g["ML_RESULT"]
    print(f"  OK  7절 — 확증 일치도 {R['확증']['일치도']:.2f} "
          f"p={R['확증']['순열p']:.4f} · MDE={R['MDE']:.2f}")
    assert 0 <= R["확증"]["일치도"] <= 1
    assert np.isfinite(R["MDE"])
    fpr = R["검정력"].검출률.iloc[0]
    if fpr > 0.15:
        fails.append(f"검정력 곡선의 β=0 거짓양성률이 {fpr:.3f} — 명목 0.05 를 크게 넘는다")
    else:
        print(f"  OK  검정력 영점 — 거짓양성률 {fpr:.3f}")

    # 길이에 심었으니 확증 검정이 잡아야 한다 (열림분율이 길이를 따라간다)
    if R["확증"]["순열p"] >= 0.05:
        fails.append(f"길이에 크게 심었는데 확증 검정이 못 잡았다: p={R['확증']['순열p']}")
    else:
        print("  OK  확증 검정 — 심은 신호를 잡았다")

    # 증분: 열림분율이 길이의 변장이므로 증분은 유의하면 **안 된다**
    I = R["증분"]
    if (I.순열p < 0.05).any():
        print(f"  ※ 증분이 유의하게 나왔다 (조성 신호를 접촉 특징이 날랐을 수 있다):\n"
              f"{I.to_string(index=False)}")
    else:
        print("  OK  증분 검정 — 길이 위의 증분 없음 (열림분율은 길이의 변장이므로 맞다)")

    # 6절-D 국소 채널 시험 — EAAAK 나선을 심었으니 잡아야 한다
    HX = g.get("HX")
    if HX is None or not len(HX):
        fails.append("6절-D 국소 채널 시험이 표를 못 냈다")
    else:
        pos = int((HX.차 > 0).sum())
        if pos != len(HX):
            fails.append(f"심은 EAAAK 나선을 국소 시험이 못 잡았다:\n{HX.to_string(index=False)}")
        else:
            print(f"  OK  6절-D 국소 채널 — EAAAK 가 {pos}/{len(HX)} 길이에서 더 나선 "
                  f"(효과크기 중앙 {HX.효과크기.median():.1f})")

    # 대리모형
    assert "SEQ_FEATS" in g, "8절이 안 돌았다"
    print("  OK  8절 대리모형 실행 완료")

    # 파일이 써졌나
    for f in ("features.csv",):
        p = os.path.join("/tmp/fvflow_nbtest", f)
        assert os.path.isfile(p), f"{f} 가 안 써졌다"
    print("  OK  features.csv 저장됨")

    # ── 2. 조성 효과가 **없는** 세계 — 거짓양성이 안 나야 한다 ─────────────
    print("\n\n")
    g2 = run(plant_composition=False, label="(조성 효과 없음 — 거짓양성 시험)")
    V2 = pd.DataFrame(g2["VERD"].values())
    bad = V2[(V2.조성p.notna()) & (V2.조성p < 0.01) & (V2.조성eta2 > 0.4)]
    if len(bad):
        fails.append(f"조성 효과를 안 심었는데 채널 판정이 크게 잡았다:\n"
                     f"{V2.to_string(index=False)}")
    else:
        print(f"\n  OK  조성 없는 세계 — 채널 판정이 조성을 크게 잡지 않았다 "
              f"(최대 η²조성 {V2.조성eta2.max():.2f})")
    HX2 = g2.get("HX")
    if HX2 is not None and len(HX2) and int((HX2.효과크기 > 0.3).sum()) >= 3:
        fails.append(f"나선을 안 심었는데 국소 시험이 잡았다:\n{HX2.to_string(index=False)}")
    elif HX2 is not None and len(HX2):
        print(f"  OK  조성 없는 세계 — 국소 시험도 조용하다 "
              f"(효과크기 0.3 초과 {int((HX2.효과크기 > 0.3).sum())}/{len(HX2)})")

    print("\n" + "=" * 78)
    if fails:
        for f in fails:
            print("★ 실패:", f)
        print(f"{len(fails)}건 실패")
        return 1
    print("노트북 분석 경로 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
