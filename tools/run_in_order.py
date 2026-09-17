"""노트북을 **셀 0번부터 끝까지 순서대로 실제로 실행**한다.

test_notebook.py 는 분석 셀만 골라 돌리면서 앞 셀이 만들었을 이름을 가짜로 넣어 준다.
그래서 "위에서부터 차례로 돌리면 죽는" 문제를 원리적으로 못 잡는다. 이 파일이 그걸 잡는다.

무엇을 막고 무엇을 그대로 두나
  막는 것   Drive 마운트 · pip/apt · GPU · 하위 프로세스(ABangle·BioEmu·ColabFold)
  그대로    **노트북의 모든 파이썬 로직** — 셀 간 이름 전달, 스위치 분기, 이어달리기 판정

두 번에 나눠 돈다
  1차: 셀 0~4 만 돌려 CONS 를 얻는다 (항체 키를 알아야 가짜 파일을 만들 수 있다)
  2차: 그 키로 Drive 를 가짜로 채운 뒤 **처음부터 끝까지** 다시 돈다

돌리는 법:  python3 tools/run_in_order.py [노트북경로]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DEFAULT = os.path.join(ROOT, "notebooks", "FvFlow_v12.ipynb")

N_FRAMES = 40           # 실제는 150. 여기서는 빠르게.
N_AB = 5                # 항체(=블록) 수
N_LK = 4                # 항체당 링커 수


# ─────────────────────────────────────────────────────────────────────────────
# 1. Colab 흉내
# ─────────────────────────────────────────────────────────────────────────────
def install_stubs(drive_root):
    """google.colab · ImmuneBuilder · anarci 를 가짜로 끼운다."""
    colab = types.ModuleType("google.colab")
    d = types.ModuleType("google.colab.drive")
    d.mount = lambda *a, **k: os.makedirs(f"{drive_root}/MyDrive", exist_ok=True)
    colab.drive = d
    colab.userdata = types.SimpleNamespace(get=lambda k: None)
    g = sys.modules.setdefault("google", types.ModuleType("google"))
    g.colab = colab
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.drive"] = d

    ib = types.ModuleType("ImmuneBuilder")

    class _Pred:
        ranking = [0, 1, 2, 3]
        def __init__(self, nH, nL): self.nH, self.nL = nH, nL
        def save_single_unrefined(self, path, index=0):
            # ★ 진짜 원자를 쓴다. 빈 PDB 를 쓰면 3절-B 가 구조를 못 읽는 경로로
            #   빠져서, 정작 재고 싶은 계면 계산이 한 번도 안 돌아간다.
            write_ref_pdb(path, nH=self.nH, nL=self.nL)

    class ABodyBuilder2:
        def __init__(self, *a, **k): pass
        def predict(self, seqs):
            return _Pred(len(seqs.get("H", "A" * 120)), len(seqs.get("L", "A" * 110)))

    ib.ABodyBuilder2 = ABodyBuilder2
    sys.modules["ImmuneBuilder"] = ib


def strip_magic(src):
    return "\n".join(("#MAGIC " + l) if l.lstrip().startswith(("!", "%")) else l
                     for l in src.split("\n"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 가짜 입력
# ─────────────────────────────────────────────────────────────────────────────
def write_excel(path):
    """실제 test_result.xlsx 와 같은 열 구조."""
    rng = np.random.default_rng(0)
    rows = []
    for b in range(1, N_AB + 1):
        # H 사슬은 FR4 (W-G-x-G) 로 끝나고, L 사슬은 F-x-x-G 로 끝나야 kind() 가 잡는다
        H = ("EVQLVESGGGLVQPGGSLRLSCAAS" + "K"*b + "A"*(88 - b) + "WGQGTLVTVSS")
        L = ("DIQMTQSPSSLSASVGDRVTITCRAS" + "R"*(7 - b) + "A"*(75 + b) + "FGQGTKVEIK")
        for i, Ln in enumerate([15, 18, 21, 25]):
            rows.append({
                "이름": f"K_Sample{b:02d}_Linker{i+1}",
                "Block": b,
                "Type": "scFv",
                "Domain_1": H,
                "Linker_1": "GGGGS" * (Ln // 5),
                "Domain_2": L,
                # ★ 파국 구성체는 물질이 없어 **HMW 를 못 잰다** — NaN 이다.
                #   실측이 정확히 그렇다 (블록3 Whitlow218). 이걸 안 심으면
                #   8절 Stage B 의 NaN 처리 경로가 한 번도 안 돌아서,
                #   sklearn 이 ValueError 를 내는 것을 시험이 못 잡는다.
                "HMW(%)": (float("nan") if (b == 3 and i == 1) else
                           round(4 + 2*b + 0.6*(Ln - 15) + rng.normal(0, .5), 2)),
                "Monomer(%)": round(96 - 2*b - 0.6*(Ln - 15), 2),
                # ★ 수율 열. 항체 사이는 크게(60배), 항체 안은 작게(±30%) —
                #   실측의 구조 그대로다. 그리고 블록 3 의 두 번째 링커에
                #   **파국을 심는다** (형제의 2%). 6절-B 파국 탐지가 이걸 잡아야 한다.
                #   안 심으면 그 분기가 한 번도 안 돌아서 깨져도 모른다.
                "Protein A Purification 생산량(mg/L)": (
                    round(4.7, 2) if (b == 3 and i == 1)
                    else round(5.0 * (3.2 ** b) * (1 + 0.1*i), 1)),
            })
    pd.DataFrame(rows).to_excel(path, index=False)


def write_ref_pdb(path, nH, nL):
    """ABodyBuilder2 기준점 흉내 — H/L 두 사슬, 1부터 번호."""
    rng = np.random.default_rng(abs(hash(path)) % 2**31)
    with open(path, "w") as f:
        k = 0
        for cid, n, off in (("H", nH, np.array([-9., 0, 0])),
                            ("L", nL, np.array([+9., 0, 0]))):
            c = rng.normal(0, 7, (n, 3)) + off
            for i in range(n):
                for an, el in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                    k += 1
                    x, y, z = c[i] + rng.normal(0, .4, 3)
                    f.write(f"ATOM  {k:>5}  {an:<3} ALA {cid}{i+1:>4}    "
                            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {el:>2}\n")
            f.write("TER\n")
        f.write("END\n")


def write_frame(path, n, b1, b2):
    """BioEmu 프레임 흉내 — 단일 사슬, 백본만."""
    rng = np.random.default_rng(abs(hash(path)) % 2**31)
    d1 = rng.normal(0, 7, (b1, 3)) + np.array([-9., 0, 0])
    lk = np.linspace(d1[-1], np.array([9., 0, 0]), b2 - b1) + rng.normal(0, 2, (b2-b1, 3))
    d2 = rng.normal(0, 7, (n - b2, 3)) + np.array([+9., 0, 0])
    ca = np.vstack([d1, lk, d2])
    with open(path, "w") as f:
        k = 0
        for i, p in enumerate(ca, 1):
            for an, el in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                k += 1
                x, y, z = p + rng.normal(0, .3, 3)
                f.write(f"ATOM  {k:>5}  {an:<3} ALA A{i:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {el:>2}\n")
        f.write("END\n")


def populate(OUT, CONS):
    """CONS 를 보고 Drive 를 가짜로 채운다 → 노트북이 이어달리기로 전부 건너뛴다."""
    AB_SD = dict(HL=3.92, HC1=2.18, HC2=3.07, LC1=2.49, LC2=2.25, dc=0.268)
    AB6 = list(AB_SD)
    rng = np.random.default_rng(1)
    frames, geom, iface = [], [], []
    for ab in CONS.항체.unique():
        r0 = CONS[CONS.항체 == ab].iloc[0]
        os.makedirs(f"{OUT}/{ab}", exist_ok=True)
        for i in range(4):
            write_ref_pdb(f"{OUT}/{ab}/ref_m{i}.pdb", int(r0.b1), len(r0.D2))
        shutil.copy(f"{OUT}/{ab}/ref_m0.pdb", f"{OUT}/{ab}/ref.pdb")
    for _, r in CONS.iterrows():
        d = f"{OUT}/{r.항체}/flow"
        os.makedirs(d, exist_ok=True)
        n = len(r.seq)
        for j in range(N_FRAMES):
            p = f"{d}/{r.링커}__BioEmu_s{j}.pdb"
            if not os.path.isfile(p):
                write_frame(p, n, int(r.b1), int(r.b2))
            tag = f"{r.링커}__BioEmu_s{j}"
            dd = {k: 0.0 for k in AB6}
            for k in AB6:
                dd[f"Δ{k}"] = rng.normal(0, AB_SD[k])
            dd["Δdc"] = rng.normal(0.1, 0.3 + 0.04*(r.길이 - 10))
            # ★ 프레임 10개마다 하나씩 **ABangle 무성 오염을 심는다** — dc 는 크게
            #   벗어났다고 말하는데 좌표(도메인Rg)는 그대로인 프레임. 실측에서 27% 가
            #   이랬다. 안 심으면 abangle_sane 게이트가 한 번도 안 돌아서 깨져도 모른다.
            _rot = (j % 10 == 3)
            if _rot:
                dd["Δdc"] = 50.0 + rng.normal(0, 2)
                for k in AB6:
                    if k != "dc": dd[f"Δ{k}"] = rng.normal(0, 40)
            dd["dc"] = 16.206 + dd["Δdc"]
            dd["OCD6"] = sum(abs(dd[f"Δ{k}"])/AB_SD[k] for k in AB6)
            dd["OCD5"] = sum(abs(dd[f"Δ{k}"])/AB_SD[k] for k in AB6 if k != "dc")
            row = dict(블록=r.블록, 항체=r.항체, 배향=r.배향, 링커=r.링커,
                       길이=int(r.길이), 합성=bool(r.합성),
                       조성=r.get("조성", np.nan), 설계길이=r.get("설계길이", np.nan),
                       생성기="BioEmu", 태그=tag, **dd)
            # ★ 실측 열을 **전부** 싣는다. 노트북의 OBS_KEYS 와 같은 규칙이다 —
            #   여기서 수율을 빼면 6절-B 파국 탐지가 한 번도 안 돌아서,
            #   그 분기가 깨져도 시험이 통과해 버린다 (실제로 그랬다).
            for c in CONS.columns:
                if any(k in str(c).lower() or k in str(c) for k in
                       ("hmw", "monomer", "단량체", "purity", "순도", "수율",
                        "생산", "titer", "sec", "quantific", "supernatant")):
                    row[c] = r[c]
            frames.append(row)
            # 오염 프레임은 좌표상 **멀쩡하다** (그게 이 사고의 정의다)
            geom.append(dict(항체=r.항체, 링커=r.링커, 태그=tag,
                             링커접촉_잔기당=12 + rng.normal(0, 1.5),
                             링커밀착율=0.6, 링커최근접=6.0,
                             링커Rg_잔기당=0.3, 링커신장도=0.6,
                             링커나선도=0.12 + rng.normal(0, .05),
                             링커i_i4=11.0,
                             도메인Rg=18.0 + (0.0 if _rot else rng.normal(0, 1.2)),
                             말단간=45.0))
        json.dump(dict(요청=N_FRAMES, 저장=N_FRAMES, 잔존율=1.0),
                  open(f"{d}/{r.링커}__BioEmu.done", "w"))
    pd.DataFrame(frames).to_csv(f"{OUT}/frames.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(geom).to_csv(f"{OUT}/geom.csv", index=False, encoding="utf-8-sig")
    for i, ab in enumerate(sorted(CONS.항체.unique())):
        iface.append(dict(항체=ab, 계면_BSA=700 + 60*i, 계면_BSA_SD=12.0,
                          계면_접촉수=180 + 12*i, 계면_잔기수=44 + i,
                          계면_접촉밀도=4.0 + .2*i, 계면_접촉밀도_SD=.05,
                          계면_소수성=.42 + .02*i, 계면_소수성_SD=.004,
                          기준점_흔들림=.6 - .05*i, 기준점_흔들림_SD=.02, n모델=4))
    pd.DataFrame(iface).to_csv(f"{OUT}/interface.csv", index=False, encoding="utf-8-sig")
    # ★ 가짜 calvados.csv — 10절-D 판정 경로를 실제로 태운다.
    #   B22_닫힘 은 양수(밀어냄), B22_열림 은 음수(끈끈함) 로 두어 혼합식이
    #   의미 있는 값을 내게 한다. 열림분율은 구성체마다 다르게 준다.
    crow = []
    for _, r in CONS.drop_duplicates(["항체", "링커"]).iterrows():
        h = abs(hash((r.항체, r.링커))) % 97
        crow.append(dict(항체=r.항체, 링커=r.링커,
                         B22=80.0 - h, B22_무차원=0.9 - h/200,
                         B22_닫힘=120.0 - h, B22_열림=-300.0 - 3*h,
                         B22_열림_SD=18.0 + h/20,
                         끈끈한면_노출=0.35 + h/400, 링커_훑는부피=1.2 + h/200,
                         분자간접촉=12.0 + h/10, 열림분율=0.10 + (h % 30)/100))
    pd.DataFrame(crow).to_csv(f"{OUT}/calvados.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(항체=ab, ipTM=0.91, ipTM_SD=.01, pTM=.88, 계면PAE=4.0,
                       계면PAE_SD=.05, nH=120, nL=118, n모델=3)
                  for ab in sorted(CONS.항체.unique())]).to_csv(
        f"{OUT}/iptm.csv", index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 실행
# ─────────────────────────────────────────────────────────────────────────────
def fake_abangle(paths, chunk=400, quiet=False):
    """ABangle 흉내 — 파일마다 결정적인 6값. 실제 하위 프로세스를 안 띄운다."""
    out = {}
    for p in paths:
        rng = np.random.default_rng(abs(hash(p)) % 2**31)
        out[p] = dict(HL=-58 + rng.normal(0, 4), HC1=70 + rng.normal(0, 2),
                      HC2=118 + rng.normal(0, 3), LC1=118 + rng.normal(0, 2.5),
                      LC2=83 + rng.normal(0, 2), dc=16.2 + rng.normal(0, .3))
    return out


def base_env(drive_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def display(x):
        if isinstance(x, pd.DataFrame):
            print(x.to_string(index=False)[:1500])
        else:
            print(x)

    g = {"__builtins__": __builtins__, "display": display}
    return g


def run(path, stop_on_error=True, over=None, empty=False):
    """over: 매 셀 뒤에 덮어쓸 스위치 dict.  empty: 가짜 v09 출력을 안 채운다."""
    over = dict(over or {})
    nb = json.load(open(path, encoding="utf-8"))
    src = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    tmp = tempfile.mkdtemp(prefix="fvorder_")
    drive = f"{tmp}/drive"
    os.makedirs(f"{drive}/MyDrive/FvTwist/input", exist_ok=True)
    OUT = f"{drive}/MyDrive/FvTwist/fvflow"
    os.makedirs(OUT, exist_ok=True)
    write_excel(f"{drive}/MyDrive/FvTwist/input/test_result.xlsx")
    install_stubs(drive)

    def patch(g):
        """매 셀 뒤에 다시 건다 — 셀이 진짜 구현으로 덮어쓰기 때문이다."""
        g["DRIVE"] = f"{drive}/MyDrive/FvTwist"
        g["IN"] = f"{drive}/MyDrive/FvTwist/input"
        g["OUT"] = OUT
        g["RUN_BIOEMU"] = False
        g["RUN_PANEL"] = True
        g["RUN_IPTM"] = False
        g["N_PERM"] = 120           # 순열을 줄여 빠르게
        g["abangle6_many"] = fake_abangle
        g["sh"] = lambda *a, **k: False
        g["venv311"] = lambda *a, **k: f"{tmp}/venv"
        g["venv_metrics"] = lambda *a, **k: f"{tmp}/venv"
        g["venv_be"] = lambda *a, **k: f"{tmp}/venv"
        g["N_SEED_REP"] = 0
        # ★ 이제 기본이 True 다. 여기선 calvados 설치가 안 되므로 HAVE_CALV=False
        #   경로를 타게 둔다 — 그 분기(설치 실패 → 이유를 찍고 멈춤)도 시험 대상이다.
        g["RUN_CALVADOS"] = True
        # ★ 그래도 10절-D 의 **판정 경로**는 돌려야 한다. 그래서 가짜 calvados.csv 를
        #   미리 깔아 둔다 (populate 가 쓴다) — 혼합식·프레임잡음·항체내몫 분기가
        #   한 번도 안 돌면 깨져도 모른다.
        g.update(over)                # ★ 호출자가 준 스위치가 마지막에 이긴다

    g = base_env(drive)
    fails = []
    print(f"작업 디렉터리 {tmp}")
    print("=" * 78)
    for i, (s, t) in enumerate(src):
        if t != "code":
            continue
        patch(g)
        head = s.strip().split("\n")[0][:62]
        print(f"\n▶ 셀 {i:2d}  {head}")
        # 4번 셀이 끝나면 CONS 를 보고 Drive 를 채운다 (항체 키를 그때 알 수 있다)
        try:
            exec(compile(strip_magic(s), f"<cell {i}>", "exec"), g)
        except Exception as e:
            tb = traceback.format_exc()
            ln = ""
            for L in tb.split("\n"):
                if f"<cell {i}>" in L:
                    ln = L.strip()
            print(f"   ★ {type(e).__name__}: {str(e)[:200]}")
            if ln:
                print(f"     {ln}")
            # 앞 셀이 이미 죽었고 이번 것이 NameError 면 **연쇄**다 — 앞 것을 고치면
            # 같이 없어진다. 독립된 사고 8건처럼 세면 어디를 고칠지 안 보인다.
            cascade = bool(fails) and isinstance(e, NameError)
            # assert 는 이 노트북에서 **설계된 멈춤**이다 (앞 절이 안 끝났다고 말해 준다).
            # 버그와 같은 칸에 세면 고칠 자리가 안 보인다.
            fails.append(dict(셀=i, 머리=head, 오류=f"{type(e).__name__}: {str(e)[:160]}",
                              위치=ln, 연쇄=cascade,
                              설계=isinstance(e, AssertionError)))
            if stop_on_error and len(fails) >= 1 and i <= 4:
                break
        if i == 4 and "CONS" in g and not empty:
            print("   … Drive 를 가짜로 채운다 (이어달리기 경로를 타게 한다)")
            populate(OUT, g["CONS"])
    print("\n" + "=" * 78)
    root = [f for f in fails if not f["연쇄"] and not f["설계"]]
    if fails:
        print(f"순서대로 돌렸을 때 죽는 셀 {len(fails)}개"
              + (f" (그중 {len(fails)-len(root)}개는 앞 셀 때문에 따라 죽은 연쇄)"
                 if len(root) < len(fails) else ""))
        for f in fails:
            tagf = "[연쇄] " if f["연쇄"] else ("[설계된 멈춤] " if f["설계"] else "")
            print(f"  셀 {f['셀']:2d}  {tagf}{f['오류']}")
            if f["위치"]:
                print(f"          {f['위치']}")
        if root:
            print(f"\n  먼저 고칠 것: 셀 {', '.join(str(f['셀']) for f in root)}")
        else:
            print("\n  고칠 것 없음 — 전부 설계된 멈춤과 그 연쇄다.")
    else:
        print("셀 0번부터 끝까지 순서대로 전부 통과")
    shutil.rmtree(tmp, ignore_errors=True)
    return root        # ★ 설계된 멈춤과 그 연쇄는 실패로 안 센다


def _parse_argv(argv):
    """`노트북경로  --empty  --set RUN_ML=False  --set RUN_PANEL=False`"""
    path, over, empty, i = NB_DEFAULT, {}, False, 0
    while i < len(argv):
        a = argv[i]
        if a == "--empty":
            empty = True
        elif a == "--set":
            i += 1
            k, _, v = argv[i].partition("=")
            over[k] = {"True": True, "False": False}.get(v, v)
        else:
            path = a
        i += 1
    return path, over, empty


if __name__ == "__main__":
    p, over, empty = _parse_argv(sys.argv[1:])
    if over or empty:
        print(f"스위치 {over}" + ("  · 빈 Drive" if empty else ""))
    sys.exit(1 if run(p, stop_on_error=False, over=over, empty=empty) else 0)
