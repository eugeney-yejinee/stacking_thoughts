#!/usr/bin/env python3
"""FvLinker · pH 3 풀림 파이프라인 — 회사에서 **2~3시간 안에** 돌려 보는 용도.

════════════════════════════════════════════════════════════════════════════
무엇을 하는가
════════════════════════════════════════════════════════════════════════════
  링커만 다른 scFv 구조  →  pH 7.4 와 pH 3.0 에서 각각 MD (씨앗 반복 포함)
                         →  두 궤적의 **차분**에서 무엇이 드러나는지 잰다
                         →  드러난 끈끈한 면의 크기로 항체 **안에서** HMW 순위를 예측

왜 차분인가
  100 ns 로는 도메인이 안 풀린다 (실제 언폴딩은 ms 이상). "풀렸나" 를 보면
  아무 일도 안 일어난 것처럼 보이고 잘못된 결론이 난다. 재야 하는 것은
  **pH 7 대비 pH 3 에서 무엇이 달라졌나** 다 — 끊어진 염다리, 풀린 계면
  가장자리, 링커가 옮겨간 자리, 그래서 새로 드러난 소수성 면.

왜 두 분자를 안 넣는가
  k_assoc = k_encounter × P_stick 인데 만남 빈도는 링커가 바뀌어도 거의 안
  변한다 (k ∝ R·D, D ∝ 1/R). 변하는 건 **붙을 확률**이고 그건 드러난 끈끈한
  면으로 정해진다 — **한 분자 계산**이다. 전원자 회합 PMF 는 쌍당 약 44 GPU-일.

════════════════════════════════════════════════════════════════════════════
관측 패널 — Δ소수성SASA 하나로는 얇다
════════════════════════════════════════════════════════════════════════════
  A. 노출 (붙을 확률)    SAP_max · SAP_sum · 최대연속패치 · 소수성SASA
  B. 계면 (f 항)         VH-VL 접촉수 · 염다리수 · Rg
  C. 링커 (c 항)         링커노출(가림) · 링커Re · 링커Rg
  D. 분모               씨앗간 SD · 블록 표류(수렴) · **항체내몫**

  ★ 항체내몫을 **y(HMW)를 보기 전에** 전 관측값에 대해 찍는다. 0.3 미만이면
    그 관측값은 링커가 아니라 항체 정체를 재고 있다 — 지금까지 이 프로젝트의
    모든 축이 거기서 죽었다. 살아남은 것만 HMW 와 대조한다.

════════════════════════════════════════════════════════════════════════════
사용법
════════════════════════════════════════════════════════════════════════════
    # ① 먼저 이것만. GPU 도 Drive 도 필요 없다. CPU 에서 4~6분.
    python3 fvlinker_ph3md.py --selftest

    # ② 진짜 구조로 배선 확인 + 시간 추정. 본 실행은 안 한다.
    python3 fvlinker_ph3md.py --preflight

    # ③ 본 실행. 주어진 시간에 맞춰 **스스로 규모를 정하고 버린 것을 찍는다.**
    python3 fvlinker_ph3md.py --budget-hours 3

    # 끊기면 같은 명령을 다시. 체크포인트에서 이어간다.

설치 (Colab):
    pip install pdbfixer mdtraj openpyxl
    pip install "openmm[cuda12]"       # ★ GPU 박스에서만. conda 불필요. CUDA 12 전용
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 경로 ────────────────────────────────────────────────────────────────────
DRIVE = os.environ.get("FVL_DRIVE", "/content/drive/MyDrive/FvTwist")
IN = os.environ.get("FVL_IN", f"{DRIVE}/input")
OUT = os.environ.get("FVL_OUT", f"{DRIVE}/fvflow")   # ★ build_v12.py 와 같은 경로.
WORK = f"{OUT}/ph3md"                                 #   이름이 FvLinker 여도 안 바꾼다

# ── 시뮬레이션 상수 ─────────────────────────────────────────────────────────
PH_LIST = [7.4, 3.0]          # 이 둘의 **차분**이 신호다
DT_PS = 0.004                 # 4 fs — 수소질량 재분배 + HBonds 구속
SAVE_PS = 20.0                # 프레임 저장 간격
EQ_NS = 0.2
MIN_ITERS = 2000
PAD_NM = 1.0
IONIC_M = 0.05
TEMP_K = 300.0
OBS_KEYS = ("hmw", "monomer", "단량체", "purity", "순도")

W = 78


_FF_CACHE = {}


def get_ff(ffpair):
    """★ ForceField 를 **한 번만** 만든다. charmm36_2024.xml 은 16.9 MB 라 생성에
    5.7초가 걸리는데, 예전에는 prepare() 를 부를 때마다 새로 만들었다 —
    구성체 19 × pH 2 × 씨앗 2 = 76회면 그것만 7분이다."""
    from openmm.app import ForceField
    k = tuple(ffpair)
    if k not in _FF_CACHE:
        _FF_CACHE[k] = ForceField(*k)
    return _FF_CACHE[k]


def save_json(obj, path):
    """★ 원자적으로 쓴다. 이 파일은 **완료된 계 전부**를 담고 있어서, 쓰는 도중에
    Colab 이 끊기면 지금까지 돌린 몇 시간이 통째로 날아간다. 임시 파일에 다 쓰고
    이름만 바꾼다 — rename 은 같은 파일시스템에서 원자적이다."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_json(path):
    """깨진 json 을 만나면 **조용히 빈 값으로 시작하지 않는다** — 크게 알린다."""
    if not os.path.isfile(path):
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:
        bad = path + ".corrupt"
        os.replace(path, bad)
        print(f"  ★★ {os.path.basename(path)} 가 깨졌다 ({type(e).__name__}). "
              f"{os.path.basename(bad)} 로 옮기고 처음부터 간다.", flush=True)
        return {}


def head(t, c="="):
    print("\n" + c * W); print(t); print(c * W, flush=True)
def say(*a):
    print(*a, flush=True)


# ════════════════════════════════════════════════════════════════════════════
# [0] 환경
# ════════════════════════════════════════════════════════════════════════════
def env(force_cpu=False):
    head("[0] 환경", "─")
    for m in ("openmm", "pdbfixer", "mdtraj"):
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", None) or \
                getattr(getattr(mod, "version", None), "version", "설치됨")
            say(f"  ✔ {m} {v}")
        except Exception as e:
            say(f"  ✗ {m} 없음 ({type(e).__name__})")
            say(f"\n  ★ 설치:  pip install pdbfixer mdtraj openpyxl")
            say('     GPU:    pip install "openmm[cuda12]"')
            raise SystemExit(1)

    import openmm as om
    names = []
    for n in ("CUDA", "HIP", "OpenCL", "CPU", "Reference"):
        try:
            om.Platform.getPlatformByName(n); names.append(n)
        except Exception:
            pass
    plat = "CPU" if force_cpu else (names[0] if names else "Reference")
    say(f"  등록 플랫폼 {names} → **{plat}**")
    if plat in ("CPU", "Reference") and not force_cpu:
        say("  ★★ GPU 가 안 잡혔다. CPU 로는 계당 며칠 걸린다.")
        say('     Colab 런타임을 GPU 로 바꾸고  pip install "openmm[cuda12]"  해라.')
        say("     (그래도 계속 가려면 --force-cpu)")
        raise SystemExit(1)

    # ★ 힘장 짝이 맞아야 한다. charmm36_2024.xml 은 charmm36/water.xml 과 **안 맞는다**
    #   (RuntimeError). C36m = par_all36m_prot.prm 를 쓰는 charmm36_2024 쪽이다.
    #   무질서 링커를 다루므로 C36m 이 아니면 링커가 인위적으로 뭉친다.
    from openmm.app import ForceField
    for pair in (("charmm36_2024.xml", "charmm36_2024/water.xml"),
                 ("charmm36.xml", "charmm36/water.xml")):
        try:
            ForceField(*pair)
            note = "C36m — 무질서 영역용" if "2024" in pair[0] else "★ 구판 C36 (링커가 뭉칠 수 있다)"
            say(f"  힘장 **{pair[0]}** + {pair[1]}   {note}")
            return plat, pair
        except Exception as e:
            say(f"  ✗ {pair[0]} + {pair[1]} → {type(e).__name__}")
    raise SystemExit("★ 쓸 수 있는 CHARMM 힘장이 없다")


# ════════════════════════════════════════════════════════════════════════════
# [1] 구조와 라벨
# ════════════════════════════════════════════════════════════════════════════
def read_excel():
    """엑셀 → (항체후보키, 링커) 별 링커서열과 HMW. 없으면 빈 표로 계속 간다.

    링커 **서열**이 필요하다 — 궤적에서 링커 구간을 서열로 찾기 때문이다.
    잔기 번호나 사슬 나눔에 기대면 조용히 엉뚱한 자리를 집는다.
    """
    import pandas as pd
    c = (sorted(glob.glob(f"{IN}/*test_result*.xls*"))
         or sorted(glob.glob(f"{IN}/*.xls*")))
    if not c:
        say(f"  엑셀 ★없음 ({IN}) — 링커 구간과 HMW 없이 간다")
        return pd.DataFrame(columns=["링커", "링커서열", "HMW", "블록"])
    X = pd.read_excel(c[-1]); X.columns = [str(s).strip() for s in X.columns]
    col = lambda p: next((k for k in X.columns if k.lower().replace(" ", "") == p), None)
    nm = col("이름") or col("name") or col("sample") or X.columns[0]
    blk = col("block") or col("블록")
    lc = sorted([k for k in X.columns if k.lower().replace(" ", "").startswith("linker")],
                key=lambda k: int("".join(f for f in k if f.isdigit()) or 0))
    hm = next((k for k in X.columns if "hmw" in k.lower()), None)
    rows = []
    for _, r in X.iterrows():
        seq = next((str(r[k]).strip().upper() for k in lc
                    if isinstance(r[k], str) and len(str(r[k]).strip()) >= 5), "")
        rows.append(dict(링커=str(r[nm]).strip().rsplit("_", 1)[-1], 링커서열=seq,
                         블록=re.sub(r"\.0$", "", str(r[blk])) if blk else "1",
                         HMW=pd.to_numeric(r[hm], errors="coerce") if hm else np.nan))
    d = pd.DataFrame(rows)
    say(f"  엑셀     {os.path.basename(c[-1])} — 링커 {d.링커.nunique()}종"
        + (f" · HMW 열 '{hm}'" if hm else " · ★HMW 열 없음"))
    return d


def find_structures(lab):
    """{OUT}/<항체>/flow/<링커>__BioEmu_s*.pdb  (없으면 <항체>/ref*.pdb)."""
    head("[1] 시작 구조", "─")
    found = []
    for abdir in sorted(glob.glob(f"{OUT}/*/")):
        ab = os.path.basename(abdir.rstrip("/"))
        if ab in ("ph3md", "calvados", "input", "fig"):
            continue
        by = {}
        for p in sorted(glob.glob(f"{abdir}flow/*__BioEmu_s*.pdb")):
            by.setdefault(os.path.basename(p).split("__BioEmu")[0], []).append(p)
        for lk, ps in sorted(by.items()):
            found.append(dict(항체=ab, 링커=lk, pdb=ps[0], 출처="BioEmu"))
        if not by:
            for p in sorted(glob.glob(f"{abdir}ref*.pdb"))[:1]:
                found.append(dict(항체=ab, 링커="ref", pdb=p, 출처="ABB2"))
    if not found:
        say(f"  ★ {OUT} 아래에서 PDB 를 못 찾았다.")
        say(f"    찾아본 곳: {OUT}/<항체>/flow/<링커>__BioEmu_s*.pdb")
        say(f"               {OUT}/<항체>/ref*.pdb")
        say("    경로가 다르면 환경변수로:  FVL_OUT=/실제/경로 python3 fvlinker_ph3md.py …")
        raise SystemExit(1)

    seqmap = dict(zip(lab.링커, lab.링커서열)) if len(lab) else {}
    hmwmap = dict(zip(lab.링커, lab.HMW)) if len(lab) else {}
    for f in found:
        f["링커서열"] = seqmap.get(f["링커"], "")
        f["HMW"] = float(hmwmap.get(f["링커"], np.nan))

    # 항체 안에서 링커가 2종 이상인 것만 — 그게 검정 가능한 유일한 대조다
    from collections import Counter
    cnt = Counter(f["항체"] for f in found)
    keep = [f for f in found if cnt[f["항체"]] >= 2]
    say(f"  구성체 {len(found)} · 항체 {len(cnt)}")
    say(f"  **링커 2종 이상 항체 {len({f['항체'] for f in keep})}개 / 구성체 {len(keep)}개**"
        f" 가 검정 대상 (나머지 {len(found)-len(keep)}개는 항체내 대조가 없어 제외)")
    noseq = [f"{f['항체']}/{f['링커']}" for f in keep if not f["링커서열"]]
    if noseq:
        say(f"  ★ 링커 서열을 못 찾은 구성체 {len(noseq)}개 — 링커 관측값은 NaN 이 된다:")
        say(f"    {noseq[:6]}")
    return keep or found


# ════════════════════════════════════════════════════════════════════════════
# [2] 준비 — 조용히 틀리는 자리
# ════════════════════════════════════════════════════════════════════════════
def prepare(pdb_in, ph, ffpair, pad_nm=None, solvate=True):
    """수소 제거 → pH별 양성자화 → 용매화. 검산값을 함께 돌려준다.

    ★ 수소를 **먼저 전부 지운다.** createDisulfideBonds 는 HG 가 없는 Cys 만
      이황화 후보로 본다. 수소 붙은 PDB 를 넣으면 scFv 이황화가 오류 없이
      전부 사라진다 — scFv 는 도메인당 이황화가 있으니 그 MD 는 쓰레기가 된다.

    ★ addHydrogens(pH=3.0) 은 Asp/Glu 를 실제로 중성화한다 (hydrogens.xml 의
      maxph=4.4). His 만 건드린다는 통설은 틀렸다. 다만 pH ≤ 4.4 **전역 계단**이라
      환경 민감도가 없다 — pH 3.0 은 거의 모든 Asp/Glu pKa 아래라 맞고,
      pH 4~5 였다면 틀렸을 것이다.
    """
    from openmm import unit
    from openmm.app import (HBonds, Modeller, NoCutoff, PDBFile, PME, element)
    from pdbfixer import PDBFixer

    fx = PDBFixer(filename=pdb_in)
    fx.findMissingResidues()
    fx.findNonstandardResidues()
    fx.replaceNonstandardResidues()
    fx.removeHeterogens(keepWater=False)      # ★ 물·리간드를 **수소 처리 전에** 뺀다
    fx.findMissingAtoms()
    fx.addMissingAtoms()                      # BioEmu 뼈대 → 곁사슬 복원

    ff = get_ff(ffpair)
    m = Modeller(fx.topology, fx.positions)
    m.delete([a for a in m.topology.atoms() if a.element == element.hydrogen])  # ★ 필수
    m.addHydrogens(ff, pH=float(ph))

    ss = sum(1 for b in m.topology.bonds()
             if getattr(b[0], "name", "") == "SG" and getattr(b[1], "name", "") == "SG")
    nres = m.topology.getNumResidues()
    seq_top = m.topology

    if solvate:
        m.addSolvent(ff, model="tip3p",
                     padding=(PAD_NM if pad_nm is None else pad_nm) * unit.nanometer,
                     neutralize=True, ionicStrength=IONIC_M * unit.molar)
        sysm = ff.createSystem(m.topology, nonbondedMethod=PME,
                               nonbondedCutoff=1.0 * unit.nanometer, constraints=HBonds,
                               rigidWater=True, hydrogenMass=1.5 * unit.amu)
    else:
        # ★ 진공. **자체시험 전용**이다 — 물을 안 채우므로 배선만 확인하고 물리는 못 믿는다.
        #   addSolvent 59s + createSystem 31s 가 이 계에서 준비 시간의 절반을 넘는다.
        sysm = ff.createSystem(m.topology, nonbondedMethod=NoCutoff,
                               constraints=HBonds, hydrogenMass=1.5 * unit.amu)
    nb = next(f for f in sysm.getForces() if f.__class__.__name__ == "NonbondedForce")
    prot = [a.index for a in m.topology.atoms()
            if a.residue.name not in ("HOH", "WAT", "NA", "CL", "SOD", "CLA")]
    q = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in prot)
    return dict(topology=m.topology, positions=m.positions, system=sysm,
                atoms=sysm.getNumParticles(), ss=ss, q=round(float(q), 1), nres=nres,
                protein_top=seq_top)


def simulate(prep, plat, steps, chk, dcd, every, seed=0, resume=True,
             min_iters=MIN_ITERS, eq_ns=EQ_NS):
    """돌리고 **프로덕션 구간만의 초 수**를 같이 돌려준다.

    ★ 시간 추정은 프로덕션만으로 해야 한다. 최소화·평형은 계당 한 번뿐인데
      CPU 에서는 그게 전체의 대부분이라, 같이 재면 추정이 몇 배 부풀려진다.
    """
    from openmm import (LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit)
    from openmm.app import CheckpointReporter, DCDReporter, Simulation

    sysm = prep["system"]
    # ★ 진공(비주기)계에 barostat 을 붙이면 OpenMM 이 죽는다. 주기 상자가 있을 때만.
    if sysm.usesPeriodicBoundaryConditions() and not any(
            f.__class__.__name__ == "MonteCarloBarostat" for f in sysm.getForces()):
        sysm.addForce(MonteCarloBarostat(1 * unit.bar, TEMP_K * unit.kelvin, 25))
    integ = LangevinMiddleIntegrator(TEMP_K * unit.kelvin, 1 / unit.picosecond,
                                     DT_PS * unit.picoseconds)
    integ.setRandomNumberSeed(int(seed) + 1)
    sim = Simulation(prep["topology"], sysm, integ, Platform.getPlatformByName(plat))
    done = 0
    if resume and chk and os.path.isfile(chk):
        try:
            sim.loadCheckpoint(chk)
            done = int(sim.context.getStepCount())
            say(f"        체크포인트 재개 ({done:,} 스텝 완료)")
        except Exception as e:
            say(f"        ★ 체크포인트를 못 읽었다 ({type(e).__name__}) — 처음부터")
            done = 0
    if done == 0:
        sim.context.setPositions(prep["positions"])
        sim.minimizeEnergy(maxIterations=min_iters)
        sim.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin, int(seed) + 1)
        if eq_ns > 0:
            sim.step(int(eq_ns * 1000 / DT_PS))
    left = max(0, steps - done)
    t0 = time.time()
    if left:
        if dcd:
            sim.reporters.append(DCDReporter(dcd, every, append=os.path.isfile(dcd)))
        if chk:
            # ★ 체크포인트 간격을 **DCD 와 똑같이** 둔다. 두 리포터가 같은 스텝에서
            #   같이 발화하므로 재개했을 때 프레임 수와 스텝 수가 어긋나지 않는다.
            #   예전에는 max(1000, every) 였는데, 그러면 (a) 짧은 실행은 체크포인트가
            #   아예 안 써지고 (b) 긴 실행은 DCD 가 체크포인트보다 앞서 나가
            #   재개할 때 프레임이 중복된다.
            sim.reporters.append(CheckpointReporter(chk, every))
        sim.step(left)
    # ★ 끝에서 한 번 더 저장한다. 안 하면 마지막 구간이 날아가고, 다 끝난 계를
    #   다시 실행했을 때 처음부터 또 돈다.
    if chk:
        try:
            sim.saveCheckpoint(chk)
        except Exception as e:
            say(f"        ★ 마지막 체크포인트 저장 실패: {type(e).__name__}")
    return sim, time.time() - t0, left


# ════════════════════════════════════════════════════════════════════════════
# [3] 관측
# ════════════════════════════════════════════════════════════════════════════
def measure(top_pdb, dcd, linker_seq=""):
    """궤적 → 관측 패널. fvobs 의 시험된 함수만 쓴다."""
    import mdtraj as md
    import fvobs as F

    # ★ 단백질 원자만 읽는다. 통째로 읽으면 실제 규모(45,000원자 × 1,000프레임)에서
    #   540 MB 를 한 번에 올리게 돼 Colab 에서 OOM 이 난다. 단백질만이면 48 MB 다.
    ref = md.load(top_pdb)
    sel = ref.topology.select("protein")
    if len(sel) == 0:
        raise ValueError("위상에 단백질 원자가 없다")
    t = md.load(dcd, top=top_pdb, atom_indices=sel)
    if t.n_frames == 0:
        raise ValueError("프레임이 0개다 — 저장 간격이 프로덕션보다 길지 않은지 봐라")

    s = F.sap(t)
    area, patch = F.largest_patch(t)
    nsb, _ = F.salt_bridges(t)
    rel = F.rel_exposure(t)
    per = md.shrake_rupley(t, mode="residue").mean(axis=0)
    hyd = np.array([F.BLACK_MOULD.get(r.name.upper(), 0.5) - 0.5 > 0
                    for r in t.topology.residues])
    o = dict(
        SAP_max=float(np.nanmax(s)) if np.isfinite(s).any() else float("nan"),
        SAP_sum=float(np.nansum(np.clip(s, 0, None))),
        최대연속패치=float(area),
        소수성SASA=float(per[hyd].sum()),
        총SASA=float(per.sum()),
        염다리수=int(nsb),
        Rg=float(md.compute_rg(t).mean()) * 10.0,          # Å
        n프레임=int(t.n_frames), n잔기=int(t.n_residues),
        평균노출=float(np.nanmean(rel)),
    )
    # 링커 관련 — 서열로 구간을 찾는다. 못 찾으면 **NaN 으로 두고 왜인지 남긴다**
    span = F.find_span(F.traj_sequence(t), (linker_seq or "").strip().upper())
    if span and patch:
        lo, hi = span
        lk = list(range(lo, hi))
        o["링커노출"] = float(F.linker_shield(t, patch, lk))
        re_, rg_ = F.chain_dims(t, lk)
        o["링커Re"] = re_; o["링커Rg"] = rg_
        o["링커구간"] = f"{lo}-{hi}"
    else:
        o["링커노출"] = o["링커Re"] = o["링커Rg"] = float("nan")
        o["링커구간"] = ("서열 못 찾음" if not span else "패치 없음")
    # 수렴 — Rg 의 블록 표류
    o["Rg표류"] = float(F.block_drift(md.compute_rg(t)))
    return o


# ════════════════════════════════════════════════════════════════════════════
# [4] 자체 시험 — GPU 도 Drive 도 없이 전 경로를 검증한다
# ════════════════════════════════════════════════════════════════════════════
def selftest():
    head("자체 시험 — 내장 소형계로 전 경로를 확인한다 (GPU·Drive 불필요)", "=")
    import shutil
    import tempfile

    import openmm.app as app
    plat, ffpair = env(force_cpu=True)
    tmp = tempfile.mkdtemp(prefix="fvl_selftest_")
    try:
        src = os.path.join(os.path.dirname(app.__file__), "data", "test.pdb")
        if not os.path.isfile(src):
            say(f"  ★ OpenMM 의 시험 구조를 못 찾았다: {src}"); return 1
        pdb = app.PDBFile(src)
        m = app.Modeller(pdb.topology, pdb.positions)
        m.deleteWater()
        m.delete([a for a in m.topology.atoms()
                  if a.residue.name in ("CL", "NA", "SOD", "CLA")])
        p0 = f"{tmp}/mini.pdb"
        with open(p0, "w") as fh:
            app.PDBFile.writeFile(m.topology, m.positions, fh)
        say(f"  시험계: villin 조각, 잔기 {m.topology.getNumResidues()}")

        # ★ 자체시험은 **배선 확인**이지 물리가 아니다. 상자를 작게(0.45 nm), 최소화를
        #   짧게(20회) 하고 **물을 안 채운다**. 물을 채우면 이 계에서만
        #   addSolvent 59s + createSystem 31s 라 pH 하나에 750초가 걸려서 아무도 안 돌린다.
        #   물리는 못 믿지만 배선(양성자화·이황화·관측·재개)은 전부 밟는다.
        qs, ok, preps = {}, True, {}
        for ph in PH_LIST:
            t0 = time.time()
            say(f"  pH {ph} 준비 중… (이 계에서 1~2분)")
            p = prepare(p0, ph, ffpair, solvate=False)
            qs[ph] = p["q"]; preps[ph] = p0
            top = f"{tmp}/ph{ph}.pdb"
            with open(top, "w") as fh:
                app.PDBFile.writeFile(p["topology"], p["positions"], fh)
            dcd = f"{tmp}/ph{ph}.dcd"
            _, ps, _ = simulate(p, plat, 120, None, dcd, 30, seed=1, resume=False,
                                min_iters=20, eq_ns=0.0)
            o = measure(top, dcd, linker_seq="")
            say(f"  pH {ph}: 원자 {p['atoms']:,} · 순전하 {p['q']:+.1f} · "
                f"이황화 {p['ss']} · 프레임 {o['n프레임']} · "
                f"SAP_max {o['SAP_max']:+.2f} · 패치 {o['최대연속패치']:.2f} nm² · "
                f"염다리 {o['염다리수']}  ({time.time()-t0:.0f}s)")

        dq = qs[3.0] - qs[7.4]
        say(f"\n  ★ 양성자화 검산: pH 7.4 → {qs[7.4]:+.1f} · pH 3.0 → {qs[3.0]:+.1f}"
            f"  (차이 **{dq:+.1f}**)")
        if dq < 2:
            say("    ★★ 실패 — pH 3 에서 전하가 안 올랐다. Asp/Glu 가 중성화되지 않았다.")
            ok = False
        else:
            say("    ✔ Asp/Glu 가 실제로 중성화된다 — 진짜 pH 3 조건이다.")

        # 체크포인트 재개 경로도 여기서 한 번 밟아 본다 — 본 실행에서 처음 만나면 늦다
        say("\n  체크포인트 재개 경로:")
        try:
            # prepare() 가 이 계에서 ~60초라 두 번 더 부르면 자체시험이 2분 늘어난다.
            # 같은 prep 으로 Simulation 만 두 번 만들면 재개 경로는 그대로 밟힌다.
            ck = f"{tmp}/rs.chk"; dd = f"{tmp}/rs.dcd"
            pc = prepare(p0, 7.4, ffpair, solvate=False)
            simulate(pc, plat, 60, ck, dd, 30, seed=1, resume=False, min_iters=20, eq_ns=0.0)
            _, _, left = simulate(pc, plat, 60, ck, dd, 30, seed=1, resume=True,
                                  min_iters=20, eq_ns=0.0)
            import mdtraj as _md
            nfr = _md.load(dd, top=f"{tmp}/ph7.4.pdb").n_frames if os.path.isfile(dd) else -1
            say(f"    재개 후 남은 스텝 {left} (0 이어야 한다) · DCD 프레임 {nfr} "
                f"(2 여야 한다 — 중복 없이)")
            good = (left == 0) and (nfr == 2)
            say("    " + ("✔ 재개가 정확하다 — 중간에 끊겨도 이어서 간다" if good
                          else "★★ 재개가 어긋난다. 끊기면 결과가 망가진다"))
            ok = ok and good
        except Exception as e:
            say(f"    ★ 재개 실패: {type(e).__name__}: {e}"); ok = False

        say("\n  관측 함수 자체 시험:")
        r = os.system(f"{sys.executable} "
                      f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_fvobs.py')}"
                      f" > {tmp}/obs.log 2>&1")
        tail = open(f"{tmp}/obs.log").read().strip().rsplit("\n", 2)[-2:]
        say("    " + " · ".join(x.strip() for x in tail))
        ok = ok and (r == 0)

        head("자체 시험 " + ("전부 통과 — 진짜 데이터로 --preflight 해라" if ok
                          else "★ 실패 — 위를 보라"), "=")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# 본체
# ════════════════════════════════════════════════════════════════════════════
def plan(systems, rate, budget_h, ns, seeds, prep_s=0.0):
    """주어진 시간에 맞춰 규모를 정한다. **버린 것을 반드시 찍는다.**

    우선순위: 모든 구성체 × 양쪽 pH × 씨앗 1  →  그 다음 씨앗을 늘린다.
    양쪽 pH 는 절대 안 줄인다 — 차분이 신호 자체이므로 한쪽만 있으면 무의미하다.
    """
    if budget_h is None:
        return systems, ns, seeds, []
    per_ns_s = 1000.0 / DT_PS / max(rate, 1e-9)          # 1 ns 당 초
    budget_s = budget_h * 3600.0
    # ★ 준비(pdbfixer·수소·용매화·createSystem)는 (구성체 × pH) 당 한 번이고
    #   실제 규모에서 계당 2분이 넘는다. 이걸 빼먹으면 추정이 크게 낙관적이 된다.
    prep_total = len(systems) * len(PH_LIST) * prep_s
    for cand_seeds in range(seeds, 0, -1):
        for cand_ns in (ns, 20.0, 10.0, 5.0, 2.0, 1.0):
            if cand_ns > ns:
                continue
            need = prep_total + len(systems) * len(PH_LIST) * cand_seeds * cand_ns * per_ns_s
            if need <= budget_s:
                drop = []
                if cand_ns < ns:
                    drop.append(f"계당 {ns:g} ns → **{cand_ns:g} ns** 로 줄임")
                if cand_seeds < seeds:
                    drop.append(f"씨앗 {seeds} → **{cand_seeds}** 로 줄임 "
                                f"(씨앗이 1이면 잡음 분모가 없다)")
                return systems, cand_ns, cand_seeds, drop
    # 1 ns × 씨앗 1 로도 안 되면 구성체를 줄인다 — 항체는 최대한 남긴다
    per_run = len(PH_LIST) * (prep_s + 1.0 * per_ns_s)
    k = max(2, int(budget_s // max(per_run, 1e-9)))
    by_ab = {}
    for s in systems:
        by_ab.setdefault(s["항체"], []).append(s)
    keep, i = [], 0
    while len(keep) < k and any(len(v) > i for v in by_ab.values()):
        for v in by_ab.values():
            if len(v) > i and len(keep) < k:
                keep.append(v[i])
        i += 1
    dropped = [s for s in systems if s not in keep]
    return keep, 1.0, 1, [
        f"계당 1 ns · 씨앗 1 로도 예산을 넘겨 **구성체 {len(dropped)}개를 뺐다**",
        "  뺀 것: " + ", ".join(f"{s['항체']}/{s['링커']}" for s in dropped[:10])
        + (" …" if len(dropped) > 10 else ""),
        "  ★ 항체마다 최소 2개는 남기려 돌아가며 골랐다 — 항체내 대조가 살아야 한다"]


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--selftest", action="store_true", help="내장 소형계로 전 경로 검증")
    ap.add_argument("--preflight", action="store_true", help="배선 확인 + 시간 추정만")
    ap.add_argument("--budget-hours", type=float, default=None, help="이 시간에 맞춰 자동 규모")
    ap.add_argument("--ns", type=float, default=20.0, help="계당 프로덕션 ns (기본 20)")
    ap.add_argument("--seeds", type=int, default=2, help="씨앗 반복 수 (기본 2 — 잡음 분모)")
    ap.add_argument("--force-cpu", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    head("FvLinker · pH 3 풀림 파이프라인", "=")
    say(f"  pH {PH_LIST} · dt {DT_PS*1000:g} fs · {TEMP_K:g} K · 씨앗 {a.seeds}")
    plat, ffpair = env(a.force_cpu)
    import pandas as pd
    lab = read_excel()
    systems = find_structures(lab)
    os.makedirs(WORK, exist_ok=True)

    # ── 사전점검 ────────────────────────────────────────────────────────────
    head("[2] 사전점검 — 짧게 돌려 배선을 확인하고 속도를 잰다", "=")
    s0 = systems[0]
    say(f"  대상 {s0['항체']} / {s0['링커']} ({s0['출처']})")
    qs, rates, preps = {}, [], []
    for ph in PH_LIST:
        t0 = time.time()
        try:
            p = prepare(s0["pdb"], ph, ffpair)
        except Exception:
            say(f"\n  ★★ pH {ph} 준비 실패 — 전체 출력:")
            import traceback; traceback.print_exc()
            return 1
        preps.append(time.time() - t0)
        say(f"  pH {ph}: 원자 {p['atoms']:,} · 잔기 {p['nres']} · "
            f"이황화 **{p['ss']}개** · 단백질 순전하 **{p['q']:+.1f}**  "
            f"(준비 {time.time()-t0:.0f}s)")
        if p["ss"] == 0 and p["nres"] > 150:
            say("      ★★ 이황화가 0개다. scFv 라면 도메인당 하나씩 있어야 한다.")
            say("         입력 PDB 에 수소가 붙어 있거나 Cys 가 잘렸을 수 있다.")
        t1 = time.time()
        try:
            _, ps, nst = simulate(p, plat, 500, None, None, 10**9, seed=0,
                                  resume=False, min_iters=200, eq_ns=0.004)
        except Exception:
            say(f"\n  ★★ pH {ph} 시뮬레이션 실패 — 전체 출력:")
            import traceback; traceback.print_exc()
            return 1
        rates.append(nst / max(ps, 1e-9))
        qs[ph] = p["q"]
        say(f"      준비 외 최소화+평형 {time.time()-t1-ps:.0f}s · "
            f"프로덕션 {nst}스텝 {ps:.1f}s → **{nst/max(ps,1e-9):.0f} 스텝/초**")

    dq = qs[3.0] - qs[7.4]
    say(f"\n  ★ 양성자화 검산: 단백질 순전하 pH 7.4 → {qs[7.4]:+.1f} · "
        f"pH 3.0 → {qs[3.0]:+.1f}  (차이 **{dq:+.1f}**)")
    if dq < 5:
        say("    ★★ pH 3 에서 전하가 거의 안 올랐다 = Asp/Glu 가 중성화되지 않았다.")
        say("       이대로면 'pH 3 MD' 라 부르며 **pH 7 을 돌리게 된다.** 멈춘다.")
        say("       (Asp pKa 3.9 / Glu pKa 4.3 → pH 3.0 에서 거의 전부 중성이어야 한다)")
        return 1
    say("    ✔ Asp/Glu 가 실제로 중성화됐다 — 진짜 pH 3 조건이다.")

    rate = float(np.mean(rates))
    prep_s = float(np.mean(preps))
    systems, ns, seeds, dropped = plan(systems, rate, a.budget_hours, a.ns, a.seeds, prep_s)
    n_run = len(systems) * len(PH_LIST) * seeds
    per_run_s = ns * 1000 / DT_PS / max(rate, 1e-9)
    say(f"\n  속도 {rate:.0f} 스텝/초 = **{rate*DT_PS*86.4/1000:.0f} ns/일**")
    say(f"  계획: 구성체 {len(systems)} × pH {len(PH_LIST)} × 씨앗 {seeds} = **{n_run}건**")
    prep_total = len(systems) * len(PH_LIST) * prep_s
    say(f"        준비 {len(systems)*len(PH_LIST)}회 × {prep_s:.0f}s = {prep_total/60:.0f}분"
        f"  +  MD {n_run}건 × {per_run_s/60:.1f}분")
    say(f"        → **총 약 {(prep_total + per_run_s*n_run)/3600:.1f}시간**")
    for d in dropped:
        say(f"  ★ 줄였다: {d}")
    if seeds < 2:
        say("  ★★ 씨앗이 1이다 — MD 는 혼돈계라 궤적 하나는 증거가 아니다.")
        say("     시간이 되면 --seeds 2 이상으로 다시 돌려라.")
    if a.preflight:
        say("\n  --preflight 이므로 여기서 멈춘다.")
        return 0

    # ── 본 실행 ─────────────────────────────────────────────────────────────
    head("[3] 본 실행 — 끝나는 대로 저장. 끊기면 같은 명령으로 이어간다", "=")
    res_path = f"{WORK}/results.json"
    R = load_json(res_path)
    steps = int(ns * 1000 / DT_PS)
    # ★ 저장 간격이 총 스텝보다 크면 DCD 에 프레임이 0개가 되고 measure 가 죽는다.
    #   최소 10프레임은 나오게 묶는다.
    every = max(1, min(int(SAVE_PS / DT_PS), max(1, steps // 10)))
    if steps and every > steps:
        every = max(1, steps // 2)
    t_all = time.time()
    from openmm.app import PDBFile
    for i, s in enumerate(systems, 1):
        for ph in PH_LIST:
            # ★ 씨앗 루프를 **안쪽**에 둔다. 씨앗은 속도 초기값만 다르므로 준비된 계를
            #   그대로 쓴다. 예전처럼 씨앗마다 prepare() 를 부르면 준비 시간이 씨앗
            #   배수로 늘어난다 — 실제 규모(계당 ~2분)에서 76회면 준비만 2.7시간이다.
            todo = [sd for sd in range(seeds)
                    if f"{s['항체']}|{s['링커']}|{ph}|s{sd}" not in R]
            if not todo:
                say(f"  [{i}/{len(systems)}] {s['항체']}/{s['링커']} pH{ph}  "
                    f"건너뜀 (씨앗 {seeds}개 다 있다)")
                continue
            try:
                t_prep = time.time()
                p = prepare(s["pdb"], ph, ffpair)
                say(f"  [{i}/{len(systems)}] {s['항체']}/{s['링커']} pH{ph} · "
                    f"원자 {p['atoms']:,} · 이황화 {p['ss']} · 순전하 {p['q']:+.1f} "
                    f"(준비 {time.time()-t_prep:.0f}s)")
            except Exception as e:
                say(f"  [{i}/{len(systems)}] {s['항체']}/{s['링커']} pH{ph}  ★ 준비 실패: "
                    f"{type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                continue
            for sd in todo:
                key = f"{s['항체']}|{s['링커']}|{ph}|s{sd}"
                tag = re.sub(r"[^\w.-]", "_", key)
                top, dcd, chk = f"{WORK}/{tag}.pdb", f"{WORK}/{tag}.dcd", f"{WORK}/{tag}.chk"
                try:
                    if not os.path.isfile(top):
                        with open(top, "w") as fh:
                            PDBFile.writeFile(p["topology"], p["positions"], fh)
                    simulate(p, plat, steps, chk, dcd, every, seed=sd)
                    o = measure(top, dcd, s.get("링커서열", ""))
                    o.update(항체=s["항체"], 링커=s["링커"], pH=ph, 씨앗=sd,
                             원자=p["atoms"], 이황화=p["ss"], 순전하=p["q"],
                             HMW=s.get("HMW", float("nan")))
                    R[key] = o
                    save_json(R, res_path)
                    say(f"      s{sd} ✔ SAP_max {o['SAP_max']:+.2f} · "
                        f"패치 {o['최대연속패치']:.2f} nm² · 염다리 {o['염다리수']} · "
                        f"링커노출 {o['링커노출']:.2f}  ({(time.time()-t_all)/60:.0f}분 경과)")
                except Exception as e:
                    say(f"      s{sd} ★ 실패: {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()

    return analyze(R, systems)


# ════════════════════════════════════════════════════════════════════════════
# [4] 분석 — y 를 보기 **전에** 항체내몫부터
# ════════════════════════════════════════════════════════════════════════════
def analyze(R, systems):
    import pandas as pd
    import fvobs as F

    head("[4] 차분 · 잡음 · 항체내몫", "=")
    if not R:
        say("  결과가 없다."); return 0
    D = pd.DataFrame(list(R.values()))
    OBS = ["SAP_max", "SAP_sum", "최대연속패치", "소수성SASA", "총SASA",
           "염다리수", "Rg", "링커노출", "링커Re", "링커Rg", "평균노출"]
    OBS = [c for c in OBS if c in D.columns]

    # 씨앗 평균과 씨앗간 SD — ★ SD 가 분모다. 이게 없으면 차분을 못 읽는다
    g = D.groupby(["항체", "링커", "pH"])
    mean = g[OBS].mean().reset_index()
    sd = g[OBS].std(ddof=1).reset_index()
    nseed = int(D.groupby(["항체", "링커", "pH"]).size().max() or 1)

    piv = mean.pivot_table(index=["항체", "링커"], columns="pH", values=OBS)
    rows = []
    for (ab, lk), r in piv.iterrows():
        d = dict(항체=ab, 링커=lk)
        for c in OBS:
            try:
                d["Δ" + c] = float(r[(c, 3.0)] - r[(c, 7.4)])
            except Exception:
                d["Δ" + c] = float("nan")
        rows.append(d)
    T = pd.DataFrame(rows)
    DCOL = [c for c in T.columns if c.startswith("Δ")]

    say(f"  씨앗 {nseed}개 · 구성체 {len(T)}개 · 항체 {T.항체.nunique()}개\n")
    if nseed >= 2:
        say("  씨앗간 SD (차분을 이것과 견줘야 한다):")
        for c in OBS[:6]:
            v = sd[c].dropna()
            if len(v):
                say(f"    {c:12s} SD {v.mean():.3f}")
    else:
        say("  ★★ 씨앗이 1개다 — 잡음 분모가 없어 차분의 크기를 해석할 수 없다.")

    say("\n  ★ 항체내몫 — **y(HMW)를 보기 전에** 찍는다.")
    say("    0.3 미만이면 그 관측값은 링커가 아니라 **항체 정체**를 재고 있다.")
    ws = []
    for c in DCOL:
        w = F.within_share(T[c].values, T.항체.values)
        ws.append(dict(관측값=c, 항체내몫=round(w, 3) if np.isfinite(w) else np.nan,
                       판정="사용" if (np.isfinite(w) and w >= 0.3) else "★버림"))
    WS = pd.DataFrame(ws).sort_values("항체내몫", ascending=False)
    say(WS.to_string(index=False))
    live = [r.관측값 for r in WS.itertuples() if r.판정 == "사용"]

    say("\n  차분표:")
    say(T.round(3).to_string(index=False))
    T.to_csv(f"{WORK}/ph3_delta.csv", index=False, encoding="utf-8-sig")
    WS.to_csv(f"{WORK}/within_share.csv", index=False, encoding="utf-8-sig")

    if not live:
        head("판정", "=")
        say("  ★ 항체내몫이 0.3 을 넘는 관측값이 하나도 없다.")
        say("    전부 항체 정체를 재고 있다 — 링커 축이 아니다. HMW 와 대조하지 않는다.")
        say("    (여기서 멈추는 게 맞다. 대조하면 항체 효과를 링커 효과로 착각하게 된다.)")
        return 0

    # ── 이제서야 y 를 본다 ──────────────────────────────────────────────────
    head("[5] 이제 HMW 와 대조한다", "=")
    say("  ★ 사전등록: **주 관측값은 pH 3 에서의 `링커노출` 하나다.**")
    say("    이유 — 기전이 '링커가 끈끈한 면을 덮으면 덜 붙는다' 이고, 이 값만이")
    say("    설계상 항체 안에서 변한다 (같은 항체면 표면도 패치도 같고 링커만 다르다).")
    say("    응집이 일어나는 상태는 pH 3 이므로 차분이 아니라 **pH 3 절대값**을 쓴다.")
    say("    나머지는 전부 **탐색**이다 — p 값을 그대로 읽으면 안 된다.\n")
    hm = D.groupby(["항체", "링커"]).HMW.first().reset_index()
    T2 = T.merge(hm, on=["항체", "링커"], how="left").dropna(subset=["HMW"])
    if len(T2) < 4:
        say(f"  HMW 가 붙은 구성체가 {len(T2)}개뿐이다 — 검정을 못 한다.")
        return 0
    say(f"  검정 대상 {len(T2)}개 · 항체 {T2.항체.nunique()}개")
    say("  예측: 항체 **안에서** 노출이 큰 링커일수록 HMW 가 높다.\n")

    import itertools
    from collections import Counter
    from itertools import permutations

    def conc(x, h):
        c = 0.0
        for i, j in itertools.combinations(range(len(x)), 2):
            a, b = (i, j) if x[i] < x[j] else (j, i)
            c += 1.0 if h[b] > h[a] else (0.5 if h[b] == h[a] else 0.0)
        return c

    # 주 검정에 쓸 pH 3 절대값 표 (차분이 아니다)
    PRIM = "링커노출"
    prim = (D[D.pH == 3.0].groupby(["항체", "링커"])[PRIM].mean().reset_index()
            .rename(columns={PRIM: "주_링커노출_pH3"}))
    T2 = T2.merge(prim, on=["항체", "링커"], how="left")

    out = []
    for c in (["주_링커노출_pH3"] + [x for x in live if x != "주_링커노출_pH3"]):
        if c not in T2.columns:
            continue
        groups, obs = [], 0.0
        for ab, gg in T2.groupby("항체"):
            if len(gg) < 2 or not np.isfinite(gg[c]).all():
                continue
            x, h = list(gg[c].values), list(gg.HMW.values)
            obs += conc(x, h); groups.append((x, h))
        if not groups:
            continue
        tot = Counter({0.0: 1.0})
        for x, h in groups:
            dd = Counter(conc(x, list(p)) for p in permutations(h))
            s = sum(dd.values()); nt = Counter()
            for k0, v0 in tot.items():
                for k1, v1 in dd.items():
                    nt[k0 + k1] += v0 * v1 / s
            tot = nt
        k = np.array(sorted(tot)); pr = np.array([tot[v] for v in sorted(tot)])
        out.append(dict(관측값=c, 일치=obs, 최대=float(k.max()),
                        귀무=round(float((k * pr).sum()), 1),
                        p=round(float(pr[k >= obs].sum()), 4)))
    P = pd.DataFrame(out)
    if not len(P):
        say("  검정할 관측값이 없다."); return 0
    P["역할"] = ["**주(사전등록)**" if c == "주_링커노출_pH3" else "탐색"
                 for c in P.관측값]
    say(P.to_string(index=False))
    P.to_csv(f"{WORK}/rank_test.csv", index=False, encoding="utf-8-sig")

    head("판정", "=")
    pr = P[P.관측값 == "주_링커노출_pH3"]
    if len(pr):
        r0 = pr.iloc[0]
        ok = r0.p < 0.05
        say(f"  주 검정 (사전등록): 링커노출@pH3 → HMW 순위 "
            f"{r0.일치:.1f}/{r0.최대:.0f} (귀무 {r0.귀무}), **정확 순열 p = {r0.p:.4f}**")
        say("  " + ("✔ 기전 예측이 맞았다. 이건 다중비교가 없는 단일 검정이다."
                    if ok else
                    "✗ 기전 예측이 안 맞았다. 이게 결론이고, 아래 탐색으로 뒤집으면 안 된다."))
    else:
        say("  ★ 주 관측값(링커노출@pH3)을 계산하지 못했다 — 링커 구간을 못 찾았을 것이다.")
        say("    엑셀의 링커 서열이 구조 서열 안에 **유일하게** 들어 있어야 한다.")
    ex = P[P.관측값 != "주_링커노출_pH3"]
    if len(ex):
        b = ex.sort_values("p").iloc[0]
        say(f"\n  탐색 {len(ex)}개 중 최소: {b.관측값} p = {b.p:.4f} "
            f"→ Bonferroni {min(1.0, b.p*len(ex)):.4f}")
        say("    ★ 탐색은 가설 생성용이다. 이걸 결과로 보고하면 낚시다 —")
        say("      다음 라운드에 **사전등록**해서 새 데이터로 확인해야 한다.")
    say(f"\n  저장: {WORK}/ph3_delta.csv · within_share.csv · rank_test.csv · results.json")
    head("끝", "=")
    return 0


if __name__ == "__main__":
    sys.exit(main())
