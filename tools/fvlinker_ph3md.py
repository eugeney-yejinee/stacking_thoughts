#!/usr/bin/env python3
"""FvLinker · pH 3 풀림 파이프라인 — **밤새 돌려놓고 자는 용도**.

────────────────────────────────────────────────────────────────────────────
무엇을 하는가
────────────────────────────────────────────────────────────────────────────
  링커만 다른 scFv 구조  →  pH 7.4 와 pH 3.0 에서 각각 MD
                         →  두 궤적의 **차분**에서 무엇이 드러나는지 잰다
                         →  드러난 끈끈한 면의 크기로 HMW 높음/낮음을 **순위**로 예측

왜 차분인가:
  100 ns 로는 도메인이 안 풀린다 (실제 언폴딩은 ms 이상). "풀렸나" 를 보면
  아무 일도 안 일어난 것처럼 보이고 잘못된 결론이 난다. 재야 하는 것은
  **pH 7 대비 pH 3 에서 무엇이 달라졌나** 다 — 끊어진 염다리, 풀린 계면
  가장자리, 링커가 옮겨간 자리, 그래서 새로 드러난 소수성 면.

왜 두 분자를 안 넣는가:
  Smoluchowski 로 쪼개면 k_assoc = k_encounter × P_stick 인데, 만남 빈도는
  링커가 바뀌어도 거의 안 변한다 (k∝R·D, D∝1/R). 변하는 건 **붙을 확률**이고
  그건 드러난 끈끈한 면으로 정해진다 — **한 분자 계산**이다. 전원자 회합 PMF 는
  쌍당 약 44 GPU-일이라 애초에 불가능하다.

────────────────────────────────────────────────────────────────────────────
밤새 돌리기 위한 장치
────────────────────────────────────────────────────────────────────────────
  · **사전점검 먼저** — 짧게 한 번 돌려 배선을 확인하고 **시간을 추정해서 찍는다.**
    이 프로젝트는 한 줄짜리 설정 버그로 실행을 세 번 날렸다. 다시는 안 그런다.
  · **체크포인트 + 재개** — Colab 은 12~24시간에 끊긴다. 다시 실행하면 이어서 간다.
  · **계마다 끝나는 즉시 저장** — 중간에 죽어도 거기까지는 건진다.
  · **조용히 틀리지 않는다** — 이황화 개수, pH별 순전하, 플랫폼을 매번 찍고 검산한다.

────────────────────────────────────────────────────────────────────────────
설치 (Colab)
────────────────────────────────────────────────────────────────────────────
    pip install pdb2pqr propka pdbfixer mdtraj
    pip install "openmm[cuda12]"      # ★ GPU 박스에서만. conda 필요 없다. CUDA 12 전용

실행:
    python3 fvlinker_ph3md.py                 # 전부 자동
    python3 fvlinker_ph3md.py --preflight     # 사전점검만 하고 멈춘다
    python3 fvlinker_ph3md.py --ns 50         # 계당 프로덕션 길이 (기본 20 ns)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np

# ── 설정 ────────────────────────────────────────────────────────────────────
DRIVE = os.environ.get("FVL_DRIVE", "/content/drive/MyDrive/FvTwist")
IN = f"{DRIVE}/input"
OUT = os.environ.get("FVL_OUT", f"{DRIVE}/fvflow")   # ★ build_v12.py 와 같은 경로.
                                                     #   이름이 FvLinker 여도 안 바꾼다
WORK = f"{OUT}/ph3md"

PH_LIST = [7.4, 3.0]               # 이 둘의 **차분**이 신호다
PROD_NS = 20.0                     # 계당 프로덕션 (--ns 로 바꾼다)
DT_PS = 0.004                      # 4 fs (수소질량 재분배)
SAVE_PS = 20.0                     # 프레임 저장 간격
EQ_NS = 0.2                        # 평형화
PAD_NM = 1.0
IONIC_M = 0.05
TEMP_K = 300.0
PREFLIGHT_STEPS = 500
MAX_SYSTEMS = 0                    # 0 = 제한 없음

W = 78
def head(t, c="="):
    print("\n" + c * W); print(t); print(c * W, flush=True)
def say(*a):
    print(*a, flush=True)


# ── [0] 환경 ────────────────────────────────────────────────────────────────
def env():
    head("[0] 환경", "─")
    mods = {}
    for m in ("openmm", "pdbfixer", "mdtraj"):
        try:
            mods[m] = __import__(m)
            v = getattr(mods[m], "__version__", None) or \
                getattr(getattr(mods[m], "version", None), "version", "?")
            say(f"  ✔ {m} {v}")
        except Exception as e:
            say(f"  ✗ {m} 없음 ({type(e).__name__})")
            mods[m] = None
    missing = [m for m, v in mods.items() if v is None]
    if missing:
        say(f"\n  ★ 설치가 필요하다:  pip install {' '.join(missing)}")
        say("     GPU 를 쓰려면:    pip install \"openmm[cuda12]\"")
        raise SystemExit(1)

    import openmm as om
    names = []
    for n in ("CUDA", "HIP", "OpenCL", "CPU", "Reference"):
        try:
            om.Platform.getPlatformByName(n); names.append(n)
        except Exception:
            pass
    plat = names[0] if names else "Reference"
    say(f"  등록 플랫폼 {names} → **{plat}**")
    if plat in ("CPU", "Reference"):
        say("  ★★ GPU 가 안 잡혔다. CPU 로는 계당 며칠 걸린다.")
        say("     Colab 런타임을 GPU 로 바꾸고  pip install \"openmm[cuda12]\"  해라.")
        say("     (그래도 계속 돌리려면 --force-cpu 를 줘라)")
        if "--force-cpu" not in sys.argv:
            raise SystemExit(1)

    # 힘장 — ★ 이 짝이 맞아야 한다. charmm36_2024.xml 은 charmm36/water.xml 과 안 맞는다
    from openmm.app import ForceField
    for pair in (("charmm36_2024.xml", "charmm36_2024/water.xml"),
                 ("charmm36.xml", "charmm36/water.xml")):
        try:
            ForceField(*pair); say(f"  힘장 **{pair[0]}** + {pair[1]}"); return plat, pair
        except Exception as e:
            say(f"  ✗ {pair[0]} + {pair[1]} → {type(e).__name__}")
    raise SystemExit("★ 쓸 수 있는 CHARMM 힘장이 없다")


# ── [1] 구조 찾기 ───────────────────────────────────────────────────────────
def find_structures():
    """Drive 에서 (항체, 링커) 별 시작 구조를 찾는다.

    1순위: {OUT}/{항체}/flow/{링커}__BioEmu_s*.pdb  (구성체별로 있다 — 링커가 다르다)
    2순위: {OUT}/{항체}/ref*.pdb                    (ABodyBuilder2 기준 구조)
    BioEmu 는 **뼈대만** 준다 → pdbfixer 가 곁사슬을 다시 붙인다.
    """
    head("[1] 시작 구조", "─")
    found = []
    for abdir in sorted(glob.glob(f"{OUT}/*/")):
        ab = os.path.basename(abdir.rstrip("/"))
        if ab in ("ph3md", "calvados", "input"):
            continue
        bes = sorted(glob.glob(f"{abdir}flow/*__BioEmu_s*.pdb"))
        by = {}
        for p in bes:
            lk = os.path.basename(p).split("__BioEmu")[0]
            by.setdefault(lk, []).append(p)
        for lk, ps in sorted(by.items()):
            found.append(dict(항체=ab, 링커=lk, pdb=ps[0], 출처="BioEmu", 뼈대만=True))
        if not by:
            refs = sorted(glob.glob(f"{abdir}ref*.pdb"))
            if refs:
                found.append(dict(항체=ab, 링커="ref", pdb=refs[0],
                                  출처="ABB2", 뼈대만=False))
    if not found:
        say(f"  ★ {OUT} 아래에서 PDB 를 하나도 못 찾았다.")
        say("    찾아본 곳: {OUT}/<항체>/flow/<링커>__BioEmu_s*.pdb 와 {OUT}/<항체>/ref*.pdb")
        say("    경로가 다르면 이 파일 맨 위의 OUT 을 고쳐라.")
        raise SystemExit(1)
    say(f"  구성체 {len(found)}개 · 항체 {len({f['항체'] for f in found})}개")
    for f in found[:8]:
        say(f"    {f['항체']:22s} {f['링커']:12s} {f['출처']:7s} "
            f"{os.path.basename(f['pdb'])}")
    if len(found) > 8:
        say(f"    … 외 {len(found)-8}개")
    if MAX_SYSTEMS:
        found = found[:MAX_SYSTEMS]
    return found


# ── [2] 준비 — 여기가 조용히 틀리는 자리다 ──────────────────────────────────
def prepare(pdb_in, ph, ffpair, out_pdb):
    """수소 제거 → pH별 양성자화 → 용매화. 검산값을 함께 돌려준다.

    ★ 수소를 **먼저 전부 지운다.** OpenMM 의 createDisulfideBonds 는 HG 가 없는
      Cys 만 이황화 후보로 본다. 수소 붙은 PDB 를 넣으면 scFv 이황화가 **오류 없이
      전부 사라진다.** scFv 는 도메인당 이황화가 있으니 그 MD 는 통째로 쓰레기가 된다.

    ★ Modeller.addHydrogens(pH=3.0) 은 Asp/Glu 를 실제로 중성화한다 (hydrogens.xml
      의 maxph=4.4). His 만 건드린다는 통설은 틀렸다. 다만 pH ≤ 4.4 **전역 계단함수**라
      환경 민감도가 없다 — pH 3.0 에서는 거의 모든 Asp/Glu pKa 아래라 맞고,
      pH 4~5 였다면 틀렸을 것이다.
    """
    from openmm import unit
    from openmm.app import PDBFile, ForceField, Modeller, element
    from pdbfixer import PDBFixer

    fx = PDBFixer(filename=pdb_in)
    fx.findMissingResidues()
    fx.findMissingAtoms()
    fx.addMissingAtoms()                       # BioEmu 뼈대 → 곁사슬 복원
    fx.removeHeterogens(keepWater=False)

    ff = ForceField(*ffpair)
    m = Modeller(fx.topology, fx.positions)
    m.delete([a for a in m.topology.atoms() if a.element == element.hydrogen])  # ★ 필수
    m.addHydrogens(ff, pH=float(ph))

    ss = sum(1 for b in m.topology.bonds()
             if b[0].name == "SG" and b[1].name == "SG")
    nres = m.topology.getNumResidues()

    m.addSolvent(ff, model="tip3p", padding=PAD_NM * unit.nanometer,
                 neutralize=True, ionicStrength=IONIC_M * unit.molar)
    sysm = ff.createSystem(m.topology, nonbondedMethod=__import__(
        "openmm.app", fromlist=["PME"]).PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=__import__("openmm.app", fromlist=["HBonds"]).HBonds,
        rigidWater=True, hydrogenMass=1.5 * unit.amu)
    # 단백질 순전하 검산 — pH 3 에서 강하게 양수여야 한다
    nb = next(f for f in sysm.getForces() if f.__class__.__name__ == "NonbondedForce")
    prot = {a.index for a in m.topology.atoms()
            if a.residue.name not in ("HOH", "NA", "CL", "WAT")}
    q = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in prot)
    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(m.topology, m.positions, fh)
    return dict(topology=m.topology, positions=m.positions, system=sysm,
                atoms=sysm.getNumParticles(), ss=ss, q=round(q, 1), nres=nres)


def simulate(prep, plat, steps, chk, dcd, report_every, resume=True,
             min_iters=2000, eq_ns=None):
    """돌리고 **프로덕션 구간만의 초 수**를 같이 돌려준다.

    ★ 시간 추정은 프로덕션 스텝만으로 해야 한다. 최소화와 평형은 계당 한 번뿐인데
      CPU 에서는 그게 전체의 대부분이라, 같이 재면 추정이 몇 배 부풀려진다.
    """
    from openmm import unit, LangevinMiddleIntegrator, MonteCarloBarostat, Platform
    from openmm.app import Simulation, DCDReporter, CheckpointReporter

    eq = EQ_NS if eq_ns is None else eq_ns
    sysm = prep["system"]
    sysm.addForce(MonteCarloBarostat(1 * unit.bar, TEMP_K * unit.kelvin, 25))
    integ = LangevinMiddleIntegrator(TEMP_K * unit.kelvin, 1 / unit.picosecond,
                                     DT_PS * unit.picoseconds)
    sim = Simulation(prep["topology"], sysm, integ,
                     Platform.getPlatformByName(plat))
    done = 0
    if resume and chk and os.path.isfile(chk):
        try:
            sim.loadCheckpoint(chk)
            done = int(sim.context.getStepCount())
            say(f"      체크포인트에서 이어간다 ({done:,} 스텝 완료)")
        except Exception as e:
            say(f"      ★ 체크포인트를 못 읽었다 ({type(e).__name__}) — 처음부터")
            done = 0
    if done == 0:
        sim.context.setPositions(prep["positions"])
        sim.minimizeEnergy(maxIterations=min_iters)
        sim.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin)
        if eq > 0:
            sim.step(int(eq * 1000 / DT_PS))
    left = max(0, steps - done)
    t0 = time.time()
    if left:
        if dcd:
            sim.reporters.append(DCDReporter(dcd, report_every, append=os.path.isfile(dcd)))
        if chk:
            sim.reporters.append(CheckpointReporter(chk, max(1000, report_every)))
        sim.step(left)
    return sim, (time.time() - t0), left


# ── [5] 관측 — pH 7 대비 pH 3 의 차분 ───────────────────────────────────────
def observe(top_pdb, dcd, linker_span=None):
    """궤적 → 노출 면적. 소수성 SASA 가 응집 핵의 크기를 대신한다."""
    import mdtraj as md
    t = md.load(dcd, top=top_pdb)
    t = t.atom_slice(t.topology.select("protein"))
    sasa = md.shrake_rupley(t, mode="residue")          # (frame, residue) nm²
    HPHO = set("ALA VAL LEU ILE PHE MET TRP PRO TYR CYS".split())
    res = list(t.topology.residues)
    hyd = np.array([r.name in HPHO for r in res])
    per = sasa.mean(axis=0)
    out = dict(
        총SASA=float(per.sum()),
        소수성SASA=float(per[hyd].sum()),
        소수성비=float(per[hyd].sum() / max(per.sum(), 1e-9)),
        최대노출소수성=float(per[hyd].max()) if hyd.any() else float("nan"),
        Rg=float(md.compute_rg(t).mean()),
        n프레임=int(t.n_frames), n잔기=int(len(res)),
    )
    out["잔기별"] = [round(float(v), 4) for v in per]
    return out


# ── 본체 ────────────────────────────────────────────────────────────────────
def main(argv):
    global PROD_NS
    if "--ns" in argv:
        PROD_NS = float(argv[argv.index("--ns") + 1])
    only_pre = "--preflight" in argv

    head("FvLinker · pH 3 풀림 파이프라인", "=")
    say(f"  pH {PH_LIST} · 계당 {PROD_NS:g} ns · dt {DT_PS*1000:g} fs · {TEMP_K:g} K")
    plat, ffpair = env()
    systems = find_structures()
    os.makedirs(WORK, exist_ok=True)

    # ── [3] 사전점검 ────────────────────────────────────────────────────────
    head("[3] 사전점검 — 짧게 한 번 돌려 배선을 확인한다", "=")
    s0 = systems[0]
    say(f"  대상 {s0['항체']} / {s0['링커']} ({s0['출처']})")
    est, qs = {}, {}
    for ph in PH_LIST:
        t0 = time.time()
        try:
            p = prepare(s0["pdb"], ph, ffpair, f"{WORK}/_pre_ph{ph}.pdb")
        except Exception as e:
            say(f"\n  ★★ pH {ph} 준비 실패 — 전체 출력:")
            import traceback; traceback.print_exc()
            say("\n  본 실행은 안 한다. 위를 보고 고쳐라.")
            return 1
        say(f"  pH {ph}:  원자 {p['atoms']:,} · 잔기 {p['nres']} · "
            f"이황화 **{p['ss']}개** · 단백질 순전하 **{p['q']:+.1f}**")
        if p["ss"] == 0:
            say("      ★★ 이황화가 0개다. scFv 라면 있어야 한다 — 입력 PDB 를 확인해라.")
        t1 = time.time()
        try:
            # 사전점검은 최소화·평형을 짧게 한다. 배선 확인이 목적이지 물리가 아니다.
            _, prod_s, nst = simulate(p, plat, PREFLIGHT_STEPS, None, None, 10 ** 9,
                                      resume=False, min_iters=200, eq_ns=0.004)
        except Exception:
            say(f"\n  ★★ pH {ph} 시뮬레이션 실패 — 전체 출력:")
            import traceback; traceback.print_exc()
            return 1
        est[ph] = (prod_s, nst)
        say(f"      준비 {t1-t0:.0f}s · 최소화+평형 {time.time()-t1-prod_s:.0f}s · "
            f"프로덕션 {nst}스텝 **{prod_s:.1f}s**  ({nst/max(prod_s,1e-9):.0f} 스텝/초)")
        qs[ph] = p["q"]

    if len(qs) == 2:
        q7, q3 = qs[7.4], qs[3.0]
        say(f"\n  ★ 양성자화 검산: 단백질 순전하 pH 7.4 → {q7:+.1f} · pH 3.0 → {q3:+.1f}"
            f"  (차이 {q3-q7:+.1f})")
        if q3 - q7 < 5:
            say("    ★★ pH 3 에서 전하가 거의 안 올랐다. Asp/Glu 가 중성화되지 않았다는 뜻이다.")
            say("       이대로 돌리면 'pH 3 MD' 라고 부르면서 **pH 7 을 돌리게 된다.** 멈춘다.")
            say("       (Asp pKa 3.9 / Glu pKa 4.3 → pH 3.0 에서 거의 전부 중성이어야 한다)")
            return 1
        say("    ✔ Asp/Glu 가 실제로 중성화됐다 — 진짜 pH 3 조건이다.")

    prod_steps = int(PROD_NS * 1000 / DT_PS)
    rate = np.mean([n / max(s, 1e-9) for s, n in est.values()])     # 스텝/초
    per = prod_steps / max(rate, 1e-9)
    n_run = len(systems) * len(PH_LIST)
    say(f"\n  프로덕션 속도 {rate:.0f} 스텝/초 = {rate*DT_PS*86.4/1000:.1f} ns/일")
    say(f"  추정: 계당 약 **{per/60:.0f}분** × {n_run}건 = "
        f"**약 {per*n_run/3600:.1f}시간** (준비 시간은 별도)")
    if per * n_run / 3600 > 14:
        say("  ★ 14시간이 넘는다. Colab 은 도중에 끊긴다 — 체크포인트가 있으니")
        say("    아침에 같은 명령을 다시 실행하면 이어서 간다. 또는 --ns 를 줄여라.")
    if only_pre:
        say("\n  --preflight 라 여기서 멈춘다.")
        return 0

    # ── [4] 본 실행 ─────────────────────────────────────────────────────────
    head("[4] 본 실행 — 끝나는 대로 저장한다. 끊기면 다시 실행하면 이어간다", "=")
    res_path = f"{WORK}/results.json"
    R = json.load(open(res_path)) if os.path.isfile(res_path) else {}
    every = int(SAVE_PS / DT_PS)
    t_all = time.time()
    for i, s in enumerate(systems, 1):
        for ph in PH_LIST:
            key = f"{s['항체']}|{s['링커']}|{ph}"
            if key in R:
                say(f"  [{i}/{len(systems)}] {key}  건너뜀 (이미 있다)")
                continue
            tag = re.sub(r"[^\w.-]", "_", key)
            top = f"{WORK}/{tag}.pdb"; dcd = f"{WORK}/{tag}.dcd"; chk = f"{WORK}/{tag}.chk"
            say(f"  [{i}/{len(systems)}] {key} …")
            try:
                p = prepare(s["pdb"], ph, ffpair, top)
                simulate(p, plat, prod_steps, chk, dcd, every)[0]
                o = observe(top, dcd)
                o.update(항체=s["항체"], 링커=s["링커"], pH=ph,
                         원자=p["atoms"], 이황화=p["ss"], 순전하=p["q"], 출처=s["출처"])
                R[key] = o
                json.dump(R, open(res_path, "w"), ensure_ascii=False, indent=1)
                say(f"      ✔ 소수성SASA {o['소수성SASA']:.1f} nm² · Rg {o['Rg']:.2f} nm "
                    f"· 프레임 {o['n프레임']}  ({(time.time()-t_all)/60:.0f}분 경과)")
            except Exception as e:
                say(f"      ★ 실패: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

    # ── [5] 차분과 순위 ─────────────────────────────────────────────────────
    head("[5] pH 7.4 → 3.0 차분 · 그리고 HMW 순위 예측", "=")
    import pandas as pd
    rows = []
    for s in systems:
        a, b = R.get(f"{s['항체']}|{s['링커']}|7.4"), R.get(f"{s['항체']}|{s['링커']}|3.0")
        if not (a and b):
            continue
        rows.append(dict(항체=s["항체"], 링커=s["링커"],
                         소수성SASA_pH7=round(a["소수성SASA"], 2),
                         소수성SASA_pH3=round(b["소수성SASA"], 2),
                         Δ소수성노출=round(b["소수성SASA"] - a["소수성SASA"], 2),
                         ΔRg=round(b["Rg"] - a["Rg"], 3),
                         Δ최대패치=round(b["최대노출소수성"] - a["최대노출소수성"], 3)))
    if not rows:
        say("  아직 pH 짝이 완성된 구성체가 없다. 다시 실행하면 이어간다.")
        return 0
    T = pd.DataFrame(rows).sort_values(["항체", "Δ소수성노출"], ascending=[True, False])
    say(T.to_string(index=False))
    T.to_csv(f"{WORK}/ph3_delta.csv", index=False, encoding="utf-8-sig")

    say("\n  ★ 예측: 항체 **안에서** Δ소수성노출이 큰 링커일수록 HMW 가 높아야 한다.")
    say("    (붙을 확률 ∝ 드러난 끈끈한 면. 만남 빈도는 링커와 거의 무관하다.)")
    for ab, g in T.groupby("항체"):
        if len(g) < 2:
            continue
        o = g.sort_values("Δ소수성노출", ascending=False)
        say(f"    {ab}:  HMW 높을 것 →  " + "  >  ".join(o.링커))
    say(f"\n  저장: {WORK}/ph3_delta.csv · {res_path}")
    say("  실측 HMW 와의 순위 일치는 fvlinker_model.py 의 정확 순열검정으로 잰다.")
    head("끝", "=")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
