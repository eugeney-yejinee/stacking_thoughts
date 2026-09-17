"""fvobs 자체 시험 — **답을 아는 경우**로만 검산한다.

합성 위상을 손으로 지어서 기대값을 미리 계산해 두고 맞춰 본다.
실제 단백질을 안 쓰는 이유: 실제 구조로는 "그럴듯한 값"밖에 못 보고,
그럴듯한 값은 조용히 틀린 코드를 통과시킨다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

OK, BAD = [], []


def check(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(f"  {'OK ' if cond else '★NG'} {name}" + (f"  {extra}" if extra else ""))


try:
    import mdtraj as md
except Exception as e:                                    # pragma: no cover
    print(f"★ mdtraj 가 없다 ({e}) — pip install mdtraj");  sys.exit(1)

from fvobs import (AA3to1, BLACK_MOULD, HYD_CUT, MAXSASA, acid_is_charged, base_is_charged,
                   block_drift, chain_dims, find_span, interface_contacts,
                   largest_patch, linker_shield, rel_exposure, salt_bridges, sap,
                   traj_sequence, within_share)


# ── 합성 궤적 만들기 ────────────────────────────────────────────────────────
def build(res_specs, frames=1, jitter=0.0, seed=0):
    """res_specs = [(잔기이름, {원자이름: (x,y,z) nm}), ...] → mdtraj Trajectory."""
    from mdtraj.core import element as E
    top = md.Topology()
    ch = top.add_chain()
    xyz = []
    for nm, atoms in res_specs:
        r = top.add_residue(nm, ch)
        for an, pos in atoms.items():
            sym = an[0]
            el = {"C": E.carbon, "N": E.nitrogen, "O": E.oxygen,
                  "S": E.sulfur, "H": E.hydrogen}.get(sym, E.carbon)
            top.add_atom(an, el, r)
            xyz.append(pos)
    base = np.array(xyz, float)
    rng = np.random.default_rng(seed)
    X = np.stack([base + (rng.normal(0, jitter, base.shape) if jitter else 0.0)
                  for _ in range(frames)])
    return md.Trajectory(X, top)


def bb(x, y=0.0, z=0.0, ca_only=False):
    """한 잔기의 최소 원자 묶음. CA 와 CB 를 같은 자리 근처에 둔다."""
    d = {"N": (x - 0.12, y, z), "CA": (x, y, z), "C": (x + 0.12, y, z),
         "O": (x + 0.15, y + 0.1, z)}
    if not ca_only:
        d["CB"] = (x, y + 0.15, z)
    return d


# ── 1. 표 자체 ──────────────────────────────────────────────────────────────
print("\n[1] 상수표")
check("20종 최대면적이 다 있다", all(a in MAXSASA for a in AA3to1.values() if a != "X")
      or len([k for k in MAXSASA if len(k) == 3]) >= 20, f"{len(MAXSASA)}종")
check("소수성 척도도 같은 집합", set(BLACK_MOULD) >= {"ALA", "PHE", "ASP", "LYS"})
check("PHE 가 ASP 보다 소수성", BLACK_MOULD["PHE"] > BLACK_MOULD["ASP"])
check("양성자화 변종이 원래 잔기로 매핑",
      MAXSASA["ASH"] == MAXSASA["ASP"] and BLACK_MOULD["GLH"] == BLACK_MOULD["GLU"]
      and AA3to1["HIP"] == "H")

# ── 2. 서열 ────────────────────────────────────────────────────────────────
print("\n[2] 서열과 구간 찾기")
t = build([("ALA", bb(0.0)), ("GLY", bb(0.5)), ("PHE", bb(1.0)), ("ASP", bb(1.5))])
check("서열을 읽는다", traj_sequence(t) == "AGFD", traj_sequence(t))
check("양성자화 이름도 원래대로",
      traj_sequence(build([("ASH", bb(0)), ("HIP", bb(.5))])) == "DH")
check("구간을 찾는다", find_span("AAAGGGGSCCC", "GGGGS") == (3, 8))
check("없으면 None", find_span("AAACCC", "GGGGS") is None)
check("여러 번 나오면 None (모호)", find_span("GGGGSXGGGGS", "GGGGS") is None)
check("빈 입력도 None", find_span("", "A") is None and find_span("AAA", "") is None)

# ── 3. 노출도 ──────────────────────────────────────────────────────────────
print("\n[3] 상대 노출도 — 뭉쳐 있으면 덜 노출된다")
spread = build([("LEU", bb(i * 2.0)) for i in range(5)])          # 2 nm 간격, 다 떨어짐
packed = build([("LEU", bb(i * 0.35)) for i in range(5)])         # 0.35 nm 간격, 뭉침
rs, rp = rel_exposure(spread), rel_exposure(packed)
check("떨어진 쪽이 더 노출", np.nanmean(rs) > np.nanmean(rp),
      f"떨어짐 {np.nanmean(rs):.3f} vs 뭉침 {np.nanmean(rp):.3f}")
check("0~1.5 범위", np.all((rs[np.isfinite(rs)] >= 0) & (rs[np.isfinite(rs)] <= 1.5)))
check("모르는 잔기는 NaN",
      not np.isfinite(rel_exposure(build([("UNK", bb(0.0))]))[0]))
try:
    rel_exposure(md.Trajectory(np.zeros((0, 1, 3)), spread.topology))
    check("빈 궤적은 거부", False, "예외가 안 났다")
except (ValueError, IndexError):
    check("빈 궤적은 거부", True)

# ── 4. SAP ─────────────────────────────────────────────────────────────────
print("\n[4] SAP — 노출된 소수성만 양수여야 한다")
hyd = build([("PHE", bb(i * 2.0)) for i in range(4)])
phil = build([("ASP", bb(i * 2.0)) for i in range(4)])
sh, sp_ = sap(hyd), sap(phil)
check("노출된 PHE 는 SAP 양수", np.nanmean(sh) > 0, f"{np.nanmean(sh):+.3f}")
check("노출된 ASP 는 SAP 음수", np.nanmean(sp_) < 0, f"{np.nanmean(sp_):+.3f}")
check("길이가 맞는다", len(sh) == 4)
# 반경 안의 이웃만 더한다: 멀리 떨어뜨리면 자기 자신만 남아 값이 작아진다
near = build([("PHE", bb(0.0)), ("PHE", bb(0.3))])
far = build([("PHE", bb(0.0)), ("PHE", bb(5.0))])
check("이웃이 가까우면 SAP 가 커진다", np.nanmax(sap(near)) > np.nanmax(sap(far)),
      f"가까움 {np.nanmax(sap(near)):.3f} vs 멂 {np.nanmax(sap(far)):.3f}")

# ── 5. 최대 연속 패치 ──────────────────────────────────────────────────────
print("\n[5] 최대 연속 패치 — 총량이 같아도 흩어지면 작아야 한다")
#  뭉친 3개(0.5 nm 간격) + 멀리 떨어진 2개
clust = build([("LEU", bb(0.0)), ("LEU", bb(0.5)), ("LEU", bb(1.0)),
               ("LEU", bb(9.0)), ("LEU", bb(12.0))])
a_c, mem_c = largest_patch(clust, link_nm=0.8)
check("연결 성분이 3개다", len(mem_c) == 3, f"{mem_c}")
#  전부 흩어뜨리면 성분 크기가 1
scat = build([("LEU", bb(i * 5.0)) for i in range(5)])
a_s, mem_s = largest_patch(scat, link_nm=0.8)
check("흩어지면 성분이 1개", len(mem_s) == 1, f"{mem_s}")
check("뭉친 쪽 면적이 더 크다", a_c > a_s, f"{a_c:.2f} vs {a_s:.2f} nm²")
check("친수성만 있으면 패치 없음",
      largest_patch(build([("ASP", bb(i * 0.5)) for i in range(4)]))[0] == 0.0)
# ★★ GLY 가 소수성으로 새면 (G4S)n 링커가 곧 '최대 소수성 패치' 가 되고,
#    링커가 자기를 덮으니 linker_shield 가 0.0 이 된다 — 주 관측값이 통째로 죽는다.
g4s = build([("GLY", bb(0.0)), ("GLY", bb(0.5)), ("GLY", bb(1.0)), ("SER", bb(1.5))])
check("★ GLY 는 소수성이 아니다 (G4S 링커가 패치가 되면 안 된다)",
      largest_patch(g4s)[0] == 0.0,
      f"패치 {largest_patch(g4s)[0]:.2f} nm² · 구성원 {largest_patch(g4s)[1]}")
check("Black-Mould 에서 GLY 는 문턱 아래", (BLACK_MOULD["GLY"] - 0.5) <= HYD_CUT,
      f"{BLACK_MOULD['GLY']-0.5:+.4f} ≤ {HYD_CUT}")
check("LEU/PHE 는 문턱 위",
      (BLACK_MOULD["LEU"] - 0.5) > HYD_CUT and (BLACK_MOULD["PHE"] - 0.5) > HYD_CUT)
# ★ 성분을 개수가 아니라 **면적**으로 골라야 한다.
#   넉넉히 떨어진 2개(각자 1.74 nm² → 합 3.47)  vs  빽빽한 3개(합 3.20).
#   개수로 고르면 3개짜리를, 면적으로 고르면 2개짜리를 집는다.
mix = build([("LEU", bb(0.0)), ("LEU", bb(0.75)),
             ("LEU", bb(9.0)), ("LEU", bb(9.30)), ("LEU", bb(9.60))])
a_m, mem_m = largest_patch(mix, link_nm=0.8)
check("★ 개수가 아니라 면적이 큰 성분을 고른다", mem_m == [0, 1],
      f"고른 것 {mem_m} · 면적 {a_m:.2f} nm² (개수로 고르면 [2,3,4] 가 된다)")

# ── 6. 링커 가림 ───────────────────────────────────────────────────────────
print("\n[6] 링커 가림 — 손으로 셀 수 있는 배치")
#  패치 4개를 2 nm 간격으로, 링커 2개를 그중 앞 2개 바로 옆(0.3 nm)에
sp = [("LEU", bb(0.0)), ("LEU", bb(2.0)), ("LEU", bb(4.0)), ("LEU", bb(6.0)),
      ("GLY", bb(0.3)), ("GLY", bb(2.3))]
t6 = build(sp)
check("절반만 드러남", abs(linker_shield(t6, [0, 1, 2, 3], [4, 5], 0.8) - 0.5) < 1e-9,
      f"{linker_shield(t6, [0,1,2,3], [4,5], 0.8):.2f}")
check("링커가 없으면 전부 드러남",
      linker_shield(t6, [0, 1, 2, 3], [], 0.8) == 1.0)
far6 = build([("LEU", bb(0.0)), ("LEU", bb(2.0)), ("GLY", bb(50.0))])
check("링커가 멀면 전부 드러남", linker_shield(far6, [0, 1], [2], 0.8) == 1.0)
try:
    linker_shield(t6, [], [4]); check("빈 패치는 거부", False, "예외가 안 났다")
except ValueError:
    check("빈 패치는 거부", True)

# ── 7. 염다리 ──────────────────────────────────────────────────────────────
print("\n[7] 염다리")
close = build([("ASP", {**bb(0.0), "OD1": (0.0, 0.3, 0.0), "OD2": (0.05, 0.3, 0.0)}),
               ("LYS", {**bb(1.0), "NZ": (0.3, 0.3, 0.0)})])     # OD1–NZ = 0.30 nm
n_sb, pairs = salt_bridges(close, cut_nm=0.4)
check("가까우면 1개 잡는다", n_sb == 1, f"{pairs}")
# ★ 카복실기 2개 × 구아니디늄 3개 = 원자쌍 6개. 원자쌍마다 문턱을 걸면 놓친다.
#   잔기 쌍으로 묶어 최대 점유율을 써야 한다.
multi = build([("GLU", {**bb(0.0), "OE1": (0.0, 0.3, 0.0), "OE2": (0.05, 0.3, 0.0)}),
               ("ARG", {**bb(1.0), "NH1": (0.30, 0.3, 0.0), "NH2": (0.34, 0.3, 0.0),
                        "NE": (0.38, 0.3, 0.0)})])
check("★ 여러 원자쌍이 한 염다리로 묶인다", salt_bridges(multi, cut_nm=0.4)[0] == 1,
      f"{salt_bridges(multi, cut_nm=0.4)[1]}")
farsb = build([("ASP", {**bb(0.0), "OD1": (0.0, 0.3, 0.0)}),
               ("LYS", {**bb(1.0), "NZ": (3.0, 0.3, 0.0)})])
check("멀면 0개", salt_bridges(farsb, cut_nm=0.4)[0] == 0)
#  ★ pH 3 의 핵심 ①: 이름이 ASH 인 중성 Asp 는 안 센다
neutral = build([("ASH", {**bb(0.0), "OD1": (0.0, 0.3, 0.0)}),
                 ("LYS", {**bb(1.0), "NZ": (0.3, 0.3, 0.0), "HZ3": (0.35, 0.35, 0)})])
check("중성화된 ASH 는 염다리로 안 센다", salt_bridges(neutral, cut_nm=0.4)[0] == 0,
      "← pH3 에서 염다리가 사라지는 것이 신호다")
#  ★ pH 3 의 핵심 ②: **CHARMM 은 양성자화해도 이름을 ASP 로 둔다.**
#    이름만 보면 하전으로 세어 버리고 pH3 신호를 통째로 놓친다. HD2 로 판정해야 한다.
charmm_ph3 = build([("ASP", {**bb(0.0), "OD1": (0.0, 0.3, 0.0),
                             "OD2": (0.05, 0.3, 0.0), "HD2": (0.07, 0.35, 0.0)}),
                    ("LYS", {**bb(1.0), "NZ": (0.3, 0.3, 0.0), "HZ3": (0.35, 0.35, 0)})])
check("★ 이름이 ASP 라도 HD2 가 있으면 안 센다 (CHARMM pH3 실제 모양)",
      salt_bridges(charmm_ph3, cut_nm=0.4)[0] == 0,
      "← 이걸 놓치면 'pH3 에서 염다리가 끊긴다' 를 영영 못 본다")
_res = lambda t, i: list(t.topology.residues)[i]
check("acid_is_charged: HD2 없으면 하전", acid_is_charged(_res(close, 0), True))
check("acid_is_charged: HD2 있으면 중성", not acid_is_charged(_res(charmm_ph3, 0), True))
check("base_is_charged: HZ3 있는 LYS 는 하전", base_is_charged(_res(charmm_ph3, 1), True))
check("base_is_charged: HD1+HE2 인 HIS 만 하전",
      base_is_charged(_res(build([("HIS", {**bb(0), "HD1": (0, .3, 0), "HE2": (.1, .3, 0)})]), 0), True)
      and not base_is_charged(_res(build([("HIS", {**bb(0), "HE2": (.1, .3, 0)})]), 0), True))
# ★★ 수소가 **아예 없는** 구조 — 규칙이 거꾸로 물어 염다리가 조용히 0 이 되는 자리
heavy = build([("ASP", {"CA": (0, 0, 0), "CB": (0, .15, 0), "OD1": (0.0, 0.3, 0.0)}),
               ("LYS", {"CA": (1, 0, 0), "CB": (1, .15, 0), "NZ": (0.3, 0.3, 0.0)})])
check("★ 수소 없는 PDB 에서도 염다리를 찾는다 (이름으로 후퇴)",
      salt_bridges(heavy, cut_nm=0.4)[0] == 1,
      "← 수소 기반 규칙만 쓰면 여기서 0 이 나오고 아무도 모른다")
check("산/염기 한쪽만 있으면 0",
      salt_bridges(build([("ASP", {**bb(0.0), "OD1": (0, .3, 0)})]))[0] == 0)

# ── 8. 계면 접촉 ───────────────────────────────────────────────────────────
print("\n[8] 계면 접촉")
#  A = 잔기 0,1 / B = 잔기 2,3.  0-2 만 0.3 nm, 나머지는 멀다
t8 = build([("ALA", bb(0.0)), ("ALA", bb(3.0)), ("ALA", bb(0.3)), ("ALA", bb(9.0))])
check("접촉 1쌍", abs(interface_contacts(t8, [0, 1], [2, 3], 0.45) - 1.0) < 1e-9,
      f"{interface_contacts(t8, [0,1], [2,3], 0.45):.1f}")
check("멀면 0쌍", interface_contacts(t8, [1], [3], 0.45) == 0.0)
try:
    interface_contacts(t8, [], [2]); check("빈 도메인은 거부", False, "예외가 안 났다")
except ValueError:
    check("빈 도메인은 거부", True)

# ── 9. 사슬 치수 ───────────────────────────────────────────────────────────
print("\n[9] 링커 치수 — 직선이면 Re 를 손으로 셀 수 있다")
#  CA 를 0.38 nm 간격 직선으로 6개 → Re = 5 × 0.38 nm = 1.90 nm = 19.0 Å
t9 = build([("GLY", bb(i * 0.38, ca_only=True)) for i in range(6)])
re, rg = chain_dims(t9, range(6))
check("직선 Re = 19.0 Å", abs(re - 19.0) < 1e-6, f"{re:.3f} Å")
check("Rg < Re", rg < re, f"Rg {rg:.2f} Å")
check("잔기가 모자라면 NaN", not np.isfinite(chain_dims(t9, [0])[0]))

# ── 10. 수렴과 항체내몫 ────────────────────────────────────────────────────
print("\n[10] 수렴 점검과 항체내몫")
check("표류가 없으면 0 근처", abs(block_drift(np.sin(np.arange(200)))) < 0.5,
      f"{block_drift(np.sin(np.arange(200))):.3f}")
ramp = np.arange(200, dtype=float)
check("단조 증가는 표류를 잡는다", block_drift(ramp) > 1.0, f"{block_drift(ramp):.2f}")
check("표본이 모자라면 NaN", not np.isfinite(block_drift([1.0, 2.0])))
rng = np.random.default_rng(1)
g = np.repeat(np.arange(5), 4)
between = np.repeat(rng.normal(0, 10.0, 5), 4) + rng.normal(0, 0.4, 20)
within = np.tile(np.arange(4) * 3.0, 5) + rng.normal(0, 0.2, 20)
sb, sw = within_share(between, g), within_share(within, g)
check("항체 간이 지배하면 몫이 작다", sb < 0.15, f"{sb:.3f}")
check("항체 안이 지배하면 몫이 크다", sw > 0.85, f"{sw:.3f}")
check("표본이 모자라면 NaN", not np.isfinite(within_share([1.0, 2.0], [0, 1])))
check("한 항체뿐이면 NaN", not np.isfinite(within_share([1., 2., 3.], [0, 0, 0])))

# ── 11. 다중 프레임 — 앙상블 평균이 실제로 평균인가 ────────────────────────
print("\n[11] 앙상블 평균")
jit = build([("LEU", bb(i * 2.0)) for i in range(4)], frames=12, jitter=0.02, seed=3)
check("여러 프레임에서도 돈다", np.isfinite(sap(jit)).all() and jit.n_frames == 12)
one = build([("LEU", bb(i * 2.0)) for i in range(4)], frames=1)
five = build([("LEU", bb(i * 2.0)) for i in range(4)], frames=5)
# ★ shrake_rupley 는 프레임 수에 따라 마지막 자리가 흔들린다 (구면 점 배치·누산 순서).
#   같은 좌표라도 비트 단위로 같지 않다 — 관측된 상대차 ~1e-4. 그래서 1e-9 이 아니라
#   1e-3 으로 본다. 이 사실을 모르면 재현성 시험이 영문 모르게 깨진다.
_d = abs(float(np.nanmean(sap(one))) - float(np.nanmean(sap(five))))
check("같은 좌표면 프레임 수가 달라도 같은 값 (수치오차 1e-3 이내)", _d < 1e-3,
      f"차이 {_d:.2e}  ← shrake_rupley 는 비트 재현성이 없다")

print("\n" + "=" * 70)
print(f"통과 {len(OK)} · 실패 {len(BAD)}")
if BAD:
    print("실패:", BAD)
    sys.exit(1)
print("전부 통과")
