"""FvLinker 관측값 — MD 궤적에서 **응집 핵과 링커 가림**을 잰다.

엔진에 안 묶여 있다. mdtraj Trajectory 와 잔기 색인만 받는다.
전부 **답을 아는 경우**로 시험한다 (test_fvobs.py).

────────────────────────────────────────────────────────────────────────────
왜 이 관측값들인가
────────────────────────────────────────────────────────────────────────────
예측 대상은 항체 **안에서의** HMW 순위다. 그런데 지금까지 이 프로젝트의 모든
축이 같은 자리에서 죽었다 — 항체내몫 < 0.3, 즉 링커가 아니라 **항체 정체**를
재고 있었다. 그래서 관측값은 둘로 갈라 놓는다:

  A. 항체 성질 (f 항)   — 계면 강도, pH3 에서 끊기는 염다리
  B. 링커 성질 (c 항)   — 링커가 끈끈한 면을 **얼마나 덮는가**

B 가 항체 안에서 변하는 유일한 것이다. A 는 항체 간 비교에만 쓴다.
`within_share()` 를 **y 를 보기 전에** 돌려서 어느 쪽인지 먼저 판정한다.
"""
from __future__ import annotations

import numpy as np

# ── 잔기별 최대 노출 면적 (nm²) — Tien et al. 2013 이론값 ──────────────────
#   SAP 의 분모다. 곁사슬이 완전히 노출됐을 때의 면적.
MAXSASA = {
    "ALA": 1.290, "ARG": 2.740, "ASN": 1.945, "ASP": 1.930, "CYS": 1.670,
    "GLN": 2.250, "GLU": 2.230, "GLY": 1.040, "HIS": 2.240, "ILE": 1.970,
    "LEU": 2.010, "LYS": 2.360, "MET": 2.240, "PHE": 2.280, "PRO": 1.590,
    "SER": 1.550, "THR": 1.720, "TRP": 2.590, "TYR": 2.550, "VAL": 1.740,
}
# CHARMM 의 양성자화 변종 이름도 같은 잔기로 취급한다 (pH 3 에서 나온다)
for _a, _b in (("ASH", "ASP"), ("GLH", "GLU"), ("HIP", "HIS"), ("HSP", "HIS"),
               ("HSD", "HIS"), ("HSE", "HIS"), ("LYN", "LYS"), ("CYX", "CYS")):
    MAXSASA[_a] = MAXSASA[_b]

# ── 소수성 척도 — Black & Mould 1991, 0.5 를 빼서 중심을 맞춘다 ────────────
#   SAP 원논문이 쓰는 척도다. 양수면 소수성, 음수면 친수성.
BLACK_MOULD = {
    "ALA": 0.616, "ARG": 0.000, "ASN": 0.236, "ASP": 0.028, "CYS": 0.680,
    "GLN": 0.251, "GLU": 0.043, "GLY": 0.501, "HIS": 0.165, "ILE": 0.943,
    "LEU": 0.943, "LYS": 0.283, "MET": 0.738, "PHE": 1.000, "PRO": 0.711,
    "SER": 0.359, "THR": 0.450, "TRP": 0.878, "TYR": 0.880, "VAL": 0.825,
}
for _a, _b in (("ASH", "ASP"), ("GLH", "GLU"), ("HIP", "HIS"), ("HSP", "HIS"),
               ("HSD", "HIS"), ("HSE", "HIS"), ("LYN", "LYS"), ("CYX", "CYS")):
    BLACK_MOULD[_a] = BLACK_MOULD[_b]

ACID_NAMES = {"ASP", "GLU", "ASH", "GLH", "ASPP", "GLUP"}
BASE_NAMES = {"LYS", "ARG", "HIS", "HIP", "HSP", "HSD", "HSE", "LYN"}
# 산성 잔기의 **양성자** 이름. 이게 붙어 있으면 중성이다 (pH 3 의 신호 그 자체).
ACID_PROTON = {"ASP": ("HD2",), "ASH": ("HD2",), "ASPP": ("HD2",),
               "GLU": ("HE2",), "GLH": ("HE2",), "GLUP": ("HE2",)}


def has_hydrogens(top):
    """이 위상에 수소가 붙어 있나. 양성자화 판정 방식을 여기서 가른다."""
    for a in top.atoms:
        if a.element is not None and a.element.symbol == "H":
            return True
    return False


def acid_is_charged(res, with_h=None):
    """Asp/Glu 가 **하전 상태**인가.

    ★ CHARMM 은 양성자화해도 잔기 이름을 `ASP`/`GLU` **그대로 둔다** (ASPP/GLUP
      템플릿에 그래프로 맞춘다). 그래서 이름으로 판정하면 pH 3 에서도 하전으로
      세어 버리고, "pH 3 에서 염다리가 끊긴다" 는 신호를 통째로 놓친다.
      **수소 유무**로 봐야 한다 — 이황화가 HG 유무로 갈리는 것과 같은 함정이다.

    ★★ 그런데 구조에 수소가 **아예 없으면** 이 규칙이 거꾸로 문다: 산은 전부
      "하전"으로, 염기는 전부 "중성"으로 읽혀 염다리가 조용히 0 개가 된다.
      그래서 with_h 로 갈라 놓고, 수소가 없으면 이름으로만 판정한다.
      with_h=None 이면 잔기가 속한 위상을 보고 알아서 정한다.
    """
    nm = res.name.upper()
    if nm not in ACID_NAMES:
        return False
    if nm in ("ASH", "GLH", "ASPP", "GLUP"):
        return False                      # 이름이 이미 중성형이면 그대로 믿는다
    if with_h is None:
        with_h = any(a.element is not None and a.element.symbol == "H"
                     for a in res.atoms)
    if not with_h:
        return True                       # 수소가 없다 → 이름대로 ASP/GLU = 하전
    names = {a.name for a in res.atoms}
    return not any(h in names for h in ACID_PROTON.get(nm, ()))


def base_is_charged(res, with_h=None):
    """Lys/Arg/His 가 **하전 상태**인가. His 는 두 자리가 다 차야(HIP) 양전하다."""
    nm = res.name.upper()
    if nm not in BASE_NAMES:
        return False
    if nm == "ARG":
        return True
    if nm in ("HIP", "HSP"):
        return True
    if nm == "LYN":
        return False
    if with_h is None:
        with_h = any(a.element is not None and a.element.symbol == "H"
                     for a in res.atoms)
    names = {a.name for a in res.atoms}
    if nm == "LYS":
        return ("HZ3" in names) if with_h else True
    if nm in ("HIS", "HSD", "HSE"):
        # 수소가 있으면 HD1+HE2 가 다 있어야 HIP(+). 없으면 판정 불가 → 중성으로 본다
        return (("HD1" in names) and ("HE2" in names)) if with_h else False
    return False


def _resnames(traj):
    return [r.name.upper() for r in traj.topology.residues]


def rel_exposure(traj):
    """잔기별 **상대 노출도** (0~1). 앙상블 평균. SAP 의 재료다.

    반환: (n잔기,) 배열. 최대면적을 모르는 잔기(리간드 등)는 NaN.
    """
    import mdtraj as md
    if traj.n_frames == 0:
        raise ValueError("프레임이 0개다")
    sasa = md.shrake_rupley(traj, mode="residue")        # (frame, res) nm²
    per = sasa.mean(axis=0)
    mx = np.array([MAXSASA.get(n, np.nan) for n in _resnames(traj)], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = per / mx
    return np.clip(rel, 0.0, 1.5)


def sap(traj, radius_nm=0.5):
    """Spatial Aggregation Propensity (Chennamsetty & Trout 2009).

        SAP(i) = Σ_{j: |CB_i − CB_j| < R}  ⟨SAA_j / SAA_j^max⟩ · h_j

    원논문은 원자 단위지만 여기서는 **잔기 단위**로 근사한다 (CB 기준).
    이웃 반경 R 은 원논문의 5 Å 를 그대로 쓴다.
    반환: (n잔기,) SAP 값. 양수 = 노출된 소수성 = 응집 성향.
    """
    rel = rel_exposure(traj)
    names = _resnames(traj)
    h = np.array([BLACK_MOULD.get(n, np.nan) - 0.5 for n in names], float)
    xyz = _cb_coords(traj)                                # (n잔기, 3) 평균 좌표
    ok = np.isfinite(rel) & np.isfinite(h) & np.isfinite(xyz).all(axis=1)
    n = len(names)
    out = np.full(n, np.nan)
    if ok.sum() < 2:
        return out
    idx = np.where(ok)[0]
    d = np.linalg.norm(xyz[idx][:, None, :] - xyz[idx][None, :, :], axis=-1)
    near = d < float(radius_nm)
    contrib = (rel[idx] * h[idx])
    out[idx] = (near * contrib[None, :]).sum(axis=1)
    return out


def _cb_coords(traj):
    """잔기별 CB 좌표 (Gly 는 CA). 앙상블 평균. 단위 nm."""
    top = traj.topology
    n = top.n_residues
    xyz = np.full((n, 3), np.nan)
    for i, r in enumerate(top.residues):
        a = next((x for x in r.atoms if x.name == "CB"), None) or \
            next((x for x in r.atoms if x.name == "CA"), None)
        if a is not None:
            xyz[i] = traj.xyz[:, a.index, :].mean(axis=0)
    return xyz


def largest_patch(traj, rel_cut=0.25, link_nm=0.8):
    """가장 큰 **연속** 소수성 노출 패치의 면적 (nm²).

    응집은 총 면적이 아니라 **하나로 이어진 패치**가 핵이 된다. 총량이 같아도
    잘게 흩어져 있으면 안 붙는다. 그래서 총 hSASA 와 따로 잰다.

    노출도 rel_cut 이상이고 소수성이 양수인 잔기를 CB 거리 link_nm 로 이어
    가장 큰 연결 성분을 찾고, 그 성분의 SASA 합을 돌려준다.
    """
    import mdtraj as md
    rel = rel_exposure(traj)
    names = _resnames(traj)
    h = np.array([BLACK_MOULD.get(n, np.nan) - 0.5 for n in names], float)
    sel = np.where((rel > rel_cut) & (h > 0) & np.isfinite(rel))[0]
    if len(sel) == 0:
        return 0.0, []
    xyz = _cb_coords(traj)[sel]
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adj = d < float(link_nm)
    seen, best = set(), []
    for s in range(len(sel)):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack.extend(np.where(adj[u])[0].tolist())
        if len(comp) > len(best):
            best = comp
    per = md.shrake_rupley(traj, mode="residue").mean(axis=0)
    members = sorted(int(sel[c]) for c in best)
    return float(per[members].sum()), members


def linker_shield(traj, patch_res, linker_res, cut_nm=0.8):
    """패치 중 **링커에 가려지지 않은** 분율 (0~1). 1 = 전부 드러남.

    ★ 이게 항체 안에서 변하는 유일한 양이다. 같은 항체면 도메인 표면이 같으니
      패치도 같고, 링커만 다르다. 그래서 이 값의 항체내몫이 커야 한다.
    """
    patch_res = list(patch_res); linker_res = list(linker_res)
    if not patch_res:
        raise ValueError("패치가 비었다")
    if not linker_res:
        return 1.0
    xyz = _cb_coords(traj)
    p, l = xyz[patch_res], xyz[linker_res]
    if not (np.isfinite(p).all() and np.isfinite(l).all()):
        ok_p = np.isfinite(p).all(axis=1); ok_l = np.isfinite(l).all(axis=1)
        p, l = p[ok_p], l[ok_l]
        if len(p) == 0 or len(l) == 0:
            return float("nan")
    d = np.linalg.norm(p[:, None, :] - l[None, :, :], axis=-1)
    covered = (d < float(cut_nm)).any(axis=1)
    return float(1.0 - covered.mean())


def salt_bridges(traj, cut_nm=0.4):
    """염다리 목록과 개수. pH 3 에서 Asp/Glu 가 중성이 되면 **사라져야 한다.**

    산성 곁사슬 산소(OD*/OE*)와 염기성 질소(NZ/NH*/NE*/ND1/NE2)가 cut 안에
    들어온 프레임 비율이 0.3 을 넘으면 하나로 센다.
    """
    top = traj.topology
    wh = has_hydrogens(top)          # ★ 한 번만 판정해서 전 잔기에 같은 규칙을 쓴다
    acid, base = [], []
    for r in top.residues:
        if acid_is_charged(r, wh):
            acid += [a.index for a in r.atoms if a.name.startswith(("OD", "OE"))
                     and a.name not in ("OD", "OE")]
        elif base_is_charged(r, wh):
            base += [a.index for a in r.atoms
                     if a.name in ("NZ", "NH1", "NH2", "NE", "ND1", "NE2")]
    if not acid or not base:
        return 0, []
    # ★ (프레임, 산, 염기, 3) 중간배열이 실제 규모에서 수십 MB 라 프레임을 잘라 돈다.
    pairs = []
    occ = np.zeros((len(acid), len(base)), float)
    CH = max(1, int(2e7 // max(len(acid) * len(base) * 3, 1)))      # 청크 크기
    for s0 in range(0, traj.n_frames, CH):
        A = traj.xyz[s0:s0 + CH, acid, :]; B = traj.xyz[s0:s0 + CH, base, :]
        d = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)
        occ += (d < float(cut_nm)).sum(axis=0)
    occ /= max(traj.n_frames, 1)
    ai, bi = np.where(occ > 0.3)
    seen = set()
    for x, y in zip(ai, bi):
        ra = top.atom(acid[x]).residue; rb = top.atom(base[y]).residue
        k = (ra.index, rb.index)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((f"{ra.name}{ra.resSeq}", f"{rb.name}{rb.resSeq}",
                      round(float(occ[x, y]), 3)))
    return len(seen), pairs


def interface_contacts(traj, res_a, res_b, cut_nm=0.45):
    """두 도메인 사이 잔기 접촉 수 (프레임 평균). VH–VL 계면이 버티는지 본다.

    ★ 순수 파이썬 삼중 루프로 짜면 실제 scFv(120×110 잔기 × 1000 프레임)에서
      1300만 번 호출이 돼 못 쓴다. mdtraj 의 C 구현(compute_contacts)에 넘긴다.
    """
    import mdtraj as md
    res_a, res_b = list(res_a), list(res_b)
    if not res_a or not res_b:
        raise ValueError("도메인 잔기 목록이 비었다")
    pairs = np.array([(i, j) for i in res_a for j in res_b], dtype=int)
    if len(pairs) == 0:
        return 0.0
    d, _ = md.compute_contacts(traj, contacts=pairs, scheme="closest-heavy",
                               periodic=bool(traj.unitcell_lengths is not None))
    return float((d < float(cut_nm)).sum(axis=1).mean())


def chain_dims(traj, res_idx):
    """링커 구간의 실제 말단간 거리 Re 와 회전반경 Rg (Å). 앙상블 평균.

    ALBATROSS 의 서열 예측과 **직접 비교**할 수 있는 값이다 — MD 가 예측기를
    검증한다. pH 7 에서 맞고 pH 3 에서 갈라지면 예측기를 못 믿는다는 뜻이다.
    """
    res_idx = list(res_idx)
    if len(res_idx) < 2:
        return float("nan"), float("nan")
    top = traj.topology
    ca = [next((a.index for a in top.residue(i).atoms if a.name == "CA"), None)
          for i in res_idx]
    ca = [c for c in ca if c is not None]
    if len(ca) < 2:
        return float("nan"), float("nan")
    P = traj.xyz[:, ca, :] * 10.0                       # nm → Å
    re = float(np.linalg.norm(P[:, -1, :] - P[:, 0, :], axis=-1).mean())
    c = P - P.mean(axis=1, keepdims=True)
    rg = float(np.sqrt((c ** 2).sum(axis=-1).mean(axis=1)).mean())
    return re, rg


def block_drift(values, nblocks=4):
    """수렴 점검. 궤적을 블록으로 나눈 평균의 **추세**를 SD 로 나눈 값.

    |drift| 가 1 을 넘으면 아직 안 정해진 값이다 — 더 돌려야 한다.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) < nblocks * 2:
        return float("nan")
    b = np.array([m.mean() for m in np.array_split(v, nblocks)])
    sd = v.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float((b[-1] - b[0]) / sd)


def within_share(values, groups):
    """항체내 분산 몫 (0~1). **y 를 보기 전에** 돌리는 사전점검이다.

    0.3 미만이면 그 관측값은 링커가 아니라 **항체 정체**를 재고 있다.
    지금까지 이 프로젝트의 모든 축이 여기서 죽었다.
    """
    v = np.asarray(values, float); g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if len(v) < 3 or len(set(g.tolist())) < 2:
        return float("nan")
    tot = v.var(ddof=1)
    if tot <= 0:
        return float("nan")
    wit = 0.0; n = 0
    for k in set(g.tolist()):
        m = g == k
        if m.sum() >= 2:
            wit += ((v[m] - v[m].mean()) ** 2).sum(); n += m.sum() - 1
    if n == 0:
        return float("nan")
    return float(min(1.0, (wit / n) / tot))


def find_span(seq, sub):
    """전체 서열에서 부분서열의 [시작, 끝) 잔기 색인. 못 찾으면 None.

    링커 구간을 **서열로** 찾는다. 사슬 나눔이나 잔기 번호에 기대면
    조용히 엉뚱한 자리를 집는다 — ABangle 이 그렇게 14% 오염됐다.
    """
    if not seq or not sub:
        return None
    i = seq.find(sub)
    if i < 0:
        return None
    if seq.find(sub, i + 1) >= 0:
        return None                     # 여러 번 나오면 모호하다 — 안 쓴다
    return (i, i + len(sub))


AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V", "ASH": "D", "GLH": "E", "HIP": "H", "HSP": "H",
          "HSD": "H", "HSE": "H", "LYN": "K", "CYX": "C"}


def traj_sequence(traj):
    """궤적 위상 → 1문자 서열. 양성자화 변종 이름도 원래 잔기로 돌려놓는다."""
    return "".join(AA3to1.get(r.name.upper(), "X") for r in traj.topology.residues)
