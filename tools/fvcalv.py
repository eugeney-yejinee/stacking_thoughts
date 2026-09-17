"""CALVADOS 두 사슬 궤적 → 회합 경향 관측량.

    왜 이게 필요한가
    ─────────────────
    BioEmu 는 **단일 사슬만** 다룬다. 우리가 잰 것은 전부 "단량체 하나가 어떤
    모양인가" 이고, HMW 는 "둘이 만나서 붙었나" 다. 그 사이는 모델이 아니라
    가정이었다. CALVADOS 는 여러 사슬을 한 상자에 넣고 **실제로 재는** 모델이라
    그 간극을 건넌다.

    그리고 단일 사슬 기술자로는 무엇을 정의해도 결국 링커 길이의 함수로
    환원된다는 것을 실측에서 확인했다 — `링커신장도` 가 길이와 rho −0.99 였다
    (신장도 = 말단간/윤곽길이인데 말단간은 도메인 기하가 정하니 거의 상수다).
    두 사슬을 만나게 해야 새 축이 나온다.

    무엇을 재나
    ─────────────
    1. **B22** (2차 비리얼 계수). 음수 = 알짜 인력 = 응집 경향.
       SLS·CG-MALS 로 실측도 되는 양이라 나중에 검증할 수 있다.
    2. **끈끈한 면 노출** — 링커가 표면을 덮고 있나 비켜 있나.
       사용자 가설("링커가 자유분방하면 표면이 더 자주 드러난다")의 직접 대응물.
    3. **분자간 접촉** — 만남당 붙음의 조야한 대리값.

    쓰지 말아야 할 곳
    ──────────────────
    CALVADOS 의 λ 끈끈함 척도는 **IDP** 의 Rg 와 LLPS 를 재현하도록 피팅됐다.
    모양 상보성이 만드는 **특이적 계면을 모델하지 못한다.** 그래서
    VH(1)-VL(2) 도메인 교환(diabody)은 재현되지 않는다. 이 모듈이 재는 것은
    **콜로이드적·비특이적 회합**이다. 둘은 다른 기전이고, 이것으로 도메인 교환
    가설을 검정했다고 말하면 안 된다.

    시험:  python3 tools/test_fvcalv.py
"""
from __future__ import annotations

import numpy as np

#: 볼츠만 상수 (kJ/mol/K) — CALVADOS/OpenMM 의 단위계
KB = 0.008314462618


# ─────────────────────────────────────────────────────────────────────────────
# 1. B22 — 평균력퍼텐셜에서
# ─────────────────────────────────────────────────────────────────────────────
def b22_from_pmf(r, w, T=298.15, r_core=None):
    """중심간 거리 PMF 에서 2차 비리얼 계수를 낸다.

        B22 = −2π ∫₀^∞ [exp(−W(r)/kT) − 1] r² dr

    r 은 두 분자 질량중심 거리 (nm), w 는 그 거리에서의 PMF (kJ/mol).
    PMF 는 **먼 거리에서 0 이 되도록** 맞춰져 있어야 한다 — 안 그러면 상수
    치우침이 적분 전체를 끌고 간다. 여기서 바깥 5% 구간의 중앙값으로 영점을 잡는다.

    r_core 아래는 완전 배제(exp(−βW)=0)로 본다. 안 주면 r 의 첫 점을 쓴다.
    그 아래 구간이 기여하는 것은 딱딱한 구의 배제부피 +2π·r_core³/3 이고,
    이것을 빼먹으면 B22 가 음수 쪽으로 통째로 밀린다.

    반환은 nm³. 실험에서 흔히 쓰는 mL·mol/g² 로 바꾸려면
    B22[mL·mol/g²] = B22[nm³] · N_A · 1e-21 / M² (M 은 g/mol).
    """
    r = np.asarray(r, float)
    w = np.asarray(w, float)
    if r.ndim != 1 or r.shape != w.shape:
        raise ValueError("r 과 w 는 같은 길이의 1차원 배열이어야 한다")
    if len(r) < 3:
        raise ValueError("적분하려면 점이 3개는 있어야 한다")
    o = np.argsort(r)
    r, w = r[o], w[o]
    if np.any(r <= 0):
        raise ValueError("r 은 양수여야 한다")

    # ── 꼬리에서 0 으로 맞춘다 ────────────────────────────────────────────
    ntail = max(1, int(round(0.05 * len(r))))
    w = w - float(np.median(w[-ntail:]))

    core = float(r[0]) if r_core is None else float(r_core)
    if core < 0:
        raise ValueError("r_core 는 음수일 수 없다")

    f = np.exp(-w / (KB * T)) - 1.0          # 마이어 f 함수
    f = np.where(r < core, -1.0, f)          # 코어 안은 완전 배제
    integral = np.trapezoid(f * r ** 2, r) if hasattr(np, "trapezoid") \
        else np.trapz(f * r ** 2, r)
    # ★ r[0] ~ core 구간은 **이미 격자 위에서 f=−1 로 적분됐다.** 여기서 또
    #   해석적으로 더하면 두 번 센다 (자체 시험이 딱딱한 구에서 2배를 잡아냈다).
    #   격자에 없는 것은 0 ~ r[0] 구간뿐이고, 거기는 언제나 완전 배제다.
    hard_core = -(r[0] ** 3) / 3.0
    return float(-2.0 * np.pi * (integral + hard_core))


def b22_reduced(b22_nm3, sigma):
    """B22 를 같은 접촉지름 딱딱한 구의 값으로 나눈 무차원 값.

    sigma 는 **접촉 지름** (두 분자가 닿는 중심간 거리, nm).
    딱딱한 구의 해석해는  B2_HS = (2π/3)·σ³  이다
    (B2 = −2π∫₀^σ(−1)r²dr = 2πσ³/3).

    1 이면 딱딱한 구와 같다, < 1 이면 인력이 있다, < 0 이면 알짜 인력이다.
    분자 크기가 링커마다 조금씩 다르므로 **크기 효과를 빼고** 비교할 때 쓴다 —
    링커가 길어지면 σ 가 커져서 B22 자체는 크기 때문에도 오른다.
    """
    hs = (2.0 * np.pi / 3.0) * float(sigma) ** 3
    if hs <= 0:
        return float("nan")
    return float(b22_nm3 / hs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 끈끈한 면 노출 — 링커가 덮고 있나
# ─────────────────────────────────────────────────────────────────────────────
def sticky_exposure(pos, patch_idx, linker_idx, cut=0.8):
    """패치 비드 중 **링커에 안 가려진** 비율. 프레임마다 하나씩.

    pos        (프레임, 비드, 3) 좌표 (nm)
    patch_idx  끈끈한 패치 비드 색인 (소수성 표면 · VH-VL 계면 가장자리 등)
    linker_idx 링커 비드 색인
    cut        가림 판정 거리 (nm). 0.8 nm 은 CG 비드 하나 지름쯤이다.

    사용자 가설의 직접 대응물이다 — 링커가 자유분방하면 표면을 덜 가리고,
    덜 가리면 다른 분자와 붙을 면이 더 자주 드러난다.

    ★ 이 값이 링커마다 다르려면 링커가 **실제로 도메인 근처를 지나야** 한다.
      전혀 안 닿는 링커라면 모든 구성체에서 1.0 이 나오고 쓸모가 없다 —
      그때는 그 사실 자체를 보고하라 (상수인 특징은 정보가 없다).
    """
    pos = np.asarray(pos, float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("pos 는 (프레임, 비드, 3) 이어야 한다")
    patch_idx = np.asarray(patch_idx, int)
    linker_idx = np.asarray(linker_idx, int)
    if len(patch_idx) == 0:
        raise ValueError("패치 비드가 비었다")
    if len(linker_idx) == 0:
        return np.ones(len(pos))            # 링커가 없으면 아무것도 안 가린다
    out = np.empty(len(pos))
    for t, p in enumerate(pos):
        d = np.linalg.norm(p[patch_idx][:, None, :] - p[linker_idx][None, :, :],
                           axis=-1)
        out[t] = float((d.min(axis=1) > cut).mean())
    return out


def linker_sweep(pos, linker_idx):
    """링커가 훑는 부피의 대리값 — 비드 위치 공분산의 행렬식^(1/2) 을 프레임 평균.

    '자유분방함' 을 크기가 아니라 **퍼짐**으로 잰다. 신장도(말단간/윤곽길이)는
    도메인 기하가 말단간을 정해 버려서 길이의 변장이 되지만(실측 rho −0.99),
    이 값은 같은 말단 구속 아래에서도 링커가 얼마나 넓게 움직이는지를 본다.
    """
    pos = np.asarray(pos, float)
    linker_idx = np.asarray(linker_idx, int)
    if len(linker_idx) < 3:
        return float("nan")
    v = []
    for p in pos:
        L = p[linker_idx]
        c = np.cov((L - L.mean(0)).T)
        det = float(np.linalg.det(c))
        v.append(np.sqrt(max(det, 0.0)))
    return float(np.mean(v))


def intermolecular_contacts(pos, idx_a, idx_b, cut=0.8):
    """프레임별 분자 간 접촉 비드쌍 수. 만남당 붙음의 조야한 대리값."""
    pos = np.asarray(pos, float)
    idx_a, idx_b = np.asarray(idx_a, int), np.asarray(idx_b, int)
    out = np.empty(len(pos))
    for t, p in enumerate(pos):
        d = np.linalg.norm(p[idx_a][:, None, :] - p[idx_b][None, :, :], axis=-1)
        out[t] = float((d < cut).sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 라벨을 안 보는 사전점검 — 이 관측량이 링커를 재나 항체를 재나
# ─────────────────────────────────────────────────────────────────────────────
def within_share(values, groups):
    """항체내몫 = 항체내분산 / (항체간분산 + 항체내분산).

    ★ 이것을 **라벨 검정보다 먼저** 봐야 한다. 타깃을 항체 안에서 중심화하므로
      항체 간 성분은 통째로 지워진다. 0.3 미만이면 그 관측량은 링커가 아니라
      **항체 정체성**을 재고 있고, 검정력이 아니라 정보가 없는 것이다.
      실측에서 각도 특징들이 정확히 그래서 실패했다 (0.20~0.29).
      y 를 안 쓰므로 몇 번을 봐도 다중비교가 안 생긴다.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups)
    m = np.isfinite(v)
    v, g = v[m], g[m]
    if len(v) < 4 or len(np.unique(g)) < 2:
        return float("nan")
    means, within = [], []
    for k in np.unique(g):
        s = v[g == k]
        means.append(s.mean())
        within.extend(s - s.mean())
    sb = float(np.std(means, ddof=1))
    sw = float(np.std(within, ddof=1))
    tot = sb ** 2 + sw ** 2
    return float(sw ** 2 / tot) if tot > 1e-30 else float("nan")


__all__ = ["b22_from_pmf", "b22_reduced", "sticky_exposure", "linker_sweep",
           "intermolecular_contacts", "within_share", "KB"]
