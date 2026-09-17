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


def b22_from_rdf(r, g, r_core_from_grid=True):
    """동경분포함수 g(r) 에서 바로 B22 를 낸다 — CALVADOS 상류 조리법과 같은 식.

        B22 = −2π ∫ [g(r) − 1] r² dr

    g(r) = exp(−W(r)/kT) 이므로 (g−1) 이 곧 마이어 f 함수다. PMF 를 거치지 않으니
    g=0 인 배제 영역에서 log 가 발산하는 문제가 없다 — 그래서 궤적에서 낼 때는
    이 쪽이 `b22_from_pmf` 보다 안전하다.

    CALVADOS 의 two_IDR_MDP 예제가 쓰는 식과 같다:
        r, rdf = md.compute_rdf(t, pairs=[[0,1]], r_range=(.5,15), bin_width=.1)
        b22 = -2*np.pi*np.trapz((rdf-1)*r*r, r)

    다만 상류 식은 격자 **밖**(0 ~ r[0])을 빼먹는다. 거기는 언제나 완전 배제라
    +2π·r[0]³/3 을 더해야 한다. 단백질이면 r[0]=0.5 nm 에서 0.26 nm³ 로 작지만,
    작은 분자나 격자를 넓게 잡았을 때는 안 작다. r_core_from_grid=False 면 안 더한다.
    """
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    if r.ndim != 1 or r.shape != g.shape:
        raise ValueError("r 과 g 는 같은 길이의 1차원 배열이어야 한다")
    if len(r) < 3:
        raise ValueError("적분하려면 점이 3개는 있어야 한다")
    o = np.argsort(r)
    r, g = r[o], g[o]
    if np.any(r <= 0):
        raise ValueError("r 은 양수여야 한다")
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out = -2.0 * np.pi * tz((g - 1.0) * r ** 2, r)
    if r_core_from_grid:
        out += 2.0 * np.pi * (r[0] ** 3) / 3.0
    return float(out)


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
# 3. 두 형태의 혼합 — 열린 소수 집단이 붙는다는 기전을 수로 쓴다
# ─────────────────────────────────────────────────────────────────────────────
def b22_mixture(f_open, b_cc, b_oo, b_co=None):
    """닫힌 종과 열린 종이 섞여 있을 때의 **겉보기** B22.

        B22_app = (1−f)²·B_cc + 2f(1−f)·B_co + f²·B_oo

    ★ 선형 혼합이 아니다. B22 는 **쌍** 상호작용이라 조성에 2차로 들어간다.
      (1−f)·B_cc + f·B_oo 로 쓰면 틀린다 — 교차항 B_co 가 통째로 빠진다.

    f_open  열린 분율. **좌표 기반**으로 잰 것을 써라 (ABangle 의 dc 는
            무성 오염되면 실제 열림과 무관해진다 — 실측 rho +0.06 이었다).
    b_cc    닫힌–닫힌 (nm³).  b_oo  열린–열린.  b_co  교차항.

    b_co 를 안 주면 **구간**을 돌려준다 — 교차항은 두 동종항 사이 어딘가지만
    (min, max) 밖으로는 잘 안 나간다. 지어낸 한 값을 주는 것보다 정직하다.
    정확한 값이 필요하면 닫힌 것 하나 + 열린 것 하나를 한 상자에 넣고 따로 재라.

    기전이 '열린 것끼리 붙는다' 면 지배항은 f²·B_oo 다. f=0.25 면 계수가 0.0625
    로 작아 보이지만 B_oo 가 크게 음수면 이 항이 전체를 끌고 간다.
    """
    f = float(f_open)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"f_open 은 0~1 이어야 한다: {f}")
    b_cc, b_oo = float(b_cc), float(b_oo)
    if b_co is None:
        lo, hi = (min(b_cc, b_oo), max(b_cc, b_oo))
        return (round((1 - f) ** 2 * b_cc + 2 * f * (1 - f) * lo + f ** 2 * b_oo, 6),
                round((1 - f) ** 2 * b_cc + 2 * f * (1 - f) * hi + f ** 2 * b_oo, 6))
    return float((1 - f) ** 2 * b_cc + 2 * f * (1 - f) * float(b_co) + f ** 2 * b_oo)


def open_excess(f_open, b_cc, b_oo):
    """열림이 **더한** 몫만 떼어 본다:  B22_app(f) − B_cc.

    b_co 를 모를 때도 부호와 크기를 볼 수 있는 축약형이다 (교차항을 두 동종항의
    산술평균으로 두면 정확히 f²(B_oo−B_cc) 가 아니라 f(2−f)... 가 되므로,
    여기서는 **교차항을 B_cc 로 두는 보수적 가정**을 쓴다 → f²·(B_oo − B_cc)).
    즉 '열린 것끼리 만나야만 추가 인력이 생긴다' 는 가장 보수적인 읽기다.
    """
    f = float(f_open)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"f_open 은 0~1 이어야 한다: {f}")
    return float(f ** 2 * (float(b_oo) - float(b_cc)))


def conformer_noise(values_by_frame):
    """같은 구성체에서 **다른 프레임**을 넣었을 때 값이 얼마나 흔들리나.

    구조를 강체로 고정하면 '어느 프레임을 골랐나' 가 곧 답이 된다.
    구성체 간 차이가 이 흔들림보다 작으면 우리가 재는 것은 **프레임 선택 잡음**이다.
    씨앗 반복과 같은 역할을 한다 — 분모가 없으면 분자를 읽을 수 없다.

    반환: (프레임간SD, 구성체수가 1일 때 nan)
    """
    v = np.asarray(values_by_frame, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.std(v, ddof=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 라벨을 안 보는 사전점검 — 이 관측량이 링커를 재나 항체를 재나
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


__all__ = ["b22_from_pmf", "b22_from_rdf", "b22_reduced", "b22_mixture", "open_excess",
           "conformer_noise", "sticky_exposure", "linker_sweep",
           "intermolecular_contacts", "within_share", "KB"]
