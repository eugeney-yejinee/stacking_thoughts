"""fvcalv 자체 시험 — 답을 아는 경우로만 검산한다."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fvcalv import (b22_from_pmf, b22_reduced, sticky_exposure, linker_sweep,
                    intermolecular_contacts, within_share)

OK, BAD = [], []


def check(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(f"  {'OK ' if cond else '★NG'} {name}" + (f"  {extra}" if extra else ""))


# ── 1. B22 — 딱딱한 구는 해석해가 있다 ──────────────────────────────────────
print("\n[1] B22 — 딱딱한 구 해석해와 맞나")
R = 2.0                                   # 반지름 2 nm → 지름 4 nm
r = np.linspace(0.05, 40.0, 4000)
w = np.zeros_like(r)                      # 코어 밖은 상호작용 없음
SIG = 2 * R                               # 접촉 지름 4 nm
b = b22_from_pmf(r, w, r_core=SIG)
exact = (2 * np.pi / 3) * SIG ** 3        # B2_HS = (2π/3)σ³  = 2π∫₀^σ r²dr
check("딱딱한 구 B22", abs(b - exact) / exact < 0.02,
      f"계산 {b:.2f} vs 해석 {exact:.2f} nm³")
check("무차원화 = 1", abs(b22_reduced(b, SIG) - 1.0) < 0.02,
      f"{b22_reduced(b, SIG):.4f}")
# 코어를 키우면 배제부피가 σ³ 으로 커져야 한다
b_big = b22_from_pmf(r, np.zeros_like(r), r_core=2 * SIG)
check("B22 ∝ σ³", abs(b_big / b - 8.0) < 0.05, f"비 {b_big / b:.3f} (기대 8)")

# ── 2. 인력을 넣으면 B22 가 내려간다 ────────────────────────────────────────
print("\n[2] 인력을 넣으면 B22 가 내려가야 한다")
depth = [0.0, 1.0, 3.0, 6.0]
bs = []
for d in depth:
    wa = np.where(r < SIG + 1.0, -d, 0.0)        # 코어 밖 1 nm 우물
    bs.append(b22_from_pmf(r, wa, r_core=SIG))
check("우물이 깊을수록 B22 감소", all(bs[i] > bs[i + 1] for i in range(len(bs) - 1)),
      " ".join(f"{x:.0f}" for x in bs))
check("충분히 깊으면 음수", bs[-1] < 0, f"{bs[-1]:.1f} nm³")

# ── 3. 꼬리 영점 보정 — 상수를 더해도 답이 안 변해야 한다 ───────────────────
print("\n[3] PMF 에 상수를 더해도 B22 가 안 변해야 한다")
w2 = np.where(r < SIG + 1.0, -2.0, 0.0)
b_a = b22_from_pmf(r, w2, r_core=SIG)
b_b = b22_from_pmf(r, w2 + 17.3, r_core=SIG)
check("상수 치우침 제거", abs(b_a - b_b) / max(abs(b_a), 1e-9) < 1e-6,
      f"{b_a:.2f} vs {b_b:.2f}")

# ── 4. 끈끈한 면 노출 — 손으로 셀 수 있는 배치 ──────────────────────────────
print("\n[4] 끈끈한 면 노출")
# 패치 비드 4개를 x축에 두고, 링커 비드 하나를 그중 2개 옆에 붙인다
P = np.array([[0., 0, 0], [2., 0, 0], [4., 0, 0], [6., 0, 0],   # 패치 0..3
              [0.3, 0, 0], [2.3, 0, 0]])                        # 링커 4,5
pos = P[None, :, :]
e = sticky_exposure(pos, [0, 1, 2, 3], [4, 5], cut=0.8)
check("절반 가려짐", abs(e[0] - 0.5) < 1e-9, f"노출 {e[0]:.2f} (기대 0.50)")
e2 = sticky_exposure(pos, [0, 1, 2, 3], [], cut=0.8)
check("링커가 없으면 전부 노출", abs(e2[0] - 1.0) < 1e-9)
far = P.copy(); far[4:] += 50.0
e3 = sticky_exposure(far[None], [0, 1, 2, 3], [4, 5], cut=0.8)
check("링커가 멀면 전부 노출", abs(e3[0] - 1.0) < 1e-9)

# ── 5. 훑는 부피 — 넓게 퍼진 링커가 더 커야 한다 ────────────────────────────
print("\n[5] 링커 훑는 부피")
rng = np.random.default_rng(0)
tight = rng.normal(0, 0.3, (60, 12, 3))
loose = rng.normal(0, 1.2, (60, 12, 3))
idx = np.arange(12)
st, sl = linker_sweep(tight, idx), linker_sweep(loose, idx)
check("퍼진 쪽이 크다", sl > st * 5, f"좁음 {st:.3f} · 넓음 {sl:.3f}")
check("비드가 모자라면 NaN", not np.isfinite(linker_sweep(tight, [0, 1])))

# ── 6. 분자간 접촉 ──────────────────────────────────────────────────────────
print("\n[6] 분자간 접촉")
A = np.array([[0., 0, 0], [1., 0, 0]])
B = np.array([[0.5, 0, 0], [10., 0, 0]])
pc = np.vstack([A, B])[None]
n = intermolecular_contacts(pc, [0, 1], [2, 3], cut=0.8)
check("접촉 2쌍", abs(n[0] - 2.0) < 1e-9, f"{n[0]:.0f} (0-2, 1-2)")

# ── 7. 항체내몫 — 심은 값을 회수하나 ────────────────────────────────────────
print("\n[7] 항체내몫")
g = np.repeat(np.arange(5), 4)
between = np.repeat(rng.normal(0, 10.0, 5), 4)      # 항체 간 분산 크게
v_between = between + rng.normal(0, 0.5, 20)
v_within = np.tile(np.arange(4) * 3.0, 5) + rng.normal(0, 0.2, 20)
sb, sw = within_share(v_between, g), within_share(v_within, g)
check("항체 간이 지배하면 몫이 작다", sb < 0.15, f"{sb:.3f}")
check("항체 안이 지배하면 몫이 크다", sw > 0.85, f"{sw:.3f}")
check("표본이 모자라면 NaN", not np.isfinite(within_share([1.0, 2.0], [0, 1])))

# ── 8. 입력 검사 ────────────────────────────────────────────────────────────
print("\n[8] 잘못된 입력은 조용히 넘어가지 않는다")
for nm, fn in (("길이 불일치", lambda: b22_from_pmf([1, 2, 3], [0, 0])),
               ("r 음수", lambda: b22_from_pmf([-1, 2, 3], [0, 0, 0])),
               ("점 부족", lambda: b22_from_pmf([1, 2], [0, 0])),
               ("pos 차원", lambda: sticky_exposure(np.zeros((3, 3)), [0], [1])),
               ("빈 패치", lambda: sticky_exposure(np.zeros((2, 3, 3)), [], [1]))):
    try:
        fn(); check(nm, False, "예외가 안 났다")
    except (ValueError, IndexError):
        check(nm, True)

print("\n" + "=" * 70)
print(f"통과 {len(OK)} · 실패 {len(BAD)}")
if BAD:
    print("실패:", BAD)
    sys.exit(1)
print("전부 통과")
