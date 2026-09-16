"""계면 친화도가 이량체 분율에 정말로 상쇄되는가 — 직접 푼다."""
import numpy as np
from scipy.optimize import brentq

def dimer_fraction(Kd, c_eff, c_eff2, Ctot):
    """M_c ⇌ M_o (분자내 계면 하나가 풀림),  2 M_o ⇌ D (분자간 계면 둘)."""
    K_open  = Kd / c_eff          # [M_o]/[M_c]
    K_assoc = c_eff2 / Kd**2      # [D]/[M_o]^2   (계면 2개)
    def bal(mc):
        mo = K_open*mc
        return mc + mo + 2*K_assoc*mo**2 - Ctot
    mc = brentq(bal, 1e-30, Ctot)
    mo = K_open*mc
    D  = K_assoc*mo**2
    return 2*D/Ctot, mo/Ctot

Ctot, c_eff, c_eff2 = 1e-5, 1e-2, 1e-2
print("① Kd 를 3.5 자릿수 쓸어도 이량체 분율이 움직이는가 (c_eff 고정)")
print(f"{'Kd (M)':>10} {'열린분율':>10} {'이량체분율':>12}")
for Kd in (1e-5, 1e-6, 1e-7, 1e-8, 3e-9):
    d, o = dimer_fraction(Kd, c_eff, c_eff2, Ctot)
    print(f"{Kd:>10.1e} {o:>10.4f} {d:>12.6f}")
print("  → 상쇄된다. [D] = (c_eff2/c_eff²)·[M_c]² 이라 Kd 가 사라진다.\n")

print("② 링커(c_eff)만 쓸면?")
print(f"{'c_eff':>10} {'열린분율':>10} {'이량체분율':>12}")
for ce in (1e-1, 1e-2, 1e-3, 1e-4):
    d, o = dimer_fraction(1e-7, ce, c_eff2, Ctot)
    print(f"{ce:>10.1e} {o:>10.4f} {d:>12.6f}")
print("  → 여기가 전부다. 링커 기하만 움직인다.\n")

print("③ ★ 그런데 **열린분율을 고정하고** 물으면? (= 우리가 하는 상호작용 질문)")
print("   [D] = (c_eff2/Kd²)·[M_o]²  에서 [M_o] 를 관측량으로 고정한다")
print(f"{'Kd (M)':>10} {'고정 열린분율':>14} {'이량체분율':>12}")
mo_fixed = 1e-7
for Kd in (1e-5, 1e-6, 1e-7, 1e-8):
    D = (c_eff2/Kd**2)*mo_fixed**2
    print(f"{Kd:>10.1e} {mo_fixed/Ctot:>14.4f} {2*D/Ctot:>12.3e}")
print("  → **조건부로는 상쇄가 안 된다.** Kd 가 작을수록(셀수록) 이량체가 제곱으로 는다.")
print("     상쇄는 [M_o] 를 Kd 의 함수로 되돌려 넣을 때만 일어난다.")
print("     우리는 [M_o] 를 BioEmu 로 **측정**하므로 그 대입을 하지 않는다.")
