#!/usr/bin/env python3
"""FvFlow v12 노트북 생성기.

왜 생성기인가 — .ipynb 는 JSON 이라 손으로 고치면 셀 하나 바꾸는 데도 위험하다.
셀을 여기 파이썬 문자열로 두고 `python3 tools/build_v12.py` 로 찍어낸다.
**노트북을 직접 고치지 말고 이 파일을 고쳐라.**

    python3 tools/build_v12.py            # notebooks/FvFlow_v12.ipynb 를 쓴다
    python3 tools/build_v12.py --check    # 모든 코드 셀이 파싱되는지만 본다
"""
import ast
import json
import os
import sys

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# ═════════════════════════════════════════════════════════════════════════════
md(r'''
# FvFlow v12 · 링커가 VH-VL 배향 **분포**를 바꾸는가, 그게 순도와 이어지는가

```
                  ┌── ABodyBuilder2 ──→ 기준점 (이상적 VH-VL 각도)
scFv ─┬─ VL / VH ─┤                        └── 계면 강도 (BSA · 접촉 · 소수성)
      │  (링커 뺌) └── AF2-Multimer ──→ ipTM  ← 두 사슬로. 분자간 짝짓기를 묻는다
      │                                          │
      └─ 링커 포함 한 서열 ──→ BioEmu ──→ 앙상블 ─┼─→ ABangle Δ(6축) · OCD5 · 열림분율
                                                 └─→ 링커 접촉 · 나선도
              합성 링커 요인패널 (조성 7 × 길이 4 + 순서대조 + 씨앗반복)
                                                  ↓
                              구성체당 특징 한 줄  →  ML  →  실측 순도
                                                     ↑
                                       계면강도 × 열림  (7절-C 상호작용)
```

**묻는 것 네 개** — 순서가 있다. 앞이 죽으면 뒤는 읽지 않는다.

1. **채널이 있는가.** BioEmu 가 링커 **조성**을 보는가, 아니면 사실상 길이만 보는가.
   → 6절-D(국소, 결정적) · 6절-C(전역)가 답한다. 실측 패널로는 원리적으로 못 답한다.
2. **신호가 있는가.** 앙상블 특징이 순도와 이어지는가.
   → 7절 ① 의 **항체 안 순열 검정**이 답한다.
3. **길이 말고 다른 게 있는가.** 앙상블이 링커 **길이**를 넘어서 무언가 더하는가.
   → 7절 ③ 의 **증분 검정**이 답한다.
4. **같은 링커인데 왜 블록마다 다른가.** 도메인 교환 이량체가 답이라면, 링커가 얼마나
   여느냐만이 아니라 **열린 계면이 얼마나 끈끈한가**가 같이 정해야 한다.
   → 7절-C 의 **계면강도 × 열림 상호작용**이 답한다. 이게 이번에 새로 들어온 축이다.

---

## v11 에서 달라진 것

### 뺀 것 (돌지 않는 코드였다)
| 뺀 것 | 이유 |
|---|---|
| ESMFold · AlphaFold2 | BBFlow 씨앗 전용. BBFlow 를 끄면 쓸 데가 없다 |
| BBFlow · gatr · gafl 설치 (약 200줄) | v112 에서 기각 — 서열을 원리적으로 안 읽는다 |
| 서열채널 시험 | 이미 `p = 0.9477` 로 음성 종결 |
| `delta` · `overlay` · `fit_both` · `_align` · `_frame` · `fw_mask` | 아무도 안 부른다 |
| `pae` · `plddt` · `dockq` · `cys_exposure` · `interface_area` · `span` | 아무도 안 부른다 |
| RMSF 5단 그림 · PC1 영화 | 그림만 나오고 판정에 안 들어간다. 숫자만 남겼다 |
| 폰트 설정 3벌 중복 | 함수 하나로 합쳤다 |

### 넣은 것
| 넣은 것 | 이유 |
|---|---|
| **4-B절 합성 요인패널** | 조성 7 × 길이 4 + 순서대조 4. **길이와 조성을 가르는 유일한 장치** |
| **씨앗 반복 (4절)** | 같은 서열을 두 번 돌린다. **모든 비교의 분모.** v11 엔 없었다 |
| **6절-D 국소 채널 시험** | `(EAAAK)ₙ` vs `(G4S)ₙ` 의 링커 나선도. 논문이 서열 민감도를 실제로 보인 자리 |
| **6절-E 표본 독립성 (ICC)** | steering 은 논문에 없는 SMC 수정판이고 배치 안 프레임을 묶는다 |
| **6절 특징표 + 두 원장** | 교란 원장(길이의 변장인가) · 잡음 원장(감쇠배율) |
| **7절 ML** | 항체 고정효과 · LOGO · 순열 검정 · **증분 검정** · **검정력 곡선** · 설계 조언 |
| **8절 대리모형 + 분산 분해** | Stage A/B 몫을 부트스트랩으로 따로 재서 찍는다 |
| **3절-B 계면 강도** | ABB2 기준 구조에서 **공짜로** — BSA · 접촉밀도 · 소수성 · 4모델 흔들림 |
| **3절-C AF2-Multimer ipTM** | VH+VL 을 **두 사슬로**. ABB2 가 구조상 못 던지는 질문이다 |
| **7절-C 계면 상호작용** | 같은 링커인데 블록마다 다른 이유. 정확 순열 · 단측 · 경쟁 설명 대조 |

### 그리고 조용히 틀린 값을 내던 자리 열셋을 고쳤다
전부 **오류를 내지 않고 그럴듯한 숫자를 돌려주던** 것들이라 더 위험했다.
ABangle 검증 CLI(무의미한 각도 6개가 조용히 나오던 것, 덤으로 412분 → 10~20분),
항체 키 영속화(v09 출력을 통째로 못 찾던 것), `태그` 키 충돌,
OCD6 의 dc 지배와 그 dc 로 표본을 고르던 순환, 실측 열이 분석 직전에 버려지던 것,
합성 구성체가 숙주의 실측값을 물고 가던 것, 독립 단위가 블록이던 것,
정규식 배향 판별, 빈 도메인 칸이 링커 색인을 밀던 것, 빈 블록 칸,
`.done` 표식 조기 기록, `matplotlib.use("Agg")` 로 그림이 한 장도 안 보이던 것,
그리고 N=150 지형의 "3 kcal/mol 등고선" 이 사실 **표본 하나짜리 잡음 바닥**이던 것.
자세한 목록은 저장소의 `README.md` 에 있다.

### 그대로 두는 것
v09 가 Drive 에 만든 것을 **하나도 다시 안 만든다.** 같은 `OUT` 을 보고,
`flow.csv` 가 있으면 그대로 읽는다. 4절은 `.done` 표식으로 건너뛴다.

---

## 사전 등록 — 결과 보기 전에 박는다

**확증 특징 1개**: `열림분율` = 앙상블에서 VH-VL 이 **안 붙은 프레임의 비율**.
HMW:Monomer 가 인구 비율이니 계산도 인구 비율이어야 한다.

**탐색 특징 3개**: `OCD6중앙` · `OCD6_MAD` · `링커접촉_잔기당` — Holm 보정해서 읽는다.

**기준선**: 링커 **길이**. 증분 검정의 귀무모형이다.

**판정 규칙**
1. 4-B절에서 조성 η² 가 길이 η² 에 견줄 만하지 않으면 → BioEmu 는 길이만 본다.
   그 뒤의 모든 상관은 "긴 링커가 나쁘다"의 다른 말이다. **그렇게 보고한다.**
2. 7절 확증 검정의 순열 p ≥ 0.05 면 → 신호 없음. 단, **검정력 곡선을 같이 읽는다** —
   못 본 것인지 없는 것인지는 거기서만 갈린다.
3. 증분 검정의 순열 p ≥ 0.05 면 → 앙상블은 길이 위에 아무것도 더하지 않았다.
   BioEmu 를 돌린 값어치가 없었다는 뜻이고, 그것도 결과다.
''')

# ═════════════════════════════════════════════════════════════════════════════
code(r'''
# ── 0절 · 드라이브 · 설치 ───────────────────────────────────────────────────
# v11 의 HF 토큰 뭉치를 뺐다. 여기 쓰는 모델(esmfold 는 이제 안 쓰고 ABB2·BioEmu 는
# 공개)에 토큰이 필요 없고, Colab getpass 가 dict 를 돌려주는 사고만 남았었다.
from google.colab import drive
drive.mount("/content/drive")

import os, sys, shutil, types

# anarci 는 ImmuneBuilder 의 import 시점 의존이고, anarci 는 hmmscan 바이너리를 쓴다
!apt-get -qq install -y hmmer > /dev/null
!pip -q install ImmuneBuilder anarci biopython openpyxl uv scikit-learn 2>&1 | tail -1
# ★ ABangle·DockQ 를 여기 깔면 안 된다. DockQ 가 numpy<2.0 을 끌고 와서
#   런타임 numpy 를 내리고 pandas 가 깨진다. 2절이 별도 3.11 환경에 넣는다.

# ImmuneBuilder 는 import 시점에 openmm/pdbfixer 를 끌어온다. 둘 다 **refine 전용**이고
# refine 은 Cα 배향을 바꾸지 않는다. 없으면 껍데기를 끼우고 refine 없이 저장한다.
class _Any:
    def __init__(self, n="stub"): self._n = n
    def __getattr__(self, k): return _Any(f"{self._n}.{k}")
    def __call__(self, *a, **k): return _Any(self._n + "()")
    def __truediv__(self, o): return self
    __rtruediv__ = __mul__ = __rmul__ = __pow__ = __truediv__
    def __getitem__(self, k): return self
class _StubMod(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"): raise AttributeError(k)     # 던더를 가로채면 import 가 무한루프
        if k.endswith(("Exception", "Error")): return type(k, (Exception,), {})
        return _Any(f"{self.__name__}.{k}")
for m in ("pdbfixer", "openmm", "openmm.app", "openmm.unit"):
    if m not in sys.modules:
        try: __import__(m)
        except Exception: sys.modules[m] = _StubMod(m)

# **여기서 다 불러본다.** 4절까지 가서 터지지 말고 지금 터져라.
print("hmmscan:", shutil.which("hmmscan") or "★없음 — anarci 가 죽는다")
print("uv      :", shutil.which("uv") or "★없음 — ABangle·BioEmu 환경이 안 깔린다")
import numpy as _np
if _np.__version__.startswith("1."):
    print(f"★ numpy 가 {_np.__version__} 로 내려가 있다 (DockQ 가 내렸을 것이다).")
    print("  되돌린 뒤 **런타임을 재시작**하고 이 셀부터 다시 돌려라.")
    !pip -q install -U "numpy>=2.0,<2.6" 2>&1 | tail -1
for mod in ("numpy", "pandas", "scipy", "sklearn", "Bio", "anarci", "ImmuneBuilder"):
    try: __import__(mod); print(f"  {mod:<14} OK")
    except Exception as e: print(f"  {mod:<14} ★{type(e).__name__}: {str(e)[:60]}")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 1절 · 경로 · 스위치 · 데이터

`OUT` 은 **v09 가 쓰던 그 폴더**다. 이미 만든 앙상블·`flow.csv` 를 그대로 읽는다.

### 오늘 밤 도는 것 — 스위치 넷이 정한다
```
RUN_ABB2   = True    기준점. ΔABangle 의 영점 (항체당 ~30초)
RUN_BIOEMU = True    실측 26건. v09 가 끝냈으면 .done 표식으로 전부 건너뛴다
RUN_PANEL  = True    ★ 합성 요인패널 24건. 이 노트북에서 제일 값진 셀
RUN_ML     = True    분석. GPU 도 Drive 도 안 쓴다 — 언제든 혼자 다시 돌려도 된다
```

**왜 합성 패널이 제일 값진가.** 실측 패널은 링커마다 길이가 하나씩이라
길이와 조성이 **영원히** 얽혀 있다. 무슨 통계를 써도 못 푼다.
같은 길이에서 조성만 바꾼 칸이 있어야 풀린다 — 그게 합성 패널이다.
24건 × 10~16분 ≈ **4~6시간**, 하룻밤이면 끝난다.
''')

code(r'''
# ── 1절 · 경로 · 스위치 · 예산 ─────────────────────────────────────────────
import os, glob, time, json, numpy as np, pandas as pd

DRIVE = "/content/drive/MyDrive/FvTwist"
IN    = f"{DRIVE}/input"
OUT   = f"{DRIVE}/fvflow"          # ★ v09 와 **같은 폴더**. 만든 것을 그대로 읽는다.
for d in (IN, OUT): os.makedirs(d, exist_ok=True)

XL_TEST = None        # None 이면 IN 의 test_result*.xlsx 중 최신
BLOCKS  = "all"       # "all" = 실측 26건 전부. [1] 처럼 리스트를 주면 그 블록만.

# ── 무엇을 돌릴까 ───────────────────────────────────────────────────────────
RUN_ABB2   = True     # 기준점. ΔABangle 의 영점이라 **반드시** 필요하다
RUN_BIOEMU = True     # 실측 패널. v09 가 끝냈으면 전부 건너뛴다
RUN_PANEL  = True     # ★ 합성 요인패널 (조성 6 × 길이 4). 채널 판정의 유일한 장치
RUN_ML     = True     # 6~8절 분석. GPU 불필요 — 따로 다시 돌려도 된다

# ★ 이미 만들어 둔 표(interface.csv · frames.csv · geom.csv)를 읽어 쓸까.
#   3절-B 가 제일 먼저 참조하므로 **여기** 1절에 둔다 — 5절에 두면 위에서부터
#   차례로 돌릴 때 3절-B 가 NameError 로 죽는다.
REUSE_CSV  = True     # False 로 두면 표를 무시하고 전부 다시 잰다

# ── CALVADOS — **두 사슬**을 실제로 만나게 한다 (10절) ─────────────────────
#   BioEmu 는 단일 사슬만 다룬다. HMW 는 둘이 붙은 결과다. 그 사이가 비어 있었다.
#   그리고 단일사슬 기술자로는 무엇을 정의해도 길이의 함수로 환원된다 —
#   `링커신장도` 가 길이와 rho −0.99 였다. 두 사슬을 만나게 해야 새 축이 나온다.
RUN_CALVADOS = True   # 10절. GPU 권장 · (구성체 × 형태)당 수십 분
CALV_IONIC   = 0.15   # 이온강도 (M). **실제 제형에 맞춰라** — Whitlow218 의 K/E 가
                      #   여기서 실제로 작동한다 (Debye 스크리닝이 명시적이다)
CALV_PH      = 7.4    # 전하 결정용
CALV_TEMP    = 298.15 # K
CALV_NCHAIN  = 2      # 2 = B22 용 한 쌍. 늘리면 묽은상 회합까지 본다
CALV_STEPS   = 2_000_000
# ★ 먼저 **몇 건만** 돌려 시간을 재라. 28구성체 × 6구조 = 168건이고, 2M 스텝짜리를
#   CPU 로 하나씩 돌리면 며칠이다. None 이면 전부 — 시간을 재고 나서 풀어라.
CALV_MAX_RUNS = 6     # 이번 실행에서 돌릴 최대 시뮬레이션 수 (None = 제한 없음)
CALV_ONLY     = None  # 예: ("실측4_B1_LH", "G4S") 로 한 구성체만
CALV_PATCH_Q = 0.25   # '끈끈한 패치' = 소수성 상위 25% 표면 잔기
# ★ 어떤 앙상블 구조를 강체로 넣을까 — 이게 10절에서 제일 중요한 선택이다.
#   CALVADOS 3 은 접힌 부분을 **입력 구조로 구속**하므로, VH+VL 을 통째로 강체로
#   두면 **어느 프레임을 넣느냐가 답을 정한다.** 그래서 닫힌 것과 열린 것을 둘 다
#   넣고 섞는다 (아래 b22_mixture). 프레임을 여러 개 쓰는 이유는 잡음 바닥 때문이다 —
#   구성체 간 차이가 프레임 선택 잡음보다 작으면 우리가 재는 것은 잡음이다.
CALV_N_CLOSED = 3     # 닫힌 대표 구조 개수
CALV_N_OPEN   = 3     # 열린 대표 구조 개수 (튀는 것들)
CALV_OPEN_Q   = 0.90  # '열림' = 그 구성체 도메인Rg 의 상위 10%

# ── 사전 등록 임계 — 결과 보기 전에 박는다 ─────────────────────────────────
# 짝지음 판정 — **절대 dc** 로 한다. Δdc(기준점 대비)로 하면 기준점이 특이한 항체에서
# '붙었다' 의 뜻이 구성체마다 달라진다.
# SAbDab 1296구조에서 dc 는 평균 16.21 Å · σ 0.268 Å · 전 범위 15.14~17.68 Å 다.
# 6σ = 1.6 Å 이면 자연 전 범위를 덮는다 — 그보다 벌어지면 '기울어진 Fv' 가 아니라
# **안 붙은 Fv** 이고, ABangle 의 각도 4개는 붙은 계면을 전제로 정의된 양이라 무의미해진다.
DC0, DC_SD = 16.206, 0.268
PAIR_NSD   = 6.0      # |dc − DC0| < 6σ = 1.6 Å
PAIR_DC    = PAIR_NSD * DC_SD     # 옛 이름 호환 (Å)
N_PERM   = 2000       # 순열 횟수. 블록당 4! = 24, 5블록이면 24⁵ ≈ 800만 → 2000 이면 충분
ALPHA    = 0.05

# ── 밤새 돌리기 위한 시간 예산 ──────────────────────────────────────────────
DEADLINE_H = 8.0
T0 = time.time()
def left():   return DEADLINE_H*3600 - (time.time() - T0)
def elapsed(): return f"[{(time.time()-T0)/60:.0f}분]"
def budget(need_s, what=""):
    """남은 예산이 need_s 보다 적으면 False. 그 자리에서 이유를 찍는다."""
    if left() >= need_s: return True
    print(f"  ⏱ 예산 소진 ({left()/60:.0f}분 남음, {need_s/60:.0f}분 필요) — {what} 중단."
          f" 이미 만든 것은 남아 있으니 내일 이어서 돌리면 된다.")
    return False

FL   = lambda r: f"{OUT}/{r.항체}/flow"       # 앙상블이 쌓이는 곳
REFD = lambda ab: f"{OUT}/{ab}"               # 기준점이 쌓이는 곳
safe = lambda x: "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(x))

print(f"출력 {OUT}  (v09 와 같은 폴더 — 이미 만든 것은 다시 안 만든다)")
print(f"블록 {BLOCKS} · 짝지음 임계 {PAIR_DC} Å · 순열 {N_PERM} · 예산 {DEADLINE_H} h")
print("  켜짐: " + ", ".join(k for k, v in
      [("ABB2", RUN_ABB2), ("BioEmu", RUN_BIOEMU),
       ("합성패널", RUN_PANEL), ("ML", RUN_ML),
       ("CALVADOS", RUN_CALVADOS)] if v))
''')

# ═════════════════════════════════════════════════════════════════════════════
code(r'''
# ── 1절-B · 엑셀 → 구성체 표 CONS ──────────────────────────────────────────
# Domain_i · Linker_i · Domain_{i+1} 을 Fv 단위로 끊는다. v09 와 같은 로더다 —
# **항체 키와 링커 라벨이 v09 와 똑같이 나와야** Drive 의 파일을 찾을 수 있다.
import re

# FR4 모티프. 중쇄는 W-G-x-G, 경쇄는 F-x-x-G — 첫 글자로 이미 갈린다.
# (실측 VL 은 FAGG 다. 고정 문자열 목록이면 놓친다.)
FR4_H = re.compile(r"W[GAS][QKRAHPSGE]G[TQAS]")
FR4_L = re.compile(r"F[GAQSN][GQSAPTE]G[TQAS]")
NT_H  = re.compile(r"^[EQADVSKG][VILM][QKMET]L")         # EVQL/QVQL/QIQL/EVKL...
NT_L  = re.compile(r"^[DEQNSAY][ISVFTA][VQTGE][LMVAT]")  # DIQM/DIVL/EIVL/QSAL/SYEL...
ORIENT = {}     # 자동판별이 틀리면 여기에 적는다. 예: {"Sample01_G4S4": "LH"}

_KIND = {}
def kind(s):
    """중쇄/경쇄 판별. **ANARCI 가 1순위**, 정규식은 ANARCI 가 도메인을 못 잡을 때만.

    ★ 여기서 틀리면 조용히 모든 것이 무너진다. 배향이 뒤집히면
      (a) 3절이 ABodyBuilder2 에 VL 을 H 로 준다 → 기준점 자체가 엉터리가 되거나
          예외로 죽고, 그 항체의 모든 프레임이 5절에서 말없이 사라진다.
      (b) write_pair 가 사슬 라벨을 반대로 붙인다 → ABangle 안에서 사슬 이름이
          겹쳐 ValueError 가 나고, 그 구성체가 통째로 frames.csv 에서 빠진다.
    ANARCI 는 이미 필수 의존이다 (ABangle 내부도 쓴다). 정규식으로 도박할 이유가 없다."""
    if s in _KIND: return _KIND[s]
    k = "?"
    try:
        from anarci import run_anarci
        _, _, det, _ = run_anarci([("x", s)], scheme="chothia")
        if det and det[0]:
            ct = det[0][0]["chain_type"]
            k = "H" if ct == "H" else ("L" if ct in ("K", "L") else "?")
    except Exception:
        pass
    if k == "?":                                  # ANARCI 가 못 잡을 때만 정규식
        t = s[-22:]
        h, l = bool(FR4_H.search(t)), bool(FR4_L.search(t))
        if h == l: h, l = bool(NT_H.match(s)), bool(NT_L.match(s))
        if h != l: k = "H" if h else "L"
    _KIND[s] = k
    return k

def load(path):
    """엑셀 → 구성체 표."""
    if not path or not os.path.isfile(path):
        print(f"  건너뜀: {path}"); return pd.DataFrame()
    X = pd.read_excel(path)
    X.columns = [str(c).strip() for c in X.columns]
    col = lambda p: next((c for c in X.columns if c.lower().replace(" ", "") == p), None)
    nm  = col("이름") or col("name") or col("sample") or X.columns[0]
    obs = [c for c in X.columns if any(k in c.lower() or k in c
           for k in ("hmw", "monomer", "수율", "생산", "titer", "sec",
                     "quantific", "supernatant"))]
    blk  = col("block") or col("블록")      # **실험 세트.** 세트마다 절대값이 다르다
    tcol = col("type") or col("종류")       # ScFv 단독인지 mAb 융합인지
    dcol = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("domain")],
                  key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))
    lcol = sorted([c for c in X.columns if c.lower().replace(" ", "").startswith("linker")],
                  key=lambda c: int("".join(f for f in c if f.isdigit()) or 0))
    print(f"  [{os.path.basename(path)}] 도메인열 {dcol} · 링커열 {lcol} · 실측열 {obs}")
    rows = []
    for _, r in X.iterrows():
        # ★ 도메인을 **원래 열 위치로** 들고 있어야 한다. v11 은 빈 칸을 걸러 내면서
        #   목록을 압축했다 — Domain_2 가 비어 있으면 Domain_1 과 Domain_3 이 이웃이 되어
        #   Linker_1 과 짝지어졌다. 서열이 조용히 틀려지고 b1/b2 경계도 같이 틀어진다.
        D = {i: str(r[c]).strip().upper() for i, c in enumerate(dcol)
             if isinstance(r[c], str) and len(str(r[c])) > 50}
        L = {i: (str(r[c]).strip().upper() if isinstance(r[c], str) else "")
             for i, c in enumerate(lcol)}
        base = str(r[nm]).strip()
        if not D:
            print(f"    {base}: 도메인 없음 (50aa 넘는 서열 칸이 없다)"); continue
        K = {i: kind(d) for i, d in D.items()}
        for i in sorted(D):
            if (i + 1) not in D:                       # **이웃한** 도메인 쌍만
                continue
            d1, d2 = D[i], D[i+1]
            od = ORIENT.get(base) or (K[i] + K[i+1])
            if set(od) != {"H", "L"}:                       # Fv 짝만
                print(f"    {base}#{i+1}: 건너뜀 (판별 {K[i]}{K[i+1]}) "
                      f"D{i+1} …{d1[-12:]} / D{i+2} …{d2[-12:]}")
                continue
            tag = base if len(D) == 2 else f"{base}#{i+1}"
            _t  = str(r[tcol]).strip().lower() if tcol and pd.notna(r.get(tcol)) else ""
            fmt = "순수" if (_t == "scfv" or (not _t and re.search(r"H{5,}$", d2))) else "융합"
            rows.append(dict(행=tag, 블록=(r[blk] if blk else 1), 포맷=fmt,
                             링커=base, 링커서열=L.get(i, ""),
                             D1=d1, D2=d2, 배향=od,
                             **{o: r[o] for o in obs if pd.notna(r.get(o))}))
    return pd.DataFrame(rows)

if XL_TEST is None:
    _c = sorted(glob.glob(f"{IN}/*test_result*.xls*"))
    XL_TEST = _c[-1] if _c else None
    print("실측 파일:", os.path.basename(XL_TEST) if XL_TEST else "★없음")
CONS = load(XL_TEST)
assert len(CONS), "구성체가 0개다 — 위 진단을 보고 엑셀 열 이름이나 ORIENT 를 고쳐라"

# 링커 라벨 = 이름의 마지막 조각 (K_Sample01_G4S → G4S). 블록끼리 같은 라벨이 맞춰진다
CONS["링커"] = CONS.행.astype(str).str.rsplit("_", n=1).str[-1].map(safe)

# ★ 블록이 비면 groupby 가 그 행을 **조용히 버린다.** 여기서 막는다.
CONS["블록"] = CONS.블록.fillna(1)
CONS["블록"] = CONS.블록.map(lambda v: str(int(v)) if isinstance(v, float) and v == int(v)
                            else str(v)).astype(str)
assert CONS.블록.notna().all() and (CONS.블록 != "nan").all(), "블록에 빈 칸이 있다"

# ── 항체 키 — **v09 가 만든 Drive 폴더 이름이다. 절대 바뀌면 안 된다.** ──────
# 키는 `groupby(D1+D2).ngroup()` 으로 매겨지는데, ngroup 은 **정렬 순서로 번호를 다시
# 매긴다.** 엑셀에 행이 하나 늘거나, D1/D2 파싱이 조금이라도 바뀌거나, 배향 판별이
# 달라지면 **모든 항체 번호가 밀린다.** 그러면 v09 가 만들어 둔
#   {OUT}/{항체}/flow/{링커}__BioEmu_s*.pdb
# 를 통째로 못 찾고, 조용히 빈 폴더를 보며 "처음부터 다시" 를 시작한다.
# → 한 번 매긴 키를 Drive 에 적어 두고, 그 다음부터는 **그것을 정답으로 삼는다.**
_KEYMAP = f"{OUT}/antibody_keys.csv"
_kc = ["D1", "D2", "블록", "배향"]
_fresh = ("실측" + (CONS.groupby(CONS.D1 + CONS.D2).ngroup() + 1).astype(str)
          + "_B" + CONS.블록.astype(str) + "_" + CONS.배향)
if os.path.isfile(_KEYMAP):
    _km = pd.read_csv(_KEYMAP, dtype={"블록": str})
    CONS = CONS.merge(_km[_kc + ["항체"]].drop_duplicates(), on=_kc, how="left")
    _new = CONS.항체.isna()
    if _new.any():
        CONS.loc[_new, "항체"] = _fresh[_new.values]
        print(f"  ★ 키맵에 없는 항체 {int(_new.sum())}행 — 새로 매겼다. "
              f"v09 출력이 있다면 이 이름으로는 못 찾는다.")
    print(f"  항체 키를 {os.path.basename(_KEYMAP)} 에서 읽었다 (v09 이름을 그대로 쓴다)")
else:
    CONS["항체"] = _fresh
    print(f"  항체 키를 새로 매기고 {os.path.basename(_KEYMAP)} 에 적는다")
CONS[_kc + ["항체"]].drop_duplicates().to_csv(_KEYMAP, index=False, encoding="utf-8-sig")
# ★ (항체, 링커) 는 파일명 키다. 비었거나 겹치면 **네 구성체가 한 파일로 뭉개진다.**
bad = (CONS.링커.isna() | CONS.링커.astype(str).isin(["nan", "None", ""])
       | CONS.duplicated(["항체", "링커"], keep=False))
CONS.loc[bad, "링커"] = "L" + CONS.loc[bad, "링커서열"].str.len().astype(str)
dup = CONS.duplicated(["항체", "링커"], keep=False)
if dup.any():
    CONS.loc[dup, "링커"] = (CONS.loc[dup, "링커"].astype(str) + "_"
        + (CONS[dup].groupby(["항체", "링커"]).cumcount() + 1).astype(str))
if int(bad.sum()):
    print(f"  ★ 링커 라벨 {int(bad.sum())}개를 길이 기반으로 대체했다")
assert not CONS.duplicated(["항체", "링커"]).any(), "(항체,링커) 가 여전히 겹친다"

CONS = CONS.drop(columns=["행"])
CONS["seq"] = CONS.D1 + CONS.링커서열 + CONS.D2
CONS["b1"]  = CONS.D1.str.len()                        # D1 = 1..b1
CONS["b2"]  = CONS.b1 + CONS.링커서열.str.len()         # D2 = b2+1..end
# ★ 경계가 서열과 맞는지 검산한다. 여기가 틀리면 write_pair 가 엉뚱한 자리를 자르고
#   ABangle 이 '그럴듯한' 각도를 돌려준다 — 오류가 안 나므로 끝까지 모른다.
assert (CONS.seq.str.len() == CONS.b1 + CONS.링커서열.str.len()
        + CONS.D2.str.len()).all(), "seq 와 b1/b2 가 안 맞는다"
assert (CONS.b2 <= CONS.seq.str.len() - 30).all(), "링커 뒤 도메인이 30aa 미만이다"
CONS["길이"] = CONS.링커서열.str.len()
CONS["합성"] = False
# 배향 효과는 **같은 Fv(도메인쌍)** 안에서만 판정할 수 있다. 정제태그를 떼고 짝으로 묶는다
_ht = lambda x: re.sub(r"(GGGGS)?H{5,}$", "", str(x))
CONS["Fv키"] = ["Fv%d" % i for i in pd.factorize(
    pd.Series([tuple(sorted((_ht(a), _ht(b)))) for a, b in zip(CONS.D1, CONS.D2)]))[0]]

# ★ 블록 선택은 **항체 키를 매긴 뒤에** 한다 — 순서가 바뀌면 ngroup 이 달라져
#   v09 가 만든 파일 이름과 안 맞는다.
if BLOCKS != "all":
    CONS = CONS[CONS.블록.isin(list(BLOCKS))].reset_index(drop=True)
assert len(CONS), f"블록 {BLOCKS} 에 구성체가 없다"

# ── 실측 열 — **한 곳에서만 정한다** ────────────────────────────────────────
# v11 은 이 목록을 세 곳에서 서로 다르게 만들었다: 로더는 8개 키워드로 찾고,
# 합성 구성체 비우기는 3개로 비우고, 분석은 'hmw' 하나만 집어 갔다.
# 그래서 Monomer·Titer·SEC 같은 열이 **로드되고 표에 찍히고 나서 분석 직전에 버려졌다** —
# 사용자가 재려던 '순도' 가 분석에 도달한 적이 없다.
OBS_KEYS = ("hmw", "monomer", "단량체", "purity", "순도", "수율", "생산",
            "titer", "sec", "quantific", "supernatant")
OBS = [c for c in CONS.columns if any(k in str(c).lower() or k in str(c)
                                      for k in OBS_KEYS)]
# 확증 검정에 쓸 열 하나. HMW 계열이 있으면 그것, 없으면 첫 실측 열.
_hm = [c for c in OBS if "hmw" in str(c).lower()]
HC  = (_hm[0] if _hm else (OBS[0] if OBS else None))
HIGHER_IS_WORSE = bool(_hm)      # HMW(응집체 %)는 낮을수록 좋다. 순도는 반대다.
if len(OBS) > 1:
    print(f"  실측 열 {len(OBS)}개 발견: {OBS}")
    print(f"    → 확증 검정은 **{HC}** 하나로 한다 (사전 등록). 나머지는 탐색이다.")
print(f"\n▶ 구성체 {len(CONS)} · 항체 {CONS.항체.nunique()} · 블록 {sorted(CONS.블록.unique())}")
print(f"  실측 열: {HC or '★없음 — 7절이 안 돈다'}")
# ★ v09 가 만든 것을 실제로 찾을 수 있나 — **여기서 확인한다.** 분석까지 가서
#   "구성체 0개" 를 만나는 것보다 지금 아는 편이 낫다.
_dirs = {ab: os.path.isdir(f"{OUT}/{ab}") for ab in CONS.항체.unique()}
_nf = [ab for ab, ok in _dirs.items() if not ok]
_nb = sum(1 for _, r in CONS.iterrows()
          if glob.glob(f"{FL(r)}/{r.링커}__BioEmu_s*.pdb"))
print(f"  v09 출력 — 항체 폴더 {sum(_dirs.values())}/{len(_dirs)} · "
      f"BioEmu 앙상블이 있는 구성체 {_nb}/{len(CONS)}")
if _nf: print(f"    ★ 폴더 없음: {_nf[:6]}{' …' if len(_nf) > 6 else ''}")
if _nb == 0:
    print("    ★ 앙상블이 하나도 없다. 항체 키가 v09 와 다르거나 4절을 아직 안 돌렸다.")
    print(f"      {OUT} 아래 실제 폴더: {sorted(os.listdir(OUT))[:8] if os.path.isdir(OUT) else '없음'}")
_bs = CONS.groupby("블록").링커.nunique()
print(f"  링커 3종 이상 블록(= ML 이 쓰는 독립 단위): {sorted(_bs[_bs>=3].index)} "
      f"→ **{int((_bs>=3).sum())}개**")
print(f"  링커 1~2종 블록: {sorted(_bs[_bs<3].index)}  (배향 비교에만 쓰인다)")
display(CONS.groupby(["블록", "항체"]).agg(
    링커수=("링커", "nunique"), 링커=("링커", lambda x: list(x)),
    길이=("길이", lambda x: sorted(set(x)))).reset_index())
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 2절 · 함수

v11 의 2절은 21,800자에 함수가 서른 개였고 그중 **스물을 아무도 부르지 않았다**
(`delta`·`overlay`·`fit_both`·`_align`·`_frame`·`fw_mask`·`pae`·`plddt`·`dockq`·
`cys_exposure`·`interface_area`·`span`·`linker_stats`·`_sub`·`read_ca`·`split_models`·
`_ca`·`_gap`·`venv_bb`·`_gatr`). BBFlow·ESMFold·AF2 를 끄면서 호출부가 통째로 사라졌는데
함수만 남아 있었다.

**남은 것은 넷뿐이다.**

| 함수 | 하는 일 |
|---|---|
| `write_pair` | 구조 → H/L 두 사슬 PDB. ABangle 이 이 형식을 요구한다 |
| `venv_metrics` · `abangle6_many` | 별도 3.11 환경에서 ABangle 을 **배치로** 돌린다 |
| `sh` · `venv311` | 하위 프로세스 · 환경 |
| `kabsch` · `ca_xyz` | 좌표 정렬 · PDB 읽기 (6절 기하와 9절 그림이 쓴다) |

★ **`abangle6_many` 를 파일 하나씩 부르면 안 된다.** 인터프리터 기동 + abangle import
비용이 파일 수만큼 든다. 앙상블이 5,000개를 넘으면 그것만 수십 분이다.
한 프로세스가 목록을 받아 한 줄씩 뱉게 한다.
''')

code(r'''
# ── 2절 · 함수 (v11 의 30개 → 필요한 것만) ─────────────────────────────────
import subprocess, shutil
import scipy.stats as st_
from Bio.PDB import PDBParser, MMCIFParser
import warnings; warnings.filterwarnings("ignore")

def sh(cmd, tail=14):
    """외부 명령 실행. **실패하면 이유를 찍는다** — 조용히 '실패'만 쓰지 않는다."""
    # Colab 의 MPLBACKEND=module://matplotlib_inline… 이 하위 프로세스로 새면
    # 그 venv 에는 matplotlib_inline 이 없어 **import 단계에서 죽는다**
    env = dict(os.environ, MPLBACKEND="Agg"); env.pop("PYTHONPATH", None)
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    except (FileNotFoundError, PermissionError) as e:
        print(f"    실행 불가: {cmd[0]}  ({type(e).__name__})"); return False
    if p.returncode:
        print(f"    rc={p.returncode}  {' '.join(str(c) for c in cmd[:3])} …")
        for l in (p.stderr or p.stdout or "").strip().split("\n")[-tail:]:
            print("      |", l)
    return p.returncode == 0

def venv311(V="/content/venv311"):
    """**Colab 은 Python 3.13 이다.** ABangle·BioEmu 는 3.11 을 요구한다.
    uv 로 따로 만들고 하위 프로세스로 부른다 — 절 사이 인터페이스가 구조 파일이라
    인터프리터가 달라도 아무 문제가 없다."""
    if not os.path.isdir(V):
        assert sh(["uv", "venv", "--python", "3.11", V]), "uv venv 실패"
    return V

_parser = lambda p: MMCIFParser(QUIET=True) if p.endswith(".cif") else PDBParser(QUIET=True)

def ca_xyz(path):
    """PDB → CA 좌표 (n,3) Å. BioEmu 출력은 백본만이라 문자열 파싱이 제일 빠르다."""
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path)
                     if l.startswith("ATOM") and l[12:16].strip() == "CA"])

def kabsch(P, Q):
    """P 를 Q 에 겹치는 회전 R. 적용은 (P - P.mean) @ R + Q.mean."""
    U, S, Vt = np.linalg.svd((P - P.mean(0)).T @ (Q - Q.mean(0)))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

def first_model(path):
    """구조 파일의 첫 MODEL. **못 읽으면 None** 을 준다.

    ★ 빈 파일이나 잘린 파일이 들어오면 Biopython 은 모델을 하나도 안 내놓고
      next() 가 **메시지 없는 StopIteration** 을 던진다. 루프 한가운데서 노트북이
      이유 없이 죽은 것처럼 보이는 사고가 여기서 난다 — ABB2 나 BioEmu 가 한 건
      실패해 0바이트 PDB 를 남기면 실제로 그렇게 된다. 한 구조가 못 읽혔다는 것과
      노트북이 못 돈다는 것은 다른 일이므로, 여기서 갈라 준다.
    """
    try:
        return next(_parser(path).get_structure("x", path).get_models())
    except Exception:
        return None

def write_pair(path, b1, b2, out, d1ch=None, od="HL"):
    """구조를 **H/L 두 사슬 PDB** 로 다시 쓴다 (링커 구간은 버린다).
    ABangle 이 이 형식을 요구한다. 단일사슬(BioEmu)이면 b1/b2 로 자른다.

    ★ 기준점(ABB2, 2사슬)과 구성체(BioEmu, 1사슬)가 **같은 이 함수**를 지난다.
      둘 다 사슬 안에서 1 부터 다시 번호를 매기므로 번호 규약이 같다.
      한쪽만 다른 경로로 쓰면 Δ 에 구성체마다 다른 상수 치우침이 생긴다.
    """
    st = first_model(path)
    if st is None:
        raise ValueError(f"구조를 못 읽었다 (비었거나 잘렸다): {path}")
    ch = list(st)
    if not ch:
        raise ValueError(f"사슬이 하나도 없다: {path}")
    if len(ch) >= 2:
        ch = sorted(ch, key=lambda c: -len(list(c)))[:2]
        if d1ch and any(c.id == d1ch for c in ch):
            ch = sorted(ch, key=lambda c: c.id != d1ch)
        g1 = [r for r in ch[0] if "CA" in r]; g2 = [r for r in ch[1] if "CA" in r]
    else:
        rs = [r for r in ch[0] if "CA" in r]; g1, g2 = rs[:b1], rs[b2:]
    lab = {od[0]: g1, od[1]: g2}                  # D1 은 배향의 첫 글자, D2 는 둘째
    with open(out, "w") as f:
        k = 0
        for cid in ("H", "L"):
            for i, r in enumerate(lab[cid], 1):
                for a in r:
                    k += 1
                    x, y, z = a.coord
                    nm = a.get_name()
                    nm = f" {nm:<3}" if len(nm) < 4 else nm    # 원자명은 14열부터 (PDB 규격)
                    f.write(f"ATOM  {k:>5} {nm} {r.get_resname():>3} {cid}{i:>4}    "
                            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                            f"{a.element:>2}\n")
            f.write("TER\n")
        f.write("END\n")
    return out

# ★★ 이 CLI 가 v12 에서 제일 중요한 수정이다. 읽고 나서 바꿔라.
#
# ABangle 의 `find_angles(path)` 를 그냥 부르면 **조용히 틀린 각도**가 나올 수 있다.
# 상류 `abangle/number.py::number_sequences` 에 zip 어긋남이 있다:
#
#     numbering = [... for num in numbering if num]          # 걸러서 길이가 준다
#     details   = [det[0] for num, det in zip(numbering, details) if num]
#                                        ↑ **이미 걸러진** 목록과 zip 한다
#
# ANARCI 가 L 만 알아보고 H 를 못 알아보면 numbering=[L_num] 인데 details 는
# 아직 [H_det, L_det] 라서, L 의 번호가 H 의 details 와 짝지어진다.
# → 사슬 H 에 **경쇄 번호**가 매겨지고 사슬 L 은 아예 번호가 안 매겨진다.
#
# 우리에게 왜 치명적인가: write_pair 가 사슬마다 1..N 으로 **빈틈없이** 번호를 매기므로,
# 번호가 안 매겨진 사슬에도 coreset 정수(H 35개, L 35개)가 전부 들어 있다.
# 그래서 get_coreset_atoms 가 정확히 35개를 돌려주고, Superimposer 가 성공하고,
# **그럴듯하지만 완전히 무의미한 여섯 개의 실수**가 나온다.
# (원래 Chothia 번호 파일이었다면 길이 불일치로 시끄럽게 죽었을 것이다.
#  우리 번호 매김이 요란한 실패를 조용한 실패로 바꿔 놓았다.)
#
# → 그래서 find_angles 를 맨몸으로 믿지 않고 **두 사슬이 제대로 인식됐는지 직접 확인**한다.
#   덤으로 파싱을 두 번 하지 않고 ANARCI 를 서열로 메모이즈해서 훨씬 빨라진다
#   (v11 의 412분짜리 ABangle 통과가 여기서 10~20분으로 내려간다).
_AB_CLI = r"""import sys, json, warnings
warnings.filterwarnings("ignore")
from abangle.calculate import find_angles
from abangle import number as _num

_MEMO = {}
_orig = _num.number_sequences
def _checked(seqs, scheme="chothia", **kw):
    key = (tuple(sorted(seqs.items())) if isinstance(seqs, dict) else str(seqs), scheme)
    if key in _MEMO:
        r = _MEMO[key]
        if isinstance(r, Exception): raise r
        return r
    try:
        out = _orig(seqs, scheme=scheme, **kw)
    except Exception as e:
        _MEMO[key] = e; raise
    # ★ 여기가 방어선. 넣은 사슬이 전부 돌아왔는지, 사슬 종류가 맞는지 본다.
    want = set(seqs) if isinstance(seqs, dict) else None
    if want is not None:
        got = set(out)
        if got != want:
            e = RuntimeError("ANARCI 가 사슬 %s 중 %s 만 인식했다 — 번호가 어긋난다"
                             % (sorted(want), sorted(got)))
            _MEMO[key] = e; raise e
        for cid, v in out.items():
            ct = getattr(v, "chain", None) or (v.get("chain") if isinstance(v, dict) else None)
            if ct is not None and str(ct).upper() != str(cid).upper():
                e = RuntimeError("사슬 %s 가 %s 로 인식됐다 — H/L 라벨이 뒤바뀌었다" % (cid, ct))
                _MEMO[key] = e; raise e
    _MEMO[key] = out
    return out
_num.number_sequences = _checked
try:
    import abangle.calculate as _c
    if hasattr(_c, "number_sequences"): _c.number_sequences = _checked
except Exception: pass

args = sys.argv[1:]
if args and args[0].startswith("@"):                 # 목록 파일 (argv 길이 한계 회피)
    args = [l.strip() for l in open(args[0][1:]) if l.strip()]
for p in args:
    try:
        d = {k: round(float(v), 3) for k, v in find_angles(p).items()}
        if len(d) < 6: d = {"__err__": "각도가 %d 개뿐이다" % len(d)}
    except Exception as e:
        d = {"__err__": "%s: %s" % (type(e).__name__, str(e)[:160])}
    print(p + "\t" + json.dumps(d), flush=True)
"""

def venv_metrics(V="/content/venv_metrics"):
    """ABangle 은 git 설치가 필요하고 `data/` 가 휠에 안 들어 있어 **저장소를 그대로
    둬야** 한다 (`data_path = __file__.parent.parent/'data'`)."""
    if os.path.isfile(f"{V}/ab.py") and os.path.isdir(f"{V}/ABangle"): return V
    venv311(V)
    sh(["uv", "pip", "install", "--python", f"{V}/bin/python", "-q",
        "fastcore", "anarci", "biopython", "numpy<2.0", "pandas"])
    if not os.path.isdir(f"{V}/ABangle"):
        sh(["git", "clone", "-q", "--depth", "1",
            "https://github.com/jaredsampson/ABangle.git", f"{V}/ABangle"])
    open(f"{V}/ab.py", "w").write(_AB_CLI)
    return V

# ABangle 저장소 All_Angles.dat (SAbDab 1296 구조) 에서 잰 **실제 항체의 자연 변이 폭**.
# Δ 를 이걸로 나누면 "자연 변이의 몇 배인가" 가 된다 (IgFold/RosettaAntibody 의 OCD 방식).
AB_SD = dict(HL=3.92, HC1=2.18, HC2=3.07, LC1=2.49, LC2=2.25, dc=0.268)
AB6   = ["HL", "HC1", "HC2", "LC1", "LC2", "dc"]

_AB = {}
def abangle6_many(paths, chunk=400, quiet=False):
    """**ABangle** (Dunbar/Deane, PEDS 2013) 6파라미터 — 이 분야의 표준이다.
    HL(비틀림) · HC1 · HC2 · LC1 · LC2 (기울임) · dc(중심간 거리).
    한 프로세스가 목록을 받아 한 줄씩 뱉는다 (기동 비용을 파일 수만큼 내지 않으려고)."""
    need = [p for p in dict.fromkeys(paths) if p not in _AB]
    if need:
        V = venv_metrics()
        env = dict(os.environ, PYTHONPATH=f"{V}/ABangle", MPLBACKEND="Agg")
        for i in range(0, len(need), chunk):
            part = need[i:i+chunk]
            lst = "/content/_ab_list.txt"; open(lst, "w").write("\n".join(part))
            pr = subprocess.run([f"{V}/bin/python", f"{V}/ab.py", "@" + lst],
                                capture_output=True, text=True, env=env)
            for l in (pr.stdout or "").split("\n"):
                if "\t" not in l: continue
                k, js = l.split("\t", 1)
                try: d = json.loads(js)
                except Exception: continue
                _AB[k] = None if "__err__" in d else d
            miss = [q for q in part if q not in _AB]
            for q in miss: _AB[q] = None
            if miss and not quiet:
                print(f"    ABangle 실패 {len(miss)}/{len(part)}:",
                      (pr.stderr or "").strip().split("\n")[-1][:120])
            if not quiet:
                print(f"    ABangle {min(i+chunk, len(need))}/{len(need)}", flush=True)
    return {p: _AB.get(p) for p in paths}

def todo(rows, done):
    """이미 결과가 있는 구성체를 뺀다. 전부 있으면 리스트가 비고, 호출부는
    **모델 로드·설치 자체를 건너뛴다.**"""
    t = [r for _, r in rows.iterrows() if not done(r)]
    print(f"  이미 있음 {len(rows)-len(t)}/{len(rows)}"
          + (f" · 남은 것 {len(t)}" if t else " — 건너뜀"))
    return t

def setup_font():
    """한글 폰트. v11 은 이 여섯 줄을 셀 세 곳에 복붙해 두었다 — 함수 하나로 합친다.

    ★ `matplotlib.use("Agg")` 를 부르지 않는다. v11 은 그림 셀마다 첫 줄에 그걸 넣어
      두었는데, Agg 는 화면 없는 백엔드라 **plt.show() 가 아무것도 안 그린다.**
      그림이 파일로는 저장되지만 노트북 안에서는 한 장도 안 보였다."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    if not any("Nanum" in f.name for f in fm.fontManager.ttflist):
        os.system("apt-get -qq install fonts-nanum > /dev/null 2>&1")
        p = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.isfile(p): fm.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
                         "figure.dpi": 120})
    return plt

print("2절 함수 준비 완료 —", len([k for k in dir() if not k.startswith("_")]), "개 이름")
''')
# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 3절 · 기준점 — ABodyBuilder2

**링커를 뺀 VL / VH 두 서열**을 준다. 링커가 없으니 "가장 이상적인 VH-VL 결합 각도"다.
모델 4개를 다 저장해서 **기준점 자체의 불확실성**을 공짜로 얻는다 —
링커 효과가 그 폭보다 작으면 읽을 것이 없다는 뜻이기 때문이다.

항체당 ~30초. v09 가 이미 만들었으면 통째로 건너뛴다.
''')

code(r'''
# ── 3절 · ABodyBuilder2 기준점 ─────────────────────────────────────────────
# 링커가 없으니 "이상적인 VH-VL 각도"다. 4모델을 다 남겨 기준점의 폭을 같이 잰다.
_f = CONS.groupby("항체").head(1)
TODO = todo(_f, lambda r: all(os.path.isfile(f"{REFD(r.항체)}/ref_m{i}.pdb")
                              for i in range(4))) if RUN_ABB2 else []
if not RUN_ABB2: print("ABB2 건너뜀 (RUN_ABB2=False)")
if TODO:
  from ImmuneBuilder import ABodyBuilder2
  ABB = ABodyBuilder2()
  for r in TODO:
    if not budget(60, "ABB2"): break
    d = REFD(r.항체); os.makedirs(d, exist_ok=True)
    try:
        H, L = (r.D1, r.D2) if r.배향 == "HL" else (r.D2, r.D1)
        pred = ABB.predict({"H": H, "L": L})
        for i in range(4): pred.save_single_unrefined(f"{d}/ref_m{i}.pdb", index=i)
        shutil.copy(f"{d}/ref_m{pred.ranking[0]}.pdb", f"{d}/ref.pdb")
        print(f"  {elapsed()} {r.항체:<22} ref.pdb + ref_m0..m3.pdb")
    except Exception as e:                      # 한 항체가 죽어도 나머지는 간다
        print(f"  ★ {r.항체} ABB2 실패: {type(e).__name__}: {e}")

_miss = [ab for ab in CONS.항체.unique() if not os.path.isfile(f"{REFD(ab)}/ref.pdb")]
print(f"기준점 {CONS.항체.nunique()-len(_miss)}/{CONS.항체.nunique()}"
      + (f"  ★ 빠짐: {_miss}" if _miss else ""))
assert not _miss or not RUN_ML, \
    "기준점이 빠진 항체가 있다 — Δ 의 영점이 없으므로 5절부터 못 간다"
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 3절-B · 계면 강도 축 — **VH-VL 이 얼마나 세게 붙는가**

### 왜 이 축이 필요한가
지금까지 우리는 **링커가 Fv 를 얼마나 여는가**만 봤다. 그런데 scFv 가 HMW 로 가는
길 가운데 하나는 **도메인 교환 이량체**다: 한 scFv 의 VH 가 *다른* scFv 의 VL 과 짝짓는다.
링커를 12잔기 아래로 줄이면 분자내 짝짓기가 기하학적으로 불가능해져 강제로 이량체가
되는 것이 diabody 이고 (Holliger et al., PNAS 1993), 그 연속선상에 있는 현상이다.

그렇다면 **열린 것만으로는 부족하다.** 열린 계면이 다른 분자와 붙을 만큼 끈끈해야 한다.

| | VH-VL 이 세게 붙는 Fv | VH-VL 이 약하게 붙는 Fv |
|---|---|---|
| 링커가 좀 풀어 놓으면 | 노출된 면이 끈끈 → 다른 scFv 와 이량체 → **HMW** | 노출돼도 잘 안 붙음 → **괜찮다** |

이것이 **같은 링커인데 블록마다 결과가 다른 것**을 설명할 수 있다. 블록마다 도메인이
다르니 계면 강도가 다르기 때문이다.

### ★ 열역학을 똑바로 세우면 — 순진한 형태는 **틀린다**
"계면이 셀수록 HMW 가 많다" 고 그냥 쓰면 안 된다. 평형을 직접 풀어 보면 이렇다.

```
M_closed ⇌ M_open            분자내 계면 하나가 풀린다     K_open  = Kd / c_eff
2 M_open ⇌ D                 분자간 계면 **둘**이 생긴다   K_assoc = c_eff2 / Kd²

[D] = K_assoc·[M_open]² = (c_eff2/Kd²)·(Kd/c_eff)²·[M_c]² = (c_eff2/c_eff²)·[M_c]²
                                                             ↑ Kd 가 **사라진다**
```

단량체는 계면이 **하나**, 이량체는 **둘**, 열리는 데 **하나**가 드니 지수가 정확히
상쇄된다. 수치로 확인했다 — Kd 를 3.5 자릿수 쓸어도 이량체 분율이 0.001988 → 0.001992
로 꿈쩍도 안 한다. 같은 조건에서 링커(c_eff)만 쓸면 0.00002 → 0.80 으로 움직인다.

**즉 평형만 놓고 보면 계면 친화도는 이량체 분율을 바꾸지 않는다.** 링커 기하가 전부다.
(실험 문헌도 이 계열이다 — 이황화 안정화 scFv, 고리화 scFv, Arndt/Plückthun 1998 은
전부 "closed 를 안정화하면 응집이 준다" 는 **주변부** 관계를 본다. 평형에서 상쇄되고도
그 방향이 남는 것은 응집이 비가역이라 동역학 영역이기 때문이다.)

### 그런데 **조건부로** 물으면 상쇄가 안 된다 — 그게 우리가 묻는 것이다
상쇄는 `[M_open]` 을 다시 `Kd` 의 함수로 **되돌려 넣을 때만** 일어난다.
우리는 열린 분율을 BioEmu 로 **측정한다.** 그 대입을 하지 않는다.

```
[D] = (c_eff2 / Kd²) · [M_open]²      ← [M_open] 을 관측량으로 고정하면
                                         Kd 가 작을수록(셀수록) 이량체가 **제곱으로** 는다
```

수치로: 열린 분율을 고정하고 Kd 를 3자릿수 쓸면 이량체가 0.2 → 200,000 으로 간다.

> **묻는 것: 같은 만큼 열렸을 때, 계면이 센 Fv 가 그 열림을 더 많은 응집으로 바꾸는가.**

```
y_ij = α_j + β·log(열림_ij) + γ·(계면강도_j × log(열림_ij)) + ε
```

★ **주효과로는 못 넣는다.** 계면강도는 항체마다 상수이고, 7절의 항체 안 중심화가
항체마다 상수인 것을 **완전히 지운다** (모의에서 잔차 폭 3×10⁻¹⁶). 상호작용만 살아남는다.
열림분율은 항체 안에서 변하기 때문이다.

### ★ 그래서 **양측** 검정이다 — 부호를 미리 못 박는다
두 방향이 다 가능하고, 어느 쪽이 나오느냐가 곧 해석이다.

| 나오는 부호 | 뜻 |
|---|---|
| **rho < 0** | 조건부 예측대로다. `[D] ∝ [M_open]²/Kd²` 의 Kd 항이 보인 것 |
| **rho > 0** | 조건화가 샜다. 열림분율이 계면 친화도를 덜 반영하면 **주변부** 관계가 새어 들어오는데 그쪽은 방향이 반대다 (Arndt/Plückthun 계열) |

★ **전제가 하나 있다.** 조건부 논증은 BioEmu 의 열린 분율이 `Kd·c_eff` 를 실제로
반영할 때만 성립한다. BioEmu 가 표준 Fv 사전분포만 보고 계면 친화도에 눈멀어 있으면
조건화가 새고 부호가 뒤집힌다. 6절-C·6절-D 의 채널 판정이 그걸 잰다.

### ★ 그리고 x 는 **로그** 열림분율이어야 한다 — 선형을 쓰면 가짜 신호가 난다
기전이 이량체면 `HMW ∝ 열림²` 이므로 `logit(HMW) ≈ 2·log(열림) + 상수` 다.
즉 **로그 축에서 기울기가 상수 2** 이고, 선형 축에서는 `d/d열림 = 2/열림` 이라
**열림분율이 작은 항체가 기계적으로 더 가파른 기울기**를 갖는다.
그런데 열림분율은 계면강도와 상관된다 (센 계면은 덜 열린다). 그러면 기울기가
계면강도를 따라가는 것이 생물이 아니라 **축의 곡률**이 된다.

상호작용을 **안 심은** 모의에서 이 인공물만으로 `Spearman = 1.00` 이 나왔다 —
항체 5개의 정확 순열에서 얻을 수 있는 최소 p 를 정확히 때리는 완벽한 순서다.
로그 축으로 바꾸면 `r = 0.017` 로 사라진다.

셀은 덤으로 **지수 n** (logit(HMW) 대 log(열림) 기울기)도 찍는다.
`n ≈ 2` 면 열린 단량체 둘이 만나는 기전과 맞고, `n ≈ 1` 이면 다른 이야기다.

### 무엇으로 계면 강도를 잴 것인가 — ipTM 은 **1순위가 아니다**
ipTM 을 쓰자는 제안은 방향이 맞지만, 그대로 쓰면 안 되는 이유가 셋 있다.

1. **ipTM 은 친화도가 아니라 자기확신이다.** "이 배치가 맞다고 AF2 가 얼마나 확신하나"
   이지 ΔG 가 아니다. BindCraft 도 ipTM 을 **설계 손실함수와 통과 필터**(보고된 컷오프
   `ipTM > 0.5`)로 쓰지, 친화도의 정량 대리로 쓰지 않는다. 그 용법이 우리 용법을
   보증하지 못한다.
2. **VH-VL 계면은 PDB 에서 가장 흔한 단백질-단백질 계면이다.** 천연 Fv 열 몇 개에
   돌리면 ipTM 이 전부 0.85~0.95 에 몰려 **변별력이 없을 가능성이 높다.**
3. **ipTM 은 사슬 길이로 정규화된다.** CDR-H3 길이가 항체마다 크게 다르므로 그 경로로
   오염될 수 있다 (같은 이유로 ipSAE 같은 대체 지표가 제안돼 있다).

**사전 등록 규칙** — 결과 보기 전에 박는다.

1. ipTM 의 변동계수(CV)가 5% 이상이면 → ipTM 을 쓴다.
2. 아니면 → **계면 매몰면적(BSA)** 으로 자동 전환. ABB2 구조에서 공짜로 나온다.
3. 나머지(계면 PAE · 접촉밀도 · 소수성 · ABB2 4모델 흔들림)는 **일치도 확인용**이다.
   서로 안 맞으면 그 사실 자체를 보고한다. 골라 쓰지 않는다.
4. 계면강도가 CDR-H3 길이 · Fv 순전하 · Fv 소수성 같은 **알려진 응집 인자**와
   |rho| > 0.8 이면, 유의해도 "계면 때문" 이라 단정하지 않는다. 셀이 그 표를 찍는다.

**더 나은 도구가 있다면**: 접촉 기반 ΔG 예측기(PRODIGY 계열)는 실험 Kd 에 대해
검증된 상관이 있다. 여기 안 넣은 이유는 의존성이 하나 더 늘고 ABB2 구조로도
BSA·접촉밀도가 같은 순위를 대개 주기 때문이다. ipTM 이 변별력이 없고 BSA 도
경쟁 인자와 얽히면, 그때 붙일 다음 도구가 그것이다.

ABB2 기반 지표 넷은 **이미 Drive 에 있는 구조로 공짜로** 나온다. ipTM 만 GPU 를 쓴다.
''')

code(r'''
# ── 3절-B · 계면 강도 (ABodyBuilder2 기준 구조에서 공짜로) ─────────────────
from Bio.PDB.SASA import ShrakeRupley
_SR = ShrakeRupley()
# Kyte-Doolittle 로 소수성 잔기 판정 (계면의 끈끈함을 실제로 만드는 것)
_HYDROPHOBIC = set("AVLIMFWYC")
IFACE_CUT = 5.0        # 계면 접촉 판정 (무거운 원자 간 Å)

def _chains_HL(path):
    """ABB2 기준 구조 → (H 잔기 목록, L 잔기 목록). 사슬 이름이 H/L 이다."""
    st = first_model(path)
    if st is None:
        return None, None
    ch = {c.id: [r for r in c if "CA" in r] for c in st}
    if "H" in ch and "L" in ch and ch["H"] and ch["L"]:
        return ch["H"], ch["L"]
    big = sorted(ch.values(), key=len, reverse=True)[:2]
    return (big[0], big[1]) if len(big) == 2 else (None, None)

def _sasa_of(groups):
    """잔기 묶음 목록을 한 구조로 모아 SASA 합을 낸다."""
    from Bio.PDB import Structure, Model, Chain
    sub = Structure.Structure("s"); mdl = Model.Model(0); sub.add(mdl)
    for i, g in enumerate(groups):
        c = Chain.Chain("AB"[i]); mdl.add(c)
        for r in g: c.add(r.copy())
    _SR.compute(sub, level="R")
    return sum(r.sasa for c in sub[0] for r in c)

def interface_metrics(path):
    """한 구조의 VH-VL 계면 지표.

    BSA = SASA(H) + SASA(L) − SASA(H+L).  응집 가설이 직접 겨누는 양이다 —
    계면이 넓고 소수성이 높을수록, 열렸을 때 노출되는 면이 끈끈하다.
    """
    H, L = _chains_HL(path)
    if not H or not L or len(H) < 50 or len(L) < 50: return {}
    try:
        bsa = _sasa_of([H]) + _sasa_of([L]) - _sasa_of([H, L])
    except Exception:
        return {}
    # 계면 잔기 = 상대 사슬의 무거운 원자와 IFACE_CUT 안인 잔기
    hx = [np.array([a.coord for a in r if a.element != "H"]) for r in H]
    lx = [np.array([a.coord for a in r if a.element != "H"]) for r in L]
    nct, hres = 0, []
    for i, a in enumerate(hx):
        near = False
        for j, b in enumerate(lx):
            if len(a) and len(b) and np.min(np.linalg.norm(
                    a[:, None, :] - b[None, :, :], axis=-1)) < IFACE_CUT:
                nct += 1; near = True
        if near: hres.append(H[i].get_resname())
    for j, b in enumerate(lx):
        if any(len(a) and len(b) and np.min(np.linalg.norm(
                a[:, None, :] - b[None, :, :], axis=-1)) < IFACE_CUT for a in hx):
            hres.append(L[j].get_resname())
    _3to1 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
             "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
             "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"}
    aa = [_3to1.get(x, "X") for x in hres]
    return dict(계면_BSA=round(float(bsa), 1),
                계면_접촉수=nct,
                계면_잔기수=len(hres),
                계면_접촉밀도=round(nct/max(len(hres), 1), 2),
                계면_소수성=round(float(np.mean([a in _HYDROPHOBIC for a in aa])), 3)
                            if aa else np.nan)

CSV_IFACE = f"{OUT}/interface.csv"
IF = pd.read_csv(CSV_IFACE) if (REUSE_CSV and os.path.isfile(CSV_IFACE)) else pd.DataFrame()
_have_if = set(IF.항체) if len(IF) else set()
rows = []
for ab in CONS.항체.unique():
    if ab in _have_if: continue
    mods = sorted(glob.glob(f"{REFD(ab)}/ref_m[0-9].pdb")) or [f"{REFD(ab)}/ref.pdb"]
    mods = [m for m in mods if os.path.isfile(m)]
    if not mods:
        print(f"  {ab:<22} 기준 구조 없음 — 건너뜀"); continue
    ms = [interface_metrics(m) for m in mods]
    ms = [m for m in ms if m]
    if not ms:
        print(f"  {ab:<22} 계면 계산 실패"); continue
    d = dict(항체=ab, n모델=len(ms))
    for k in ms[0]:
        v = [m[k] for m in ms if k in m and m[k] == m[k]]
        d[k] = round(float(np.mean(v)), 3) if v else np.nan
        d[f"{k}_SD"] = round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0
    # ★ ABB2 4모델의 ABangle 흩어짐 — 공짜로 얻는 '기준점 자체의 불확실성'.
    #   4개 모델이 각도에 대해 서로 다르게 말하면 그 Fv 의 배향이 덜 결정적이라는 뜻이고,
    #   그것 자체가 계면이 헐겁다는 신호다.
    try:
        w = f"/content/_ifp/{safe(ab)}"; os.makedirs(w, exist_ok=True)
        r0 = CONS[CONS.항체 == ab].iloc[0]
        pp = [write_pair(m, r0.b1, r0.b2, f"{w}/{os.path.basename(m)}",
                         r0.배향[0], r0.배향) for m in mods]
        aa_ = [x for x in abangle6_many(pp, quiet=True).values() if x]
        if len(aa_) > 1:
            d["기준점_흔들림"] = round(float(np.mean(
                [np.std([x[k] for x in aa_], ddof=1)/AB_SD[k] for k in AB6])), 3)
    except Exception as e:
        print(f"    {ab} 흔들림 계산 실패: {type(e).__name__}")
    rows.append(d)
    print(f"  {ab:<22} BSA {d.get('계면_BSA', float('nan')):>7.1f} Å² · "
          f"접촉 {d.get('계면_접촉수', 0):>4} · 소수성 {d.get('계면_소수성', float('nan')):.2f}"
          + (f" · 흔들림 {d['기준점_흔들림']:.2f}σ" if "기준점_흔들림" in d else ""))
if rows:
    IF = pd.concat([IF, pd.DataFrame(rows)], ignore_index=True) if len(IF) else pd.DataFrame(rows)
    IF = IF.drop_duplicates("항체", keep="last")
    IF.to_csv(CSV_IFACE, index=False, encoding="utf-8-sig")
shutil.rmtree("/content/_ifp", ignore_errors=True)
print(f"\n계면 지표 {len(IF)} 항체 → {CSV_IFACE}")
if len(IF): display(IF)
''')

code(r'''
# ── 3절-B2 · ★ 앵커 스팬 r_ab 와 고정스팬 c_eff — **라벨이 필요 없다** ─────
# 흔히 쓰는 c_eff ∝ N^-1.5 는 사슬 **양 끝이 자유로울 때**의 극한이다.
# scFv 는 그게 아니다 — 닫힌 Fv 에서 링커가 이어야 할 두 부착점 거리 r_ab 가
# 고정돼 있다. 그러면 c_eff 는 그 거리에서의 확률밀도다:
#
#     ⟨r²⟩ = n·l·b        c_eff ∝ P(r_ab) = (3/2π⟨r²⟩)^{3/2} · exp(−3r_ab²/2⟨r²⟩)
#     최적 길이  n* = r_ab² / (l·b)
#
# l=3.5 Å(잔기당) · b=10 Å(Kuhn) 이면 r_ab=35 Å 에서 n* = 35 aa 다.
# → **실측 링커 15~25 aa 는 전부 상승 가지에 있다. 길수록 c_eff 가 커진다.**
#   15→25 배수: r=30 Å 에서 ×1.30 · 35 Å 에서 ×1.88 · 40 Å 에서 ×2.89.
#   그런데 r_ab ≲ 26 Å 이면 꼭짓점이 창 안으로 들어와 **부호가 뒤집힌다**.
#
# 왜 이게 중요한가: 실측에서 LH 4항체가 전부 '그 블록의 **최단** 링커가 최악' 이었고
# HL 1항체만 반대였다. 이 식은 그 반전이 일어날 조건을 **정량으로** 준다 —
# 그 항체의 r_ab 가 26 Å 아래인가. CA 두 개만 재면 답이 나온다.
#
# ★ 라벨이 필요 없다 → 합성 패널에도 그대로 돌아가고 표본 제약을 안 받는다.
KUHN_B, RES_L = 10.0, 3.5      # Å. 바꾸려면 여기서 바꾸고 아래 판정을 다시 읽어라
A3_TO_M = 1661.0               # 1 Å⁻³ = 1661 M

def anchor_span(ref_pdb, od="HL"):
    """닫힌 Fv 에서 링커가 이어야 하는 두 CA 사이 거리 (Å).

    배향 od 의 **첫 도메인 C말단 CA** → **둘째 도메인 N말단 CA**.
    이 거리가 배향마다 다르다는 것이 'VH-링커-VL 과 VL-링커-VH 가 다르다' 의
    구조적 정체다 — 같은 길이의 링커가 한쪽에서는 남고 한쪽에서는 모자란다.
    """
    st = first_model(ref_pdb)
    if st is None:
        return np.nan
    ch = {c.id: [r for r in c if "CA" in r] for c in st}
    d1, d2 = od[0], od[1]
    if d1 not in ch or d2 not in ch or not ch[d1] or not ch[d2]:
        return np.nan
    return float(np.linalg.norm(ch[d1][-1]["CA"].coord - ch[d2][0]["CA"].coord))

def c_eff_fixed(n_res, r_ab, b=None, l=None):
    """고정 스팬에서의 실효농도 (M). n_res 는 링커 잔기 수."""
    b = KUHN_B if b is None else b
    l = RES_L if l is None else l
    n_res = float(n_res); r_ab = float(r_ab)
    if not (n_res > 0 and np.isfinite(r_ab) and r_ab > 0):
        return np.nan
    r2 = n_res * l * b
    return float((3.0/(2*np.pi*r2))**1.5 * np.exp(-3.0*r_ab*r_ab/(2*r2)) * A3_TO_M)

def n_star(r_ab, b=None, l=None):
    """c_eff 를 최대로 만드는 링커 길이 (잔기).  n* = r_ab²/(l·b)"""
    b = KUHN_B if b is None else b
    l = RES_L if l is None else l
    return float(r_ab*r_ab/(l*b)) if np.isfinite(r_ab) else np.nan

SPAN_CSV = f"{OUT}/anchor_span.csv"
SPAN = pd.read_csv(SPAN_CSV) if (REUSE_CSV and os.path.isfile(SPAN_CSV)) else pd.DataFrame()
_have_sp = set(SPAN.항체) if len(SPAN) else set()
_rows = []
for ab in CONS.항체.unique():
    if ab in _have_sp: continue
    r0 = CONS[CONS.항체 == ab].iloc[0]
    mods = [m for m in (sorted(glob.glob(f"{REFD(ab)}/ref_m[0-9].pdb")) or
                        [f"{REFD(ab)}/ref.pdb"]) if os.path.isfile(m)]
    if not mods:
        print(f"  {ab:<22} 기준 구조 없음 — 건너뜀"); continue
    v = [anchor_span(m, r0.배향) for m in mods]
    v = [x for x in v if np.isfinite(x)]
    if not v:
        print(f"  {ab:<22} 앵커 CA 를 못 찾았다"); continue
    _rows.append(dict(항체=ab, 배향=r0.배향, r_ab=round(float(np.mean(v)), 2),
                      r_ab_SD=round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0,
                      n모델=len(v), n_star=round(n_star(float(np.mean(v))), 1)))
if _rows:
    SPAN = pd.concat([SPAN, pd.DataFrame(_rows)], ignore_index=True) if len(SPAN) \
        else pd.DataFrame(_rows)
    SPAN = SPAN.drop_duplicates("항체", keep="last")
    SPAN.to_csv(SPAN_CSV, index=False, encoding="utf-8-sig")
if len(SPAN):
    print("="*76); print("★ 앵커 스팬 r_ab — 링커가 반드시 이어야 하는 거리"); print("="*76)
    display(SPAN)
    print(f"  ※ ABB2 4모델 평균. r_ab_SD 가 크면(≳2 Å) 기준 구조가 이 값을 잘 못 정한다.")
    _rev = SPAN[SPAN.n_star < 25.0]
    print(f"\n  n* = r_ab²/({RES_L}·{KUHN_B}).  n* 가 실측 창(15~25 aa)보다 크면")
    print("  **길수록 더 닫힌다**(c_eff 상승). 창 안으로 들어오면 부호가 뒤집힌다.")
    if len(_rev):
        print(f"  ★★ n* < 25 인 항체: {list(_rev.항체)} — 이 항체들에서 링커 길이 효과의")
        print("     **부호가 다를 것**으로 예측된다. 라벨을 안 보고 낸 예측이다.")
    else:
        print("  모든 항체가 n* > 25 — 전부 같은 부호를 예측한다. 실측에서 부호가")
        print("  뒤집힌 항체가 있다면 이 기전으로는 설명이 안 된다는 뜻이다.")
    # 구성체별 c_eff — 항체 안에서 링커마다 다르다 (= 항체내 변동이 있다)
    _sp = SPAN.set_index("항체").r_ab
    CEFF = CONS[["항체", "링커", "길이"]].drop_duplicates().copy()
    CEFF["r_ab"] = CEFF.항체.map(_sp)
    CEFF["c_eff_mM"] = [round(c_eff_fixed(n, r)*1e3, 3)
                        for n, r in zip(CEFF.길이, CEFF.r_ab)]
    CEFF = CEFF.dropna(subset=["c_eff_mM"])
    if len(CEFF):
        CEFF.to_csv(f"{OUT}/c_eff.csv", index=False, encoding="utf-8-sig")
        display(CEFF.head(12))
        _w = CEFF.groupby("항체").c_eff_mM.agg(["min", "max"])
        print(f"  항체 안 c_eff 배수 중앙 "
              f"{float((_w['max']/_w['min']).median()):.2f}배 → 항체내 변동이 있다.")
        print("  ★ 예측: 항체별 |길이 기울기| 순위가 이 배수의 순위를 따라야 한다.")
        print("    라벨을 안 쓰고 낸 순위 예측이므로 사전 등록할 수 있다.")
''')

code(r'''
# ── 3절-C · AF2-Multimer ipTM — VH + VL 을 **두 사슬로** 준다 ───────────────
# ★ 왜 ABodyBuilder2 로는 안 되는가. ABB2 는 항체 전용 모델이라 **짝지어진 Fv 를
#   전제로 짓는다.** "이 둘이 서로 찾아가 붙을 것인가" 라는 질문 자체를 못 던진다.
#   AF2-Multimer 에 링커 없이 두 사슬로 주면 바로 그 질문이 된다 — diabody 가 묻는
#   분자간 짝짓기와 같은 질문이다. 이게 사용자 가설의 핵심이고 ipTM 을 쓰는 이유다.
#
# ★ ipTM 이 v11 에서 금지됐던 것과 여기 쓰는 것은 다르다. v11 이 경계한 것은
#   **링커 포함 단일 사슬**의 ipTM 이었다 — 토큰 수로 정규화되어 링커 길이에 직접
#   오염된다. 여기서는 링커가 없고 항체당 한 번만 재므로 그 오염 경로가 없다.
#   대신 VH/VL 길이(특히 CDR-H3)가 항체마다 다른 것은 남으므로 아래에서 같이 찍는다.
RUN_IPTM   = True
IPTM_MODELS = 3          # 모델 몇 개 평균. 5 면 더 안정, 시간은 비례
CSV_IPTM   = f"{OUT}/iptm.csv"

IPTM = pd.read_csv(CSV_IPTM) if os.path.isfile(CSV_IPTM) else pd.DataFrame()
_have = set(IPTM.항체) if len(IPTM) else set()
_need = [ab for ab in CONS.항체.unique() if ab not in _have]
print(f"  이미 있음 {len(CONS.항체.unique())-len(_need)}/{CONS.항체.nunique()}"
      + (f" · 남은 것 {len(_need)} → 약 {len(_need)*4/60:.1f} h" if _need else " — 건너뜀"))

if RUN_IPTM and _need:
    V = venv311(); CF = f"{V}/bin/colabfold_batch"
    if not os.path.isfile(CF):
        print("colabfold 설치 중 (몇 분)…")
        sh(["uv", "pip", "install", "--python", f"{V}/bin/python", "-q",
            "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"])
        sh(["uv", "pip", "install", "--python", f"{V}/bin/python", "-q", "jax[cuda12]"])
    if not os.path.isfile(CF):
        print(f"★ colabfold_batch 가 없다 — ipTM 을 건너뛴다. 3절-B 의 BSA 로 간다.")
    else:
      rows = []
      for ab in _need:
        if not budget(600, "AF2-Multimer"): break
        r = CONS[CONS.항체 == ab].iloc[0]
        H, L = (r.D1, r.D2) if r.배향 == "HL" else (r.D2, r.D1)
        tag = safe(ab); w = f"/content/mm/{tag}"; os.makedirs(w, exist_ok=True)
        # ★ 콜론이 사슬 구분자다. 링커를 넣지 않는다 — 분자간 짝짓기를 묻는 것이므로.
        open(f"{w}/in.fasta", "w").write(f">{tag}\n{H}:{L}\n")
        if not glob.glob(f"{w}/*_scores_*.json"):
            sh([CF, f"{w}/in.fasta", w, "--num-models", str(IPTM_MODELS),
                "--num-recycle", "3", "--model-type", "alphafold2_multimer_v3",
                "--msa-mode", "mmseqs2_uniref_env"], tail=12)
        js = sorted(glob.glob(f"{w}/*_scores_*.json"))
        if not js:
            print(f"  ★ {ab} AF2-Multimer 실패"); continue
        it, pt, pae_x = [], [], []
        for j in js:
            try: z = json.load(open(j))
            except Exception: continue
            if "iptm" in z: it.append(float(z["iptm"]))
            if "ptm"  in z: pt.append(float(z["ptm"]))
            # 사슬 간 PAE — EBI 정의대로 '두 도메인의 **상대 배치**에 대한 확신'
            if "pae" in z:
                M = np.asarray(z["pae"], float)
                nH = len(H)
                if M.ndim == 2 and M.shape[0] >= nH + 30:
                    pae_x.append(float(np.mean(np.concatenate(
                        [M[:nH, nH:].ravel(), M[nH:, :nH].ravel()]))))
        if not it:
            print(f"  ★ {ab} 점수 파일에 iptm 이 없다"); continue
        rows.append(dict(항체=ab, ipTM=round(float(np.mean(it)), 4),
                         ipTM_SD=round(float(np.std(it, ddof=1)), 4) if len(it) > 1 else 0.0,
                         pTM=round(float(np.mean(pt)), 4) if pt else np.nan,
                         계면PAE=round(float(np.mean(pae_x)), 2) if pae_x else np.nan,
                         nH=len(H), nL=len(L), n모델=len(it)))
        print(f"  {elapsed()} {ab:<22} ipTM {rows[-1]['ipTM']:.3f} "
              f"± {rows[-1]['ipTM_SD']:.3f} · 계면PAE {rows[-1]['계면PAE']}")
      if rows:
        IPTM = pd.concat([IPTM, pd.DataFrame(rows)], ignore_index=True) if len(IPTM) else pd.DataFrame(rows)
        IPTM = IPTM.drop_duplicates("항체", keep="last")
        IPTM.to_csv(CSV_IPTM, index=False, encoding="utf-8-sig")
elif not RUN_IPTM:
    print("ipTM 건너뜀 (RUN_IPTM=False) — 3절-B 의 BSA 로 간다")

# ── 계면 강도 지수를 **하나로** 정한다. 사전 등록 규칙대로. ────────────────
IFACE = IF.copy() if len(IF) else pd.DataFrame(dict(항체=CONS.항체.unique()))
if len(IPTM): IFACE = IFACE.merge(IPTM, on="항체", how="outer")
IFACE_SRC, IFACE_WHY = None, ""
if len(IFACE):
    _cv = lambda c: (float(IFACE[c].std(ddof=1)/abs(IFACE[c].mean()))
                     if c in IFACE and IFACE[c].notna().sum() >= 3
                     and abs(IFACE[c].mean()) > 1e-9 else np.nan)
    cv_iptm, cv_bsa = _cv("ipTM"), _cv("계면_BSA")
    print("")
    print("="*76); print("★ 계면 강도 지수 선택 — 사전 등록 규칙"); print("="*76)
    print(f"  ipTM      변동계수 {cv_iptm if cv_iptm==cv_iptm else float('nan'):.4f}  "
          f"(≥ 0.05 이면 1순위)")
    print(f"  계면_BSA  변동계수 {cv_bsa if cv_bsa==cv_bsa else float('nan'):.4f}  (대체)")
    if np.isfinite(cv_iptm) and cv_iptm >= 0.05:
        IFACE_SRC, IFACE_WHY = "ipTM", f"ipTM 의 변동계수가 {cv_iptm:.3f} 로 충분하다"
    elif np.isfinite(cv_bsa):
        IFACE_SRC = "계면_BSA"
        IFACE_WHY = (f"ipTM 이 {'없거나' if not np.isfinite(cv_iptm) else f'변동계수 {cv_iptm:.3f} 로'} "
                     f"변별력이 모자라 BSA 로 전환했다 (사전 등록된 대체)")
    if IFACE_SRC:
        IFACE["계면강도"] = IFACE[IFACE_SRC]
        print(f"  → **{IFACE_SRC}** 를 쓴다. {IFACE_WHY}")
        # 지표들이 서로 맞는가 — 골라 쓰지 않고 일치도만 본다
        cand = [c for c in ("ipTM", "계면PAE", "계면_BSA", "계면_접촉밀도",
                            "계면_소수성", "기준점_흔들림")
                if c in IFACE and IFACE[c].notna().sum() >= 3]
        if len(cand) >= 2:
            print("\n  지표 간 일치도 (Spearman) — 안 맞으면 그 사실 자체를 보고한다")
            display(IFACE[cand].corr(method="spearman").round(2))
        # ── 경쟁 설명 — 계면 강도가 사실은 다른 것의 변장인가 ─────────────
        # 항체 수준에서 응집을 좌우한다고 알려진 것들이 여럿 있다. 계면강도가
        # 그중 하나와 거의 같다면, 7절-C 가 잡는 것은 계면이 아니라 그것이다.
        # 전부 **서열만으로** 공짜로 나오므로 안 볼 이유가 없다.
        _KD2 = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
                    H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
                    T=-0.7, W=-0.9, Y=-1.3, V=4.2)
        def _cdrh3_len(seq):
            """CDR-H3 길이. ANARCI 로 잡고, 실패하면 FR4 모티프 앞 W 에서 센다.
            응집과의 관계가 가장 널리 보고된 서열 특징이다."""
            try:
                from anarci import run_anarci
                _, num, det, _ = run_anarci([("x", seq)], scheme="chothia")
                if num and num[0] and det[0][0]["chain_type"] == "H":
                    return sum(1 for (pos, _i), aa in num[0][0][0]
                               if aa != "-" and 95 <= pos <= 102)
            except Exception: pass
            # ANARCI 가 실패하면 FR4 모티프(W-G-x-G)로 되짚는다.
            # ★ 1절의 FR4_H 를 쓰지 않고 여기서 다시 만든다 — 이 셀만 따로 돌려도
            #   죽지 않아야 하기 때문이다 (셀 간 숨은 의존은 나중에 반드시 문다).
            m = re.search(r"W[GAS][QKRAHPSGE]G[TQAS]", seq[-40:])
            return float(40 - m.start()) if m else np.nan
        _cov = []
        for _, rr in CONS.drop_duplicates("항체").iterrows():
            H_, L_ = (rr.D1, rr.D2) if rr.배향 == "HL" else (rr.D2, rr.D1)
            fv = H_ + L_
            _cov.append(dict(항체=rr.항체,
                             CDRH3길이=_cdrh3_len(H_),
                             Fv순전하=sum(fv.count(a) for a in "KR")
                                      - sum(fv.count(a) for a in "DE"),
                             Fv소수성=round(float(np.mean(
                                 [_KD2.get(a, 0.0) for a in fv])), 3),
                             Fv길이=len(fv),
                             홀수Cys=int(fv.count("C") % 2)))   # 홀수면 유리 티올이 있다
        COV = pd.DataFrame(_cov)
        IFACE = IFACE.merge(COV, on="항체", how="left")
        _cc = [c for c in ("CDRH3길이", "Fv순전하", "Fv소수성", "Fv길이")
               if c in IFACE and IFACE[c].notna().sum() >= 3
               and IFACE[c].std(ddof=1) > 0]
        if _cc:
            print("\n  경쟁 설명 — 계면강도가 사실 이것들의 변장인가 (Spearman)")
            _t = pd.DataFrame([dict(경쟁변수=c,
                    rho=round(float(st_.spearmanr(IFACE.계면강도, IFACE[c],
                                                  nan_policy="omit")[0]), 2))
                               for c in _cc])
            _t["판정"] = np.where(_t.rho.abs() > 0.8,
                                  "★ 거의 같다 — 7절-C 의 해석을 이것과 나눌 수 없다",
                                  "구분된다")
            display(_t)
            if (_t.rho.abs() > 0.8).any():
                print("     ★ 겹치는 변수가 있다. 7절-C 가 유의해도 '계면 때문' 이라고")
                print("       단정할 수 없다 — 항체 5개로는 둘을 못 가른다. 그렇게 보고하라.")
    else:
        print("  → ★ 쓸 수 있는 계면 지표가 없다. 7절-C 를 건너뛴다.")

    # ── 신뢰도 사전점검 — **라벨을 만지기 전에** 한다 ──────────────────────
    # 지표 자체가 시끄러우면 상관이 √신뢰도 배로 눌린다. 항체 5개에서는 그것만으로
    # 검정이 죽는다. ABB2 4모델의 흩어짐이 곧 '같은 항체를 다시 재면 얼마나 다른가' 다.
    # ★ 사전 등록: 신뢰도 < 0.7 인 지표는 7절-C 에 못 들어간다. 결과와 무관한 규칙이다.
    print("\n" + "="*76)
    print("★ 신뢰도 사전점검 — 지표가 항체를 가를 만큼 안정한가 (라벨 보기 전)")
    print("="*76)
    rel = []
    for c in ("ipTM", "계면_BSA", "계면_접촉밀도", "계면_소수성", "기준점_흔들림"):
        if c not in IFACE or IFACE[c].notna().sum() < 3: continue
        sdb = float(IFACE[c].std(ddof=1))
        sdw = (float(IFACE.get(f"{c}_SD", pd.Series(dtype=float)).median())
               if f"{c}_SD" in IFACE else np.nan)
        r_ = max(1 - (sdw**2)/(sdb**2), 0.0) if (np.isfinite(sdw) and sdb > 1e-12) else np.nan
        rel.append(dict(지표=c, 항체간SD=round(sdb, 4),
                        항체내SD=round(sdw, 4) if np.isfinite(sdw) else np.nan,
                        신뢰도=round(r_, 3) if np.isfinite(r_) else np.nan,
                        감쇠배율=round(float(np.sqrt(r_)), 2) if np.isfinite(r_) else np.nan,
                        판정=("★ 7절-C 부적격 (신뢰도 < 0.7)" if np.isfinite(r_) and r_ < 0.7
                              else "적격" if np.isfinite(r_) else "항체내 SD 미측정")))
    if rel:
        REL = pd.DataFrame(rel); display(REL)
        print("  관측 상관 ≈ 참 상관 × 감쇠배율. 시끄러운 지표는 있어도 못 쓴다.")
        _bad = REL[(REL.신뢰도.notna()) & (REL.신뢰도 < 0.7)]
        if IFACE_SRC and IFACE_SRC in set(_bad.지표):
            print(f"  ★★ 고른 지표 '{IFACE_SRC}' 가 부적격이다. 7절-C 를 읽지 마라.")
            IFACE_SRC = None
    display(IFACE)
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 4절 · BioEmu — 링커 서열이 생성기에 **직접** 닿는 유일한 경로

```
서열 → MMseqs2 MSA → AF2 evoformer (single/pair repr) → 확산 → 백본 앙상블
```

BBFlow 에서는 링커가 **씨앗 백본을 통해서만** 들어올 수 있었는데, 그 백본에서
서열을 안 읽는 것이 측정됐다 (`p = 0.9477`, 효과크기 0.08). 즉 링커 화학은
샘플러에 닿기 **전에** 걸러졌다. BioEmu 는 그 병목이 없다.

### 그래도 짚고 갈 것 — 논문이 검증한 범위 밖이다
Lewis et al., *Science* **389**, eadv9817 (2025) 에서 확인되는 것과 안 되는 것:

| 논문이 말한 것 | 우리 실험에 대한 함의 |
|---|---|
| 도메인 운동 벤치마크 83% 가 참조구조 3 Å 이내 | VH-VL 같은 2도메인 문제 자체는 사정권이다 |
| 자유에너지 MAE 0.74~0.9 kcal/mol, 안정성 Spearman ≈ 0.6 | **정량용이 아니다.** 순위 정도만 |
| 본문 결과는 전부 **10,000 샘플** | 우리는 150이다. 66분의 1 — 분포 꼬리는 못 믿는다 |
| 성능이 서열유사도 30% 에서 평탄해진다 | 천연 단백질 기준이다. **인공 링커는 MSA 가 비어 있다** |
| "emulates **single protein chains**", 올리고머 상태는 **암묵적** | 응집(=이량체화)은 **모델 밖**이다 |
| 에너지 함수가 없어 재가중이 불가능 | `−RT ln P` 지형은 **모양만** 읽는다 |

★ 마지막 두 줄이 이 연구의 한계다. 우리가 재는 것은 **단량체 앙상블의 열린 정도**이고,
HMW 는 **이량체 형성의 결과**다. 그 사이의 연결(열린 단량체 → 사슬간 VH-VL 결합)은
모델이 아니라 **가설**이다. 상관이 나와도 그 가설을 증명하지는 않는다.

### 잔존율 게이트 — 이게 먼저다
`filter_samples=False` 로 원본을 받아 잔존율을 직접 센다. 잔존율이 링커마다 다르면
뒤의 모든 검정이 포즈가 아니라 **생존율 차이**를 재게 된다.
`N_PROBE` 로 먼저 찔러 **50% 미만이거나 링커 간 20 %p 넘게 벌어지면 멈춘다.**

### v05 · v06 이 죽은 자리 — 고쳐져 있다
- `AssertionError: n_samples (6) is not multiple of n_particles (5)` —
  `batch_size = max(1, int(batch_size_100·(100/L)²))` 이고 SMC 가 배수를 요구한다.
  드라이버가 서열마다 `batch_size_100` 을 **역으로 푼다.**
- `UnicodeEncodeError` (mdtraj XTC 작성자가 파일명을 ascii 로 인코딩) —
  항체 이름이 한글이었다. 작업 경로만 **ascii 슬러그 + md5 6자리**로 만든다.
''')

code(r'''
# ── 4절 · BioEmu 앙상블 ────────────────────────────────────────────────────
N_BE         = 150          # 구성체당 샘플 수
N_PROBE      = 20           # 먼저 이만큼 뽑아 **잔존율과 시간**을 재고 계속할지 정한다
BE_STEER     = True         # physical_steering (SMC 입자 5개 → 약 5배). 생존편향 원천 제거
BE_BATCH     = 5            # 목표 배치. **num_particles(5) 의 정확한 배수**여야 한다
BE_NP        = 5            # physical_steering.yaml 의 num_particles
BE_HOURS_MAX = 6.0          # 탐침 외삽이 이보다 길면 중단
RET_MIN      = 0.50         # 잔존율 하한
RET_SPREAD   = 0.20         # 링커 간 잔존율 최대 격차
# ★ 씨앗 반복 — **잡음 바닥을 모르면 어떤 링커 간 차이도 해석할 수 없다.**
#   같은 서열을 두 번 돌려서 나오는 차이가 BioEmu 의 실행 간 잡음이다.
#   링커 A 와 B 의 차이가 그 잡음보다 작으면, 그건 링커 효과가 아니다.
#   v11 에는 이 대조가 아예 없었다 — 그래서 4-C절의 2×2 도 해석 불가였다.
N_SEED_REP   = 2            # 대조 구성체 몇 개를 두 번 돌릴까 (0 이면 끔)

BE_DRIVER = r"""
import sys, os, json, math, numpy as np
seq, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
steer, tgt, npart = sys.argv[4] == "1", int(sys.argv[5]), int(sys.argv[6])
L = len(seq)
_bs = lambda c: max(1, int(c * (100.0/L)**2))       # bioemu/sample.py 와 **같은 식**

def bs100_for(target):
    f = (100.0/L)**2
    c = max(1, math.ceil(target/f))
    while _bs(c) < target and c < 100000: c += 1
    return c if _bs(c) == target else None

c100 = 10 if tgt <= 0 else (bs100_for(tgt) or 10)   # 10 → batch 1 (가장 안전)
print(f"  서열 {L}aa · batch_size_100={c100} -> batch_size={_bs(c100)}"
      + (f" (입자 {npart} 의 배수)" if steer else ""))
if steer and _bs(c100) >= npart and _bs(c100) % npart:
    print(f"  * batch {_bs(c100)} 이 입자 {npart} 의 배수가 아니다 - batch 1 로 내린다")
    c100 = 10

kw = dict(sequence=seq, num_samples=n, output_dir=out,
          filter_samples=False, batch_size_100=c100)  # 원본을 받고 잔존율은 우리가 센다
if steer:
    import bioemu, glob as _g, os.path as _p, re as _re
    c = [x for x in _g.glob(_p.join(_p.dirname(bioemu.__file__), "**", "*steering*.y*ml"),
                            recursive=True) if "physical" in _p.basename(x)]
    if c:
        y = _re.sub(r"num_particles:\s*\d+", f"num_particles: {npart}", open(c[0]).read())
        open(f"{out}/steer.yaml", "w").write(y)
        kw["denoiser_config"] = f"{out}/steer.yaml"
        print(f"  steering: {_p.basename(c[0])} num_particles={npart}")
    else: print("  steering yaml 을 못 찾았다 - 끄고 간다")
from bioemu.sample import main as _s
_s(**kw)

import mdtraj as md
t  = md.load(os.path.join(out, "samples.xtc"), top=os.path.join(out, "topology.pdb"))
ca = t.atom_slice(t.topology.select("name CA")).xyz * 10.0          # Angstrom
keep = np.ones(len(ca), bool)
iu = np.triu_indices(ca.shape[1], k=3)
for i in range(len(ca)):                                           # 프레임마다 - 메모리 안전
    c_ = ca[i]
    if np.linalg.norm(np.diff(c_, axis=0), axis=-1).max() >= 4.5: keep[i] = False; continue
    D = np.linalg.norm(c_[:, None, :] - c_[None, :, :], axis=-1)
    if D[iu].min() <= 1.0: keep[i] = False
json.dump(dict(n=int(len(keep)), retention=float(keep.mean())),
          open(os.path.join(out, "keep.json"), "w"))
for i in np.where(keep)[0]: t[int(i)].save_pdb(os.path.join(out, "f%04d.pdb" % int(i)))
print("RETENTION %.3f %d" % (keep.mean(), len(keep)))
"""

def venv_be(V="/content/venv_be"):
    """빈 venv 에서 시작하므로 Colab 의 tensorflow 와 bioemu 의 tensorflow-cpu 가
    같은 디렉터리를 공유해 .so 가 엇갈리던 문제가 원천적으로 없다."""
    P = f"{V}/bin/python"
    if os.path.isfile(P) and not subprocess.run(
            [P, "-c", "import bioemu, mdtraj"], capture_output=True).returncode: return V
    venv311(V)
    sh(["uv", "pip", "install", "--python", P, "-q", "bioemu[cuda]", "mdtraj", "fire"], tail=8)
    for m in ("bioemu", "mdtraj"):
        rc = subprocess.run([P, "-c", f"import {m}"], capture_output=True)
        print(f"  import {m:<8} {'OK' if rc.returncode==0 else '실패: '+rc.stderr.decode()[-200:]}")
    return V

# ★ mdtraj 의 XTC 작성자는 Cython 이고 파일명을 **ascii 로 인코딩**한다.
#   우리 항체 이름이 한글이라 UnicodeEncodeError 가 났다 ('실'.isalnum() 은 True 다).
#   → BioEmu 작업 경로만 ascii 전용 슬러그로. Drive 출력 경로는 그대로 둔다.
import hashlib as _hl
def be_slug(x):
    a = re.sub(r"[^A-Za-z0-9]+", "_", str(x).encode("ascii", "ignore").decode()).strip("_")
    h = _hl.md5(str(x).encode("utf-8")).hexdigest()[:6]   # ascii 만 남기면 충돌할 수 있다
    return f"{a}_{h}" if a else h
BEW    = lambda ab, lk, n: f"/content/be/{be_slug(ab)}__{be_slug(lk)}_{n}"
# ★ 완료 판정은 **표식 파일**로 한다. 파일 수로 세면 잔존율 92% 일 때 150 요청에
#   138개만 남아 영원히 '미완료' 가 되고 매번 통째로 재실행된다.
BEDONE = lambda r: f"{FL(r)}/{r.링커}__BioEmu.done"

def run_bioemu(rows, label="본실행", n=N_BE, keep_out=True, per_s=900):
    """rows(구성체 리스트)에 BioEmu 를 돌린다. 잔존율 dict 와 소요시간 dict 를 준다."""
    V = venv_be(); P = f"{V}/bin/python"
    open("/content/be_driver.py", "w").write(BE_DRIVER)
    RET, sec = {}, {}
    for r in rows:
        if not budget(per_s, f"BioEmu {label}"): break
        w = BEW(r.항체, r.링커, n); os.makedirs(w, exist_ok=True)
        assert w.isascii(), f"작업 경로에 non-ascii 가 남았다: {w}"
        if not os.path.isfile(f"{w}/keep.json"):
            t0 = time.time()
            # 1차: 튜닝한 배치. 실패하면 2차: batch 1 (bioemu 기본, 입자 예외로 항상 통과)
            for _tgt in (BE_BATCH, 0):
                sh([P, "/content/be_driver.py", r.seq, w, str(n),
                    "1" if BE_STEER else "0", str(_tgt), str(BE_NP)], tail=10)
                if os.path.isfile(f"{w}/keep.json"): break
                if _tgt: print(f"    재시도 — batch 1 로 내린다 ({r.링커})")
            sec[(r.항체, r.링커)] = time.time() - t0
        if not os.path.isfile(f"{w}/keep.json"):
            print(f"  {label} {r.링커:<12} 실패 — 위 로그를 보라"); continue
        k = json.load(open(f"{w}/keep.json"))
        key = (r.항체, r.링커)                     # ★ 링커 이름만으로는 항체 사이에서
        RET[key] = k["retention"]                  #   겹친다 (블록마다 Linker1 이 있다)
        print(f"  {elapsed()} {label} {r.항체:<18} {r.링커:<12} "
              f"잔존 {k['retention']:.0%} / {k['n']}"
              + (f" · {sec.get(key, 0):.0f}s" if key in sec else ""))
        if keep_out:
            d = FL(r); os.makedirs(d, exist_ok=True)
            fr = sorted(glob.glob(f"{w}/f*.pdb"))
            for j, p in enumerate(fr):
                shutil.copy(p, f"{d}/{r.링커}__BioEmu_s{j}.pdb")
            # ★ 표식은 **절반 이상 살았을 때만** 남긴다. 3개만 저장돼도 표식이 남으면
            #   다음 실행이 "이미 있음" 으로 건너뛰고, 그 구성체는 영영 3프레임짜리로
            #   통계에 들어간다.
            if len(fr) >= n // 2:
                json.dump(dict(요청=n, 저장=len(fr), 잔존율=k["retention"]),
                          open(BEDONE(r), "w"))
            else:
                print(f"    ★ {len(fr)}/{n} 만 저장됐다 — 완료 표식을 남기지 않는다"
                      f" (다음에 다시 돈다)")
    return RET, sec

def retention_gate(RET, sec, n_probe, n_full):
    """잔존율·시간 게이트. 통과하면 (True, 외삽시간). 막히면 이유를 찍고 (False, _)."""
    if not RET:
        print("  ★ 중단 — 탐침이 한 건도 성공하지 못했다"); return False, None
    lo, hi = min(RET.values()), max(RET.values())
    est = sum(sec.values())/max(n_probe, 1)*n_full/3600 if sec else float("nan")
    print(f"\n  ── 탐침 ── 잔존율 {lo:.0%}~{hi:.0%} · 본실행 외삽 {est:.1f} h")
    if lo < RET_MIN:
        print(f"  ★ 중단 — 잔존율 {RET_MIN:.0%} 미만인 구성체가 있다. 검정이 포즈가 아니라"
              " 생존율을 재게 된다. BE_STEER 를 켜거나 BioEmu 를 접는다."); return False, est
    if hi - lo > RET_SPREAD:
        print(f"  ★ 중단 — 링커마다 잔존율이 {RET_SPREAD:.0%}p 넘게 다르다."
              " 검정이 **생존율 차이**를 재게 된다."); return False, est
    if est == est and est > BE_HOURS_MAX:
        print(f"  ★ 중단 — {BE_HOURS_MAX} h 초과. N_BE 를 줄여라."); return False, est
    return True, est

if RUN_BIOEMU:
    need = [r for _, r in CONS[~CONS.합성].iterrows() if not os.path.isfile(BEDONE(r))]
    print(f"  실측 패널 — 이미 있음 {int((~CONS.합성).sum())-len(need)}/{int((~CONS.합성).sum())}"
          + (f" · 남은 것 {len(need)}" if need else " — 건너뜀 (v09 가 끝냈다)"))
    if need:
        RET, sec = run_bioemu(need[:min(4, len(need))], "탐침", N_PROBE, keep_out=False)
        go, est = retention_gate(RET, sec, N_PROBE, N_BE)
        if go:
            per = np.mean(list(sec.values()))/max(N_PROBE, 1)*N_BE + 120 if sec else 900
            RET2, _ = run_bioemu(need, "본실행", N_BE, keep_out=True, per_s=per)
            if RET2: display(pd.DataFrame([dict(항체=k[0], 링커=k[1], 잔존율=round(v, 3))
                                           for k, v in RET2.items()]))
    _done = sum(1 for _, r in CONS[~CONS.합성].iterrows() if os.path.isfile(BEDONE(r)))
    print(f"{elapsed()} BioEmu 실측 완료 {_done}/{int((~CONS.합성).sum())} 구성체")

    # ── 씨앗 반복 — 잡음 바닥을 잰다 ────────────────────────────────────────
    # 같은 서열을 한 번 더 돌린다. 링커를 안 바꿨으니 나오는 차이는 전부 **실행 잡음**이다.
    # 이 값이 6절의 링커 간 차이보다 크면, 그 차이는 링커 효과가 아니다.
    if N_SEED_REP:
        _rep = CONS[~CONS.합성].drop_duplicates("항체").head(N_SEED_REP)
        for _, r in _rep.iterrows():
            r2 = r.copy(); r2["링커"] = f"{r.링커}__rep2"      # 별도 구성체로 저장
            if os.path.isfile(BEDONE(r2)):
                print(f"  씨앗 반복 이미 있음: {r.항체}/{r.링커}"); continue
            if not budget(1200, "씨앗 반복"): break
            print(f"  씨앗 반복 — {r.항체}/{r.링커} 를 한 번 더 돌린다 (잡음 바닥)")
            run_bioemu([r2], "반복", N_BE, keep_out=True, per_s=1200)
        # CONS 에 붙여 5·6절이 자동으로 같이 재게 한다
        _add = []
        for _, r in _rep.iterrows():
            r2 = r.copy(); r2["링커"] = f"{r.링커}__rep2"; r2["씨앗반복"] = True
            for c in OBS: r2[c] = np.nan            # 실측은 원본 쪽에만 붙인다
            if os.path.isfile(BEDONE(r2)) and r2.링커 not in set(CONS.링커):
                _add.append(r2)
        if "씨앗반복" not in CONS.columns: CONS["씨앗반복"] = False
        if _add:
            CONS = pd.concat([CONS, pd.DataFrame(_add)], ignore_index=True)
            CONS["씨앗반복"] = CONS.씨앗반복.fillna(False).astype(bool)
            print(f"  씨앗 반복 {len(_add)}건을 CONS 에 붙였다 — 6절이 잡음 바닥을 낸다")
else:
    print("BioEmu 건너뜀 (RUN_BIOEMU=False)")
if "씨앗반복" not in CONS.columns: CONS["씨앗반복"] = False
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 4-B절 · 합성 요인패널 — **이 노트북에서 제일 값진 셀**

### 실측 패널만으로는 원리적으로 못 푸는 것
실측 26건은 링커마다 길이가 하나씩이다 (G4S=15, Linker2=18, …).
그래서 "조성이 달라서 앙상블이 달라졌다" 와 "길이가 달라서 달라졌다" 가
**완전히 겹쳐 있다.** 어떤 통계도 이걸 못 푼다 — 설계가 그렇게 돼 있기 때문이다.

### 푸는 방법은 하나뿐 — 같은 길이에서 조성만 바꾼 칸을 만드는 것

|  | 10 aa | 15 aa | 20 aa | 25 aa | 이 줄이 가르는 것 |
|---|---|---|---|---|---|
| **G4S** `GGGGS` | ● | ● | ● | ● | 기준칸 |
| **polyG** `GGGGG` | ● | ● | ● | ● | G4S 에서 **Ser 하나만** 뺐다 — 조성 감도의 **하한** |
| **GS** `GGSGG` | ● | ● | ● | ● | G/S 비만 다름 |
| **EAAAK** `EAAAK` | ● | ● | ● | ● | 강직 α-나선 — **양성대조** |
| **PAS** `APAPA` | ● | ● | ● | ● | 신장형, 나선 아님 |
| **ED** `GEGGS` | ● | ● | ● | ● | 음전하 |
| **KR** `GKGGS` | ● | ● | ● | ● | 양전하. ED 와 길이·Gly 같고 **부호만 반대** |

- **가로 = 길이축** (조성 고정) → 학습된 `|i−j|` 채널을 잰다
- **세로 = 조성축** (길이 고정) → **조성 채널이 있는가**. 이게 본론이다
- 28칸 전부 **VH/VL 이 완전히 같고 링커 구간만 다르다**
- 전부 5-mer 라 10/15/20/25 로 깨끗이 타일링된다 → 길이축과 조성축이 **정확히 직교**

### 요인설계가 못 가르는 축 둘 — 따로 붙인다
- **순서 대조 (스크램블, 4런)**: 조성은 그대로 두고 **잔기 순서만** 뒤섞는다.
  모델이 모티프(반복 구조)를 읽는가, 아니면 아미노산 자루만 세는가.
- **씨앗 반복 (4절, 2런)**: 같은 서열을 두 번 돌린다. **이게 모든 비교의 분모다.**
  링커 간 차이가 실행 간 차이보다 작으면 그건 링커 효과가 아니다.
  v11 에는 이 대조가 아예 없었고, 그래서 v11 의 2×2 는 원리적으로 해석 불가였다
  (2×2 의 모든 칸이 서로 다른 서열이라, "링커 때문" 과 "다시 뽑아서" 를 가를 칸이
  하나도 없다 — v11 의 마크다운은 2×2 가 그걸 통제한다고 썼지만 통제하지 못한다).

### 값어치는 표본 수가 아니라 **직교성**이다
24건을 더해도 라벨 있는 구성체는 여전히 26건이다. 물성 회귀의 n 은 **안 늘어난다.**
늘어나는 것은 **설계의 해상도**다. 실측 패널에서 `조성 vs 길이` 상관은 |r| ≈ 1 이고,
이 패널에서는 **0** 이다. 그게 전부다. 그리고 그것만으로 충분하다 —
"BioEmu 가 조성을 보는가" 는 라벨이 아예 필요 없는 질문이기 때문이다.

### 하룻밤이면 끝난다
28 + 4 + 2 = 34건 × 10~16분 ≈ **6~9시간**. 그리고 이게 음성이면 **그 자리에서
연구가 끝난다** — 남은 상관 분석은 전부 "긴 링커가 나쁘다" 의 다른 표현이므로,
밤을 더 태울 이유가 없다. 그 결론을 하룻밤에 사는 것이 이 패널의 진짜 값어치다.

### 어디서 읽을 것인가 — 전역이 아니라 **국소**에서 먼저
BioEmu 논문이 **서열 민감도**를 실제로 보인 결과는 전부 국소 이차구조다:
Fig 3A·3B 의 이차구조 성향 일치, Fig 4F 의 단일 치환 Ile7→Pro 가 *그 잔기가 있는
나선의* 헬릭스 함량을 떨어뜨리는 것, 그리고 점돌연변이 50만 건에 ΔΔG Spearman > 0.6.
한 잔기 차이가 모델에 닿는다는 직접 증거다.

반대로 **"링커 조성이 두 도메인의 전역 배향을 바꾼다"는 논문 어디에도 없다.**
그러니 조성 채널을 ΔHL/Δdc 로만 재면 모델이 잘하는 자리에서 두 단계 떨어진 곳에서
재는 것이다. 6절-D 가 링커 **안**에서 직접 읽는다 — 그게 결정적인 칸이다.
''')

code(r'''
# ── 4-B절 · 합성 링커 요인패널 (조성 6 × 길이 4 = 24칸) ────────────────────
# 전부 5-mer 라 10/15/20/25 로 깨끗이 타일링된다.
MOTIF = {
    "G4S":   "GGGGS",   # 표준 유연 링커. 이 패널의 기준칸
    "polyG": "GGGGG",   # ★ G4S 에서 Ser 하나만 뺐다 — **조성 채널의 감도 하한**.
                        #   이것조차 안 갈리면 더 큰 조성차는 볼 필요도 없다.
    "GS":    "GGSGG",   # 유연하되 G/S 비만 다르다
    "EAAAK": "EAAAK",   # ★ 강직 α-나선 (Arai 2001). **양성대조** — 논문이 서열 민감도를
                        #   보인 자리(국소 이차구조)에서 가장 크게 갈려야 할 칸이다
    "PAS":   "APAPA",   # 프롤린-알라닌. 신장형, 나선 아님
    "ED":    "GEGGS",   # 음전하. G4S 에서 G 하나를 E 로
    "KR":    "GKGGS",   # 양전하. ED 와 길이·Gly 함량 같고 **전하 부호만 반대**
}
PANEL_LENGTHS = (10, 15, 20, 25)
PANEL_HOST    = None        # None 이면 가장 앞 블록의 G4S 구성체에 붙인다
# ★ 순서 대조(스크램블) — 조성은 그대로 두고 **잔기 순서만** 뒤섞는다.
#   모델이 모티프(반복 구조)를 읽는가, 아니면 아미노산 자루만 세는가를 가른다.
#   요인설계가 못 가르는 축이고, 4런이면 된다.
PANEL_SCRAMBLE = ["EAAAK15", "PAS15", "ED20", "KR20"]

if RUN_PANEL:
    _c = CONS[(~CONS.합성) & (CONS.링커.str.upper().str.contains("G4S"))]
    if not len(_c): _c = CONS[~CONS.합성]
    _h = _c.sort_values(["블록", "길이"]).iloc[0] if PANEL_HOST is None else \
         CONS[CONS.항체 == PANEL_HOST].iloc[0]
    print(f"패널 숙주: 블록 {_h.블록} · {_h.항체} · {_h.링커} ({_h.길이}aa) · 배향 {_h.배향}")

    _new = []
    for nm, m in MOTIF.items():
        for Lp in PANEL_LENGTHS:
            s  = (m * (Lp // len(m) + 1))[:Lp]
            lk = f"P_{nm}{Lp}"                      # 실측 링커와 절대 안 겹치는 접두어
            if lk in set(CONS.링커): continue       # 다시 돌려도 안 겹친다
            r = _h.copy()
            r["링커"], r["링커서열"] = lk, s
            r["b2"]   = int(_h.b1) + Lp
            r["길이"] = Lp
            r["seq"]  = _h.seq[:int(_h.b1)] + s + _h.seq[int(_h.b2):]
            r["합성"], r["조성"], r["설계길이"] = True, nm, Lp
            # ★ 합성 구성체는 **실측이 없다.** 숙주 행을 복사해 만들었으므로 숙주의
            #   실측값을 그대로 물고 있다. 하나라도 남으면 7절이 그것을 진짜 데이터
            #   3~24개로 세고, 같은 y 값에 점이 여러 개 쌓여 상관을 끌어당긴다.
            #   → 1절에서 만든 **같은 OBS 목록**으로 전부 비운다 (v11 은 여기 키워드가
            #     로더와 달라서 Monomer·Titer·SEC 가 안 비워졌다).
            for c in OBS: r[c] = np.nan
            _new.append(r)
    # ── 순서 대조 — 조성 고정, 순서만 섞는다 ────────────────────────────
    _sc = np.random.default_rng(0)
    for nm in PANEL_SCRAMBLE:
        src = next((r for r in _new if r["링커"] == f"P_{nm}"), None)
        if src is None: continue
        lk = f"P_{nm}_scr"
        if lk in set(CONS.링커): continue
        seq_l = list(src["링커서열"]); _sc.shuffle(seq_l); sseq = "".join(seq_l)
        if sseq == src["링커서열"]: continue          # 우연히 그대로면 버린다
        r = src.copy()
        r["링커"], r["링커서열"] = lk, sseq
        r["seq"] = _h.seq[:int(_h.b1)] + sseq + _h.seq[int(_h.b2):]
        r["조성"] = str(src["조성"]) + "_scr"
        _new.append(r)
    if _new:
        CONS = pd.concat([CONS, pd.DataFrame(_new)], ignore_index=True)
    for c in ("조성", "설계길이"):
        if c not in CONS.columns: CONS[c] = np.nan
    assert not CONS.loc[CONS.합성, OBS].notna().any().any(), \
        "합성 구성체에 실측값이 남아 있다 — OBS 목록을 확인하라"

    P = CONS[CONS.합성]
    print(f"  합성 구성체 {len(P)}개 (조성 {P.조성.nunique()} × 길이 {P.설계길이.nunique()})")
    display(P.pivot_table(index="조성", columns="설계길이", values="길이",
                          aggfunc="first").fillna("—"))

    # ★ 직교성 확인 — 이 패널의 존재 이유다. 실측 패널과 나란히 찍는다.
    _gf = lambda s: sum(str(s).count(a) for a in "G") / max(len(str(s)), 1)
    _pn = P.assign(G분율=P.링커서열.map(_gf))
    _rl = CONS[~CONS.합성].assign(G분율=CONS[~CONS.합성].링커서열.map(_gf))
    _r1 = abs(np.corrcoef(_pn.G분율, _pn.길이)[0, 1])
    _r2 = abs(np.corrcoef(_rl.G분율, _rl.길이)[0, 1]) if _rl.G분율.std() > 1e-9 else 1.0
    print(f"\n  조성(G분율) vs 길이  |r|:   합성패널 {_r1:.2f}   실측패널 {_r2:.2f}")
    print("  ← 이 차이가 패널의 전부다. 실측 쪽은 조성축이 아예 없다.")

    need = [r for _, r in CONS[CONS.합성].iterrows() if not os.path.isfile(BEDONE(r))]
    print(f"\n  이미 있음 {len(P)-len(need)}/{len(P)}"
          + (f" · 남은 것 {len(need)} → 약 {len(need)*13/60:.1f} h" if need else " — 건너뜀"))
    if need:
        RET, sec = run_bioemu(need[:3], "패널탐침", N_PROBE, keep_out=False)
        go, est = retention_gate(RET, sec, N_PROBE, N_BE)
        if go:
            per = np.mean(list(sec.values()))/max(N_PROBE, 1)*N_BE + 120 if sec else 900
            run_bioemu(need, "패널", N_BE, keep_out=True, per_s=per)
    _d = sum(1 for _, r in CONS[CONS.합성].iterrows() if os.path.isfile(BEDONE(r)))
    print(f"{elapsed()} 합성 패널 완료 {_d}/{len(P)}")
else:
    for c in ("조성", "설계길이"):
        if c not in CONS.columns: CONS[c] = np.nan
    print("합성 패널 건너뜀 (RUN_PANEL=False) — 4-B절 채널 판정을 못 한다")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 5절 · ABangle → `frames.csv` · 기하 → `geom.csv`

구조 하나당 한 줄짜리 표 두 개를 만든다. 6절이 이 둘을 접어서 구성체당 한 줄로 만든다.

| | 만드는 것 | 비용 |
|---|---|---|
| `frames.csv` | ΔABangle 6축 · OCD6 · 짝지음 | write_pair + ABangle. 5,000구조에 ~7시간 |
| `geom.csv` | 링커-도메인 접촉 · 링커 Rg · 신장도 | 좌표만 읽는다. 5,000구조에 ~3분 |

★ `frames.csv` 는 **v09 의 `flow.csv` 와 같은 것**이다. 있으면 그대로 읽고 다시 안 잰다.
새 구조(합성 패널)를 더했으면 **그 구성체만** 덧붙여 잰다 —
v11 은 하나라도 모자라면 5,854구조를 통째로 다시 쟀다(412분).

### 기준점은 그 **항체의** ABB2 4모델 평균이다
Δ 의 영점이라 항체마다 따로 잡는다. 그리고 기준점 쌍 PDB도 구성체와 **같은
`write_pair`** 를 지난다 — 한쪽만 다른 경로로 쓰면 Δ 에 구성체마다 다른
상수 치우침이 생겨서, 그 치우침이 곧 '링커 효과' 로 읽힌다.
''')

code(r'''
# ── 5절 · ABangle 수집 → frames.csv ────────────────────────────────────────
# REUSE_CSV 는 1절 스위치 칸에 있다 (3절-B 가 먼저 쓴다).
CSV_FRAMES = f"{OUT}/frames.csv"
CSV_LEGACY = f"{OUT}/flow.csv"    # v09 가 쓰던 이름 — 있으면 그대로 읽는다
MET = ["OCD6"] + [f"Δ{k}" for k in AB6]

def _pairs_needed(r):
    """구성체 r 의 앙상블 프레임 목록. 파일명 끝의 _s{번호} 로 **수치 정렬**한다 —
    문자열 정렬이면 s10 이 s2 보다 앞에 와서 프레임 순서가 뒤섞이고,
    6절-E 의 배치 복원과 9절의 MODEL 순서가 어긋난다."""
    ps = [p for p in glob.glob(f"{FL(r)}/{r.링커}__*.pdb")
          if "_pair" not in p and not os.path.basename(p).startswith("_p_")]
    def _k(p):
        m = re.search(r"_s(\d+)\.pdb$", p)
        return (0, int(m.group(1))) if m else (1, 0)
    return sorted(ps, key=_k)

# ── 기존 표 읽기 ────────────────────────────────────────────────────────────
E = None
if REUSE_CSV:
    for _p in (CSV_FRAMES, CSV_LEGACY):
        if os.path.isfile(_p):
            E = pd.read_csv(_p)
            print(f"기존 표 {os.path.basename(_p)} — {len(E)}행 · "
                  f"구성체 {E.groupby(['항체','링커']).ngroups}종")
            break
_have = set(zip(E.항체, E.링커)) if E is not None and len(E) else set()
_want = {(r.항체, r.링커) for _, r in CONS.iterrows() if _pairs_needed(r)}
_todo = sorted(_want - _have)
print(f"  덮는 구성체 {len(_want & _have)}/{len(_want)}"
      + (f" · 새로 잴 것 {len(_todo)}: {[t[1] for t in _todo][:8]}"
         f"{' …' if len(_todo) > 8 else ''}" if _todo else " — 전부 있다"))

# ── 모자란 구성체만 잰다 (v11 은 하나라도 모자라면 전부 다시 쟀다) ──────────
if _todo:
    # ★ 쌍 PDB 는 **순수 중간산물**이다. Drive 에 쓰면 안 된다.
    #   v11 은 프레임마다 한 개씩 Drive 에 썼다 — 구성체 26개 × 150프레임 ≈ 3,900개.
    #   Google Drive 는 작은 파일 하나하나에 왕복이 붙어서, **이게 그 412분의 정체다.**
    #   (앙상블 PDB 자체는 Drive 에 남아야 한다. 쌍 PDB 는 ABangle 에 넘기고 버린다.)
    PAIRD = "/content/_pair"
    os.makedirs(PAIRD, exist_ok=True)
    R0C = {}
    def _R0(ab, r):
        """기준점 = 그 **항체의** ABB2 4모델 ABangle 평균."""
        if ab in R0C: return R0C[ab]
        dd = REFD(ab); os.makedirs(f"{PAIRD}/{safe(ab)}", exist_ok=True)
        refs = sorted(glob.glob(f"{dd}/ref_m[0-9].pdb")) or [f"{dd}/ref.pdb"]
        pp = []
        for p in refs:
            if not os.path.isfile(p): continue
            # 한 모델이 못 읽혀도 나머지 3개로 기준점을 잡는다 — 여기서 예외를
            # 안 막으면 기준 구조 한 건 때문에 5절이 통째로 멈춘다.
            try:
                pp.append(write_pair(p, r.b1, r.b2,
                                     f"{PAIRD}/{safe(ab)}/ref_{os.path.basename(p)}",
                                     r.배향[0], r.배향))
            except Exception as e:
                print(f"  ★ 기준 구조 건너뜀 {os.path.basename(p)}: {e}")
        a_ = [x for x in abangle6_many(pp, quiet=True).values() if x] if pp else []
        R0C[ab] = {k: float(np.mean([x[k] for x in a_])) for k in a_[0]} if a_ else None
        if R0C[ab] is None: print(f"  ★ 기준점 ABangle 실패: {ab}")
        return R0C[ab]

    # ★ 실측 열을 **전부** 싣는다. HC(확증 열) 하나만 실으면 수율·단량체가 여기서
    #   사라지고, 6절 이후로는 존재 자체를 알 수 없게 된다. README 의 고침 #5 가
    #   바로 그 사고였는데 같은 일이 이 줄에서 다시 일어날 수 있었다.
    #   수율은 HMW 와 **다른 실패 모드**다 (실측에서 항체 안 rho 가 0.00·−0.40·
    #   +0.87·−0.80·0.00 로 관계가 없다). 버리면 안 된다.
    _meta = lambda r: {c: r[c] for c in OBS if c in CONS.columns and pd.notna(r[c])}
    jobs, _t0 = [], time.time()
    for _, r in CONS.iterrows():
        if (r.항체, r.링커) not in set(_todo): continue
        if _R0(r.항체, r) is None: continue
        w = f"{PAIRD}/{safe(r.항체)}"; os.makedirs(w, exist_ok=True)
        for p in _pairs_needed(r):
            b = os.path.basename(p).rsplit(".", 1)[0]
            try: q = write_pair(p, r.b1, r.b2, f"{w}/{b}.pdb", r.배향[0], r.배향)
            except Exception: continue
            jobs.append((q, dict(블록=r.블록, 항체=r.항체, 배향=r.배향, 링커=r.링커,
                                 길이=int(r.길이), 합성=bool(r.합성),
                                 조성=r.get("조성", np.nan),
                                 설계길이=r.get("설계길이", np.nan),
                                 생성기=b.split("__")[1].split("_")[0],
                                 태그=b, **_meta(r))))
        if len(jobs) % 500 < 2:
            print(f"    쌍 PDB {len(jobs)}개 · {time.time()-_t0:.0f}s", flush=True)
    print(f"쌍 PDB {len(jobs)}개 → ABangle 배치 실행")
    AB = abangle6_many([q for q, _ in jobs])
    rows = []
    for q, m in jobs:
        a, R0 = AB.get(q), R0C.get(m["항체"])
        if not a or not R0: continue
        # ★ **원시 6값을 같이 저장한다.** v11 은 Δ 만 남겨서, 짝지음 기준이나
        #   요약 지표를 바꾸려면 ABangle 을 통째로 다시 돌려야 했다 (412분).
        #   원시값이 있으면 아래 정의를 전부 csv 만으로 다시 짤 수 있다.
        rows.append(dict(m, **{k: round(a[k], 3) for k in AB_SD if k in a},
                         **{f"Δ{k}": round(a[k]-R0[k], 3) for k in R0 if k in AB_SD},
                         # OCD5 = **dc 를 뺀** 5축 합. 짝지음 게이트가 dc 로 거르므로
                         #   dc 가 든 지표를 같이 비교하면 '결과의 성분으로 표본을 고른'
                         #   꼴이 된다. 각도 비교의 주지표는 OCD5 여야 한다.
                         OCD5=round(sum(abs(a[k]-R0[k])/AB_SD[k]
                                        for k in R0 if k in AB_SD and k != "dc"), 3),
                         OCD6=round(sum(abs(a[k]-R0[k])/AB_SD[k]
                                        for k in R0 if k in AB_SD), 3)))
    NEW = pd.DataFrame(rows)
    print(f"{elapsed()} ABangle 성공 {len(NEW)}/{len(jobs)}")
    E = NEW if E is None else pd.concat([E, NEW], ignore_index=True)
    E = E.drop_duplicates(["항체", "링커", "태그"], keep="last")
    E.to_csv(CSV_FRAMES, index=False, encoding="utf-8-sig")
    # 기준점 각도도 남긴다 — Δ 를 다시 정의하고 싶을 때 ABangle 없이 할 수 있다
    pd.DataFrame([dict(항체=ab, **v) for ab, v in R0C.items() if v]).to_csv(
        f"{OUT}/ref_angles.csv", index=False, encoding="utf-8-sig")
    print(f"  저장 {CSV_FRAMES} — {len(E)}행 · 기준점 {OUT}/ref_angles.csv")
    shutil.rmtree(PAIRD, ignore_errors=True)      # 중간산물은 버린다

assert E is not None and len(E), \
    "frames.csv 가 비었다 — 4절을 먼저 돌리거나 위 ABangle 실패 메시지를 보라"
# 합성/조성 열이 옛 flow.csv 에는 없다. CONS 에서 붙여 준다.
_key = CONS.set_index(["항체", "링커"])[["합성", "조성", "설계길이", "길이"]]
for c in ("합성", "조성", "설계길이"):
    if c not in E.columns:
        E[c] = pd.MultiIndex.from_arrays([E.항체, E.링커]).map(_key[c])
E["합성"] = E["합성"].fillna(False).astype(bool)
display(E.groupby(["생성기", "합성"]).size().unstack(fill_value=0))
''')

code(r'''
# ── 5절-B · 좌표 기하 → geom.csv ───────────────────────────────────────────
# ABangle 은 도메인 **배향**만 본다 — 링커가 어디에 붙어 있는지는 모른다.
# Okazaki 의 Glue-linker 기전(링커가 VH-VL 홈에 달라붙어 closed 를 안정화)은
# 배향이 아니라 **접촉**으로 나타난다. 그래서 좌표를 한 번 더 훑는다.
# 비용: 좌표만 읽으므로 5,000구조에 ~3분. ABangle 의 7시간과 비교가 안 된다.
CSV_GEOM    = f"{OUT}/geom.csv"
CONTACT_CUT = 10.0        # CA-CA Å. 곁사슬이 없으니 CB 가 아니라 CA (GLY 는 CB 도 없다)
GEOM_FEATS  = ["링커접촉_잔기당", "링커밀착율", "링커최근접",
               "링커Rg_잔기당", "링커신장도", "링커나선도", "링커i_i4",
               "도메인Rg", "말단간"]
# ── 링커 국소 이차구조 — **여기가 BioEmu 가 실제로 서열에 민감한 자리다** ──────
# Lewis et al. 2025 가 서열 민감도를 보여준 결과는 전부 **국소 이차구조**다:
#   · Fig 3A "excellent agreement of the predicted secondary structure propensities"
#   · Fig 3B "Most secondary structure propensities matched well"
#   · Fig 4F 단일 치환 Ile7→Pro 가 **그 잔기가 있는 나선의** 헬릭스 함량을 떨어뜨린다
#   · Complexin II 의 중심·보조 나선을 재현한다
# 반대로 "링커 조성이 두 도메인의 **전역 배향**을 바꾼다" 는 논문에 없는 주장이다.
# 그래서 조성 채널을 ΔHL/Δdc 로만 읽으면, 모델이 잘하는 자리에서 두 단계나 떨어진
# 곳에서 재는 것이 된다. 링커 **안**에서 직접 읽어야 한다.
#
# 곁사슬이 없어 DSSP 를 못 쓰므로 CA 만으로 나선을 잰다:
#   α-나선은 d(i, i+4) ≈ 6.2 Å,  신장 사슬은 ≈ 13 Å,  무작위 코일은 그 사이.
#   5.0~7.5 Å 창에 드는 비율이 곧 **나선도**다 (CA 기반의 표준 판별식이다).
HELIX_LO, HELIX_HI = 5.0, 7.5

def frame_geometry(ca, b1, b2, cut=CONTACT_CUT):
    """프레임 하나의 기하. **전부 잔기당**으로 낸다 — 길이로 안 나누면
    무엇이든 길이의 다른 이름이 되기 때문이다."""
    n = len(ca)
    if n < b2 + 5 or b2 - b1 < 3: return {}
    L, D = ca[b1:b2], np.vstack([ca[:b1], ca[b2:]])
    nl = b2 - b1
    Dm = np.linalg.norm(L[:, None, :] - D[None, :, :], axis=-1)
    near = Dm.min(1)
    Lc, Dc = L - L.mean(0), D - D.mean(0)
    # 링커 안의 i→i+4 CA 거리. 나선이면 ~6.2 Å, 신장이면 ~13 Å.
    d4 = (np.linalg.norm(L[4:] - L[:-4], axis=-1) if nl > 4 else np.array([]))
    return dict(
        링커접촉_잔기당=float((Dm < cut).sum() / nl),   # 계면 홈에 걸칠 때 최대가 된다
        링커밀착율=float((near < cut).mean()),          # 도메인에 닿은 링커 잔기 비율
        링커최근접=float(np.median(near)),
        링커Rg_잔기당=float(np.sqrt((Lc**2).sum(1).mean()) / nl),
        링커신장도=float(np.linalg.norm(L[-1]-L[0]) / (3.8*(nl-1))),   # 1.0 = 완전신장
        # ★ 국소 이차구조 — 논문이 서열 민감도를 실제로 보인 자리
        링커나선도=float(((d4 > HELIX_LO) & (d4 < HELIX_HI)).mean()) if len(d4) else np.nan,
        링커i_i4=float(np.median(d4)) if len(d4) else np.nan,
        도메인Rg=float(np.sqrt((Dc**2).sum(1).mean())),
        말단간=float(np.linalg.norm(ca[-1] - ca[0])),
    )

# ★ 태그(`{링커}__BioEmu_s0`)는 **항체 사이에서 유일하지 않다** — 블록이 달라도
#   링커 이름이 같으면 태그가 겹친다. 그래서 키는 반드시 (항체, 태그) 여야 한다.
#   태그만으로 합치면 블록 1 의 Linker1 기하가 블록 2~5 의 Linker1 에도 붙는다
#   (조용히 틀린 값이 들어가고, 아무 오류도 안 난다).
G = pd.read_csv(CSV_GEOM) if (REUSE_CSV and os.path.isfile(CSV_GEOM)) else pd.DataFrame()
_gh = set(zip(G.항체, G.태그)) if len(G) else set()
rows, _t0 = [], time.time()
for _, r in CONS.iterrows():
    fr = [p for p in _pairs_needed(r)
          if (r.항체, os.path.basename(p).rsplit(".", 1)[0]) not in _gh]
    if not fr: continue
    ok = 0
    for p in fr:
        try: g = frame_geometry(ca_xyz(p), int(r.b1), int(r.b2))
        except Exception: continue
        if not g: continue
        rows.append(dict(항체=r.항체, 링커=r.링커,
                         태그=os.path.basename(p).rsplit(".", 1)[0], **g)); ok += 1
    print(f"  {r.항체:<20} {r.링커:<10} 기하 {ok}/{len(fr)}")
if rows:
    G = pd.concat([G, pd.DataFrame(rows)], ignore_index=True) if len(G) else pd.DataFrame(rows)
    G = G.drop_duplicates(["항체", "태그"], keep="last")     # ★ 태그 하나로는 안 된다
    G.to_csv(CSV_GEOM, index=False, encoding="utf-8-sig")
    print(f"{elapsed()} 기하 {len(rows)}건 추가 → {CSV_GEOM} (총 {len(G)}행, "
          f"{time.time()-_t0:.0f}s)")
else:
    print(f"기하 전부 있음 — {len(G)}행")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 6절 · 특징표 — 프레임 표 → **구성체당 한 줄**

여기가 파이프라인과 추론이 만나는 자리다. 접는 **방식**이 곧 논지다.
가설은 "링커가 평균을 옮긴다"가 아니라 **"링커가 분포를 바꾼다"** 이므로
중앙값만 내면 가설을 시험하는 게 아니다.

### 짝지음(`|Δdc| < 2 Å`)을 두 가지로 쓴다
1. **인구 분율로** — `열림분율` = 안 붙은 프레임의 비율. **이게 확증 특징이다.**
   HMW:Monomer 가 인구 비율이니 계산도 인구 비율이어야 한다.
2. **거르개로** — 각도 요약(`OCD6중앙` 등)은 **붙은 프레임만**으로 낸다.

★ 2번이 꼭 필요하다. 떨어진 Fv 에서는 ABangle 의 각도 4개가 정의를 잃는다
(붙어 있는 계면을 전제로 정의된 양이다). 섞어서 중앙값을 내면 그 중앙값은
각도가 아니라 **'떨어진 프레임이 몇 개냐'** 를 재게 되고, 그러면
`OCD6중앙` 이 `열림분율` 의 잡음 섞인 복제본이 되어 두 특징이 같은 것이 된다.

### 사전 등록 — 특징 넷, 그 이상은 안 된다
독립 단위가 블록 5개뿐이다. 특징을 늘리면 무엇이 나와도 우연이다.

| | 특징 | 왜 |
|---|---|---|
| **확증** | `열림분율` | 인구 분율. HMW 와 같은 종류의 양이다 |
| 탐색 | `OCD6중앙` | 기준점에서 얼마나 떨어져 있나 |
| 탐색 | `OCD6_MAD` | 분포가 얼마나 넓나 — "분포" 가설의 직접 대응물 |
| 탐색 | `링커접촉_잔기당` | **조성**을 나를 수 있는 유일한 후보 (Glue 기전) |
| 기준선 | `길이` | 증분 검정의 귀무모형 |

### 그리고 두 개의 원장을 먼저 찍는다
- **교란 원장** — 각 특징이 길이의 변장인가. 블록 안 `|r| > 0.9` 면 그렇다.
- **특징 잡음 원장** — 특징은 150 프레임의 통계량이라 오차가 있다. 그 오차가
  구성체 간 흩어짐에 견줄 만큼 크면 상관이 0 쪽으로 눌린다
  (errors-in-variables 감쇠). "상관이 없다" 와 "특징이 시끄럽다" 를 여기서 가른다.
''')

code(r'''
# ── 6절 · 특징표 ───────────────────────────────────────────────────────────
from itertools import combinations

PRIMARY     = "열림분율"                                    # 확증 특징 (1개)
# ★ 탐색 특징의 각도 지표를 OCD6 이 아니라 **OCD5(dc 제외)** 로 쓴다.
#   dc 의 자연 σ 가 0.268 Å 뿐이라 |Δdc| 1 Å 만으로 OCD6 에 3.7 이 더해진다.
#   나머지 다섯 축은 각각 2~3 을 넘는 일이 드물다 → OCD6 ≈ |Δdc|/0.268 + 잡음이고,
#   '6축 종합' 이라는 이름과 달리 사실상 **도메인 중심거리 하나**를 읽는 값이다.
#   게다가 짝지음 게이트가 바로 그 dc 로 프레임을 거른다 — 결과의 성분으로 표본을
#   고른 셈이라, 많이 열린 링커일수록 남은 프레임의 OCD6 가 **낮게** 보일 수 있다.
#   OCD5 로 가면 확증 특징(열림분율, dc 기반)과 각도 특징이 서로 독립이 된다.
EXPLORATORY = ["OCD5중앙", "OCD5_MAD", "링커접촉_잔기당"]     # 탐색 특징 (3개, Holm 보정)
# ★ 꼬리 특징군 — 사전 등록 **밖**이다. 원장과 분산 분해에만 쓰고, 검정하면
#   반드시 Holm 으로 묶는다. (실측에 한 번 돌려 봤고 전부 귀무였다: 최고 p 0.22,
#   Holm 후 생존 없음. 그래도 남겨 둔다 — 설계가 바뀌면 다시 물을 축이다.)
TAIL_FEATS  = ["OCD5_p90", "OCD5_p95", "OCD5_p99", "열림_dc4", "열림_dc6",
               "열림_dc10", "안붙음_심도", "붙음중_꼬리10", "붙음중_꼬리15",
               "붙음중_꼬리20", "Rg중앙", "Rg_p90", "열림_Rg"]
# ★ 좌표 기반 열림 임계 — 전체 프레임 도메인Rg 의 상위 25%. 6절에서 채운다.
RG_OPEN_CUT = None
BASELINE    = ["길이"]                                      # 증분 검정의 귀무모형
GEN         = "BioEmu"

def mad(v):
    """중앙값 절대편차 × 1.4826 (정규분포에서 σ 와 같아지게 맞춘 상수).
    표준편차보다 꼬리에 강해서 프레임 몇 개가 튀어도 안 흔들린다."""
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))

def outside_2sigma(g, ax=("ΔHL", "Δdc")):
    """SAbDab 자연 2σ 타원 **밖**의 프레임 비율 — 2차원 마할라노비스 반경 > 2.

    ★ 이름의 '2σ' 는 1차원 감각과 다르다. 2차원에서 반경 2 안쪽은
      1 − exp(−2) = **86.5%** 이므로, 이 값은 귀무 하에서 0.135 근처가 기본값이다.
      (0.05 가 아니다. 차원이 늘수록 껍질에 질량이 몰리므로 6차원에서 반경 2 는
       질량의 68% 를 '밖' 으로 셀 만큼 관대해진다 — 그래서 여기서는 2차원만 쓴다.)
    """
    sx, sy = AB_SD[ax[0][1:]]*2, AB_SD[ax[1][1:]]*2
    return float((((g[ax[0]]/sx)**2 + (g[ax[1]]/sy)**2) > 1).mean())

def w1(a, b):
    """1차 Wasserstein 거리 — 두 분포가 얼마나 다른가를 **원래 단위로** 준다.
    2차원 엔트로피나 KL 은 n=150 에서 편향이 크다 (2D 격자면 칸당 0.2개).
    1D Wasserstein 은 경험분포만으로 일치추정량이라 이 표본 크기에서 믿을 수 있다."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 20 or len(b) < 20: return np.nan
    return float(st_.wasserstein_distance(a, b))

def paired(df):
    """VH-VL 이 붙어 있는가 — **절대 dc** 로 판정한다.
    dc 열이 없는 옛 flow.csv 면 Δdc 로 물러선다 (그때는 기준점이 특이한 항체에서
    '붙었다' 의 뜻이 구성체마다 조금씩 달라진다는 것을 알고 써야 한다)."""
    if "dc" in df.columns and df["dc"].notna().any():
        return (df["dc"] - DC0).abs() < PAIR_NSD * DC_SD
    return df["Δdc"].abs() < PAIR_DC

def abangle_sane(df, dc_big=10.0, rg_slack=2.0):
    """★★ ABangle 값이 **좌표와 같은 이야기를 하는가.** 프레임별 True/False.

    왜 필요한가 — 이게 이 노트북에서 제일 위험한 실패다.
    상류 `number_sequences` 에 zip 어긋남이 있어서 ANARCI 가 한 사슬만 인식하면
    다른 사슬은 번호가 안 매겨진다. 그런데 `write_pair` 가 1..N 으로 빈틈없이
    번호를 매기므로 **번호 없는 사슬도 coreset 35개를 그대로 채운다.**
    → Superimposer 성공 → 여섯 개의 그럴듯하지만 **무의미한 실수**가 나온다.
    오류가 안 나므로 조용히 통과한다. (검증 CLI 로 막았지만 그건 사슬 인식만 본다.)

    실측에서 실제로 이랬다:
      |Δdc| > 45 인 프레임의 도메인Rg 중앙 = 18.4 Å
      |Δdc| < 2  인 프레임의 도메인Rg 중앙 = 17.6 Å
    두 Ig 도메인(각 Rg≈13Å)의 중심이 54 Å 벌어지면 Rg 합은 √(13²+27²) = 30 Å 여야 한다.
    17~18 Å 이 그대로 나왔다 = **안 벌어졌다.** 그런데 |ΔHL| 중앙이 106° 였다.
    붙어 있는 도메인이 106도 비틀릴 수는 없다. 여섯 값이 **같이** 망가진 것이다.

    더 나쁜 것은 방향이 뒤집힌다는 점이다 — 진짜로 열린 프레임(도메인Rg 23.6,
    말단간 73 Å)이 Δdc 0.65 로 '붙음' 에 들어가 있었다.

    판정: dc 가 크게 벗어났다고 말하는데 좌표는 그대로면 그 프레임을 버린다.
    기준선은 **그 구성체의 얌전한 프레임**에서 잡는다 (항체마다 도메인 크기가 다르다).
    """
    if "도메인Rg" not in df.columns or df["도메인Rg"].isna().all():
        return pd.Series(True, index=df.index)      # 좌표 기하가 없으면 검산 못 한다
    adc = (df["dc"] - DC0).abs() if ("dc" in df.columns and df["dc"].notna().any()) \
        else df["Δdc"].abs()
    rg = pd.to_numeric(df["도메인Rg"], errors="coerce")
    ok = pd.Series(True, index=df.index)
    for _, idx in df.groupby(["항체", "링커"], sort=False).groups.items():
        a, r = adc.loc[idx], rg.loc[idx]
        base = r[a < 2.0].median()
        if not np.isfinite(base):
            base = r.median()
        # dc 는 크게 벗어났다는데 좌표상 도메인은 그대로 → ABangle 이 거짓말이다
        ok.loc[idx] = ~((a >= dc_big) & (r < base + rg_slack))
    return ok

def build_features(frames, geom, gen=GEN, pair_dc=PAIR_DC):
    """프레임 표 → 구성체당 한 줄."""
    Gf = frames[frames.생성기 == gen].copy()
    assert len(Gf), (f"생성기 '{gen}' 프레임이 없다. frames.csv 의 생성기: "
                     f"{sorted(frames.생성기.unique())}")
    if "OCD5" not in Gf.columns:       # 옛 flow.csv 호환 — Δ 만 있으면 여기서 만든다
        Gf["OCD5"] = sum(Gf[f"Δ{k}"].abs()/AB_SD[k] for k in AB6 if k != "dc")
    Gf["짝지음"] = paired(Gf)
    if geom is not None and len(geom):
        # ★ (항체, 태그) 로 합친다. 태그만 쓰면 같은 링커 이름을 가진 다른 블록의
        #   기하가 섞여 들어온다 — 오류 없이 조용히 틀린 값이 된다.
        Gf = Gf.merge(geom.drop(columns=["링커"], errors="ignore"),
                      on=["항체", "태그"], how="left", validate="many_to_one")
        # ★★ 좌표와 대조해 ABangle 이 망가진 프레임을 버린다. 병합 **뒤**여야 한다 —
        #    도메인Rg 가 있어야 검산이 된다.
        _sane = abangle_sane(Gf)
        ABANGLE_DROP = int((~_sane).sum())
        if ABANGLE_DROP:
            print(f"  ★★ ABangle 이 좌표와 어긋나는 프레임 {ABANGLE_DROP}/{len(Gf)} "
                  f"({ABANGLE_DROP/len(Gf):.0%}) 를 버린다.")
            print("     dc 는 크게 벗어났다는데 도메인Rg 는 그대로인 프레임들이다 —")
            print("     ANARCI 가 한 사슬만 인식했을 때 나오는 **무성 오염**이다.")
            print("     (오류를 안 내므로 안 버리면 그대로 통계에 들어간다.)")
            Gf = Gf[_sane].copy()
        else:
            print("  ABangle–좌표 검산 통과 — 버린 프레임 없다.")
    global RG_OPEN_CUT
    if "도메인Rg" in Gf.columns and Gf["도메인Rg"].notna().any():
        RG_OPEN_CUT = float(np.nanpercentile(Gf["도메인Rg"].astype(float), 75))
        print(f"  좌표 기반 열림 임계: 도메인Rg > {RG_OPEN_CUT:.1f} Å (전체 상위 25%)")
    meta = ([c for c in ("블록", "항체", "배향", "링커", "길이", "합성", "조성",
                         "설계길이") if c in Gf.columns]
            + [c for c in OBS if c in Gf.columns])      # ★ 실측 열 전부 (HC 만이 아니다)
    rows = []
    for (ab, lk), g in Gf.groupby(["항체", "링커"], sort=False):
        d = {k: g[k].iloc[0] for k in meta}
        d.update(항체=ab, 링커=lk, n프레임=len(g), n붙음=int(g.짝지음.sum()))
        # ── 확증: 인구 분율 ────────────────────────────────────────────
        d[PRIMARY] = float(1 - g.짝지음.mean())
        # ── 각도 요약은 **붙은 프레임만** ──────────────────────────────
        c = g[g.짝지음]
        src = c if len(c) >= 20 else g
        d["붙음부족"] = len(c) < 20
        for m_ in ("OCD5", "OCD6"):
            d[f"{m_}중앙"] = float(np.median(src[m_]))
            d[f"{m_}_MAD"] = mad(src[m_])
        d["자연2σ밖"] = outside_2sigma(src)
        # ── ★ 꼬리 특징군 — "100개 중 튀는 5개" 를 직접 재는 것들 ────────────
        #   기전은 **소수 집단**이다: 대부분은 얌전하고 일부만 열려서 이량체를
        #   만든다. 그러면 중앙값은 그 소수를 정의상 못 본다. 그래서 따로 잰다.
        #   ※ 실제로 이 앙상블은 이봉분포다 — 붙음 중앙 OCD5 ≈ 6, 안붙음 ≈ 70.
        #     게이트가 임의로 자른 게 아니라 실재하는 두 덩어리를 가른다.
        # ── ★ ABangle 을 **안 거치는** 열림 지표 — 좌표에서 직접 온다 ──────────
        #   ABangle 이 무성 오염되면 위의 짝지음/각도는 전부 못 믿는다.
        #   도메인Rg 와 말단간은 CA 좌표에서 바로 나오므로 번호매김과 무관하다.
        #   실측에서 rho(도메인Rg, 말단간) = +0.56 으로 물리적으로 맞고,
        #   rho(도메인Rg, |Δdc|) = +0.06 으로 ABangle 의 dc 와는 무관했다.
        if "도메인Rg" in g.columns and g["도메인Rg"].notna().any():
            _rg = pd.to_numeric(g["도메인Rg"], errors="coerce").dropna().values
            d["Rg중앙"] = float(np.median(_rg))
            d["Rg_p90"] = float(np.percentile(_rg, 90))
            d["열림_Rg"] = float((_rg > RG_OPEN_CUT).mean()) if RG_OPEN_CUT else np.nan
        _o = g["OCD5"].values.astype(float)
        for _p in (90, 95, 99):
            d[f"OCD5_p{_p}"] = float(np.nanpercentile(_o, _p))
        # 더 엄한 열림 — '안 붙음' 보다 **얼마나 멀리 갔나**
        _dc = (g["dc"] - DC0).abs() if "dc" in g.columns else g["Δdc"].abs()
        for _t in (4.0, 6.0, 10.0):
            d[f"열림_dc{int(_t)}"] = float((_dc.values >= _t).mean())
        d["안붙음_심도"] = float(np.median(_o[~g.짝지음.values])) \
            if (~g.짝지음.values).sum() > 5 else np.nan
        # ★ **붙어 있는데 이미 비틀린** 집단 — 도메인교환 가설이 직접 겨누는 곳.
        #   완전히 떨어지기 전 단계라 단량체로 남아 있으면서 계면을 내보인다.
        if len(c) > 5:
            _oa = c["OCD5"].values.astype(float)
            for _t in (10, 15, 20):
                d[f"붙음중_꼬리{_t}"] = float((_oa > _t).mean())
        # ★ "기준점 앙상블과의 거리" 는 못 잰다 — BioEmu 는 단일 사슬만 되므로
        #   링커 없는 Fv 의 앙상블을 만들 수 없다 (README: "only supports monomers").
        #   대신 **같은 항체의 다른 링커들을 합친 분포**와의 거리를 잰다.
        #   "이 링커가 형제들과 얼마나 다른가" — 항체 안에서 정의되므로 비교가 성립한다.
        sib = Gf[(Gf.항체 == ab) & (Gf.링커 != lk) & Gf.짝지음]
        d["W1_형제"] = w1(src.OCD5, sib.OCD5) if len(sib) >= 20 else np.nan
        for k in AB6:
            d[f"Δ{k}중앙"] = float(np.median(src[f"Δ{k}"]))
            d[f"Δ{k}_MAD"] = mad(src[f"Δ{k}"])
        # ── 링커 기하 — geom 이 없어도 **열은 반드시 만든다** (NaN 으로).
        #    열이 통째로 없으면 아래 특징 선택이 조용히 다른 집합을 쓰게 된다.
        for k in GEOM_FEATS:
            d[k] = (float(np.median(g[k].dropna()))
                    if k in g.columns and g[k].notna().any() else np.nan)
        rows.append(d)
    F = pd.DataFrame(rows)
    return F.sort_values([c for c in ("합성", "블록", "항체", "길이")
                          if c in F.columns]).reset_index(drop=True)

# ★ 5절이 안 끝났으면 **여기서 이유를 말하고 멈춘다.** 그냥 두면
#   build_features 안에서 'NoneType 에 생성기 속성이 없다' 같은 말이 나오는데,
#   그건 원인(5절이 비었다)과 아무 상관 없어 보이는 문장이라 노트북이 고장난 것처럼 읽힌다.
assert E is not None and len(E), (
    "frames.csv 가 비었다 — 6~9절은 전부 이 표 위에 선다. 5절을 먼저 끝내라.")
FEAT = build_features(E, G)
FEAT.to_csv(f"{OUT}/features.csv", index=False, encoding="utf-8-sig")
ALLF = [PRIMARY] + EXPLORATORY
print(f"특징표 {len(FEAT)}행 (실측 {int((~FEAT.합성).sum())} · 합성 "
      f"{int(FEAT.합성.sum())}) → {OUT}/features.csv")
display(FEAT[[c for c in ("블록", "항체", "링커", "길이", "합성", "n프레임", "n붙음")
              if c in FEAT.columns] + ALLF].round(3))
if FEAT.붙음부족.any():
    print("  ★ 붙은 프레임이 20개 미만인 구성체 — 각도 요약을 전체 프레임으로 냈다:")
    display(FEAT[FEAT.붙음부족][["항체", "링커", "n프레임", "n붙음", PRIMARY]])
''')

code(r'''
# ── 6절-B · 두 원장 — 특징을 믿기 전에 먼저 본다 ───────────────────────────
import scipy.stats as st_

def confound_ledger(F, feats, length_col="길이", block="항체"):
    """각 특징이 링커 **길이**와 얼마나 같은 것인가.
    판정은 **그룹(항체) 안** 상관으로 한다 — 그룹 간 차이가 전체 상관을 부풀린다."""
    rows = []
    cw = F.groupby(block)[length_col].transform(lambda v: v - v.mean()).values
    for c in feats:
        v  = F[c].astype(float)
        vw = v.groupby(F[block]).transform(lambda s: s - s.mean()).values
        m  = np.isfinite(v.values) & np.isfinite(F[length_col].values.astype(float))
        mw = np.isfinite(vw) & np.isfinite(cw)
        rw = (np.corrcoef(vw[mw], cw[mw])[0, 1]
              if mw.sum() > 2 and np.std(vw[mw]) > 1e-12 and np.std(cw[mw]) > 1e-12
              else np.nan)
        a = abs(rw) if np.isfinite(rw) else 0.0
        # ★ 그룹 **마다** 길이와 rho = ±1 이면, 그 특징은 길이의 결정적 함수다.
        #   그러면 증분 검정이 수학적으로 0 에 묶인다 — '효과 없음' 이 아니라
        #   **'질문에 답할 수 없음'** 이다. 둘을 구분해서 보고해야 한다.
        per = [st_.spearmanr(g[c], g[length_col])[0]
               for _, g in F.groupby(block) if g[c].nunique() > 1 and len(g) > 2]
        per = [x for x in per if np.isfinite(x)]
        det = bool(per) and all(abs(x) > 0.999 for x in per)
        rows.append(dict(특징=c,
            r_길이=round(float(np.corrcoef(v[m], F[length_col].values[m])[0, 1]), 2)
                   if m.sum() > 2 else np.nan,
            그룹내r_길이=round(float(rw), 2) if np.isfinite(rw) else np.nan,
            그룹내rho최소=round(float(np.min(np.abs(per))), 2) if per else np.nan,
            길이완전공선=det,
            판정=("★★ 길이의 결정적 함수 — 증분 검정이 원리적으로 무력하다" if det else
                  "★ 길이의 변장" if a > 0.9 else
                  "길이와 강하게 얽힘" if a > 0.7 else "독립적")))
    return pd.DataFrame(rows)

def bootstrap_feature_se(frames, F, feats, n_boot=300, seed=0, gen=GEN):
    """프레임 재표집으로 구성체별 특징의 표준오차.
    SE 가 구성체 간 흩어짐(SD)에 견줄 만큼 크면 그 특징으로는 아무것도 못 본다.

    ★ 안쪽 고리에 pandas 를 두지 않는다. 구성체 44 × 부트 300 × 특징 3 = 4만 번이라
      `g.iloc[...]` 한 번이 1 ms 만 돼도 40초가 그냥 날아간다. numpy 로 한 번에 뽑는다."""
    rng = np.random.default_rng(seed)
    Gf = frames[frames.생성기 == gen].copy()
    if "OCD5" not in Gf.columns:
        Gf["OCD5"] = sum(Gf[f"Δ{k}"].abs()/AB_SD[k] for k in AB6 if k != "dc")
    Gf["짝지음"] = paired(Gf)
    # 통계량을 (붙음 불리언, OCD6) 두 배열만 받는 형태로 쓴다
    stat = {
        PRIMARY:    lambda pr, oc: 1.0 - pr.mean(),
        "OCD5중앙": lambda pr, oc: np.median(oc[pr]) if pr.any() else np.nan,
        "OCD5_MAD": lambda pr, oc: (1.4826*np.median(np.abs(oc[pr]-np.median(oc[pr])))
                                    if pr.sum() > 5 else np.nan),
    }
    out = []
    for c in feats:
        if c not in stat: continue
        f, ses = stat[c], []
        for (ab, lk), g in Gf.groupby(["항체", "링커"], sort=False):
            pr = g.짝지음.values.astype(bool); oc = g.OCD5.values.astype(float)
            n = len(pr)
            if n < 5: continue
            idx = rng.integers(0, n, (n_boot, n))          # (부트, 표본) 한 번에
            b = np.array([f(pr[k], oc[k]) for k in idx], dtype=float)
            ses.append(np.nanstd(b, ddof=1))
        if not ses: continue
        se = float(np.nanmedian(ses))
        sd = float(F[c].std(ddof=1))
        rel = max(1 - (se**2)/(sd**2), 0.0) if sd > 1e-12 else np.nan
        out.append(dict(특징=c, SE중앙=round(se, 4), 구성체간SD=round(sd, 4),
                        SE대SD=round(se/sd, 2) if sd > 1e-12 else np.nan,
                        신뢰도=round(rel, 2),
                        감쇠배율=round(float(np.sqrt(rel)), 2) if rel == rel else np.nan,
                        판정=("★ 잡음이 신호만큼 크다 — 프레임을 더 뽑아라"
                              if rel < 0.7 else "쓸 만하다")))
    return pd.DataFrame(out)

# ── ML 의 **독립 단위** 를 정한다 ──────────────────────────────────────────
# ★ 블록이 아니라 **항체**다. 한 블록에 항체가 둘 이상 들어갈 수 있고 (같은 Fv 의
#   LH/HL 두 배향, 또는 같은 실험 세트의 서로 다른 도메인쌍), 항체마다
#   ABodyBuilder2 기준점이 **따로** 잡힌다. 즉 Δ 의 영점이 다르다.
#   블록으로 묶으면 '링커 간 차이' 항이 '항체 간 차이' 를 빨아들여 엉뚱한 것을 잰다.
#   항체 키에 이미 블록이 들어 있으므로(실측n_B블록_배향) 항체는 블록보다 항상 잘다.
GROUP = "항체"
# 씨앗 반복은 잡음 바닥 측정용이지 별개 구성체가 아니다 — ML 에서 뺀다
_isrep = FEAT.링커.astype(str).str.endswith("__rep2")
REAL = FEAT[(~FEAT.합성) & (~_isrep)].reset_index(drop=True)
_bs  = REAL.groupby(GROUP).링커.nunique()
USE  = REAL[REAL[GROUP].isin(_bs[_bs >= 3].index)].reset_index(drop=True)
# ★ 한 구성체에 실측값이 둘 이상이면 아래 groupby 가 조용히 하나만 집는다
if HC and len(USE):
    _chk = FEAT[~FEAT.합성].groupby(["항체", "링커"])[HC].nunique(dropna=True)
    assert (_chk <= 1).all(), f"한 구성체에 실측값이 여러 개다:\n{_chk[_chk > 1]}"

print("="*76); print("★ 교란 원장 — 이 특징이 그냥 길이인가"); print("="*76)
print("  항체 안 |r| > 0.9 면 그 특징은 길이의 다른 이름이다. n≈20 에서 둘을 못 가른다.")
print("  그리고 항체 **마다** rho = ±1 이면 길이의 결정적 함수다 — 7절 ③ 증분 검정이")
print("  수학적으로 0 에 묶인다. 그때 'p=0.87, 효과 없음' 이라 쓰면 안 된다.")
print("  올바른 문장은 **'이 설계로는 그 질문에 답할 수 없다'** 이다.\n")
LED = confound_ledger(USE, ALLF, block=GROUP)
display(LED)
DEGENERATE = set(LED[LED.길이완전공선].특징)
if DEGENERATE:
    print(f"  ★★ 길이의 결정적 함수: {sorted(DEGENERATE)}")
    print("     이 특징들만으로는 '길이 위의 증분' 을 물을 수 없다. 4-B절 합성 패널이")
    print("     같은 길이에서 조성만 바꾼 칸을 주는 유일한 탈출구다.")

# ── ★ 길이축의 정체 — '길이' 인가 그냥 '링커 이름' 인가 ────────────────────
#   실측 패널에서 링커 4종이 길이도 4종이면, 항체 안에서 길이가 링커를 **유일하게
#   식별한다.** 그러면 '길이 효과' 와 '이 링커가 좋다' 는 **같은 문장**이고,
#   rho(길이, y) 를 '길이 효과' 라고 부르는 것은 데이터가 아니라 **작명**이다.
#   위의 교란 원장은 '특징이 길이의 변장인가' 를 묻는다. 이건 다른 질문이다 —
#   **길이 자체가 링커 이름의 변장인가.**
print()
_bij = []
for _ab, _g in USE.groupby(GROUP):
    _k, _L = _g.링커.nunique(), _g.길이.nunique()
    _bij.append(dict(항체=_ab, 링커수=_k, 길이종류=_L, 길이가링커를식별=(_k == _L and _k > 1)))
BIJ = pd.DataFrame(_bij)
LENGTH_IS_NAME = bool(len(BIJ) and BIJ.길이가링커를식별.all())
display(BIJ)
if LENGTH_IS_NAME:
    print("  ★★ 모든 항체에서 **길이가 링커를 유일하게 식별한다.**")
    print("     → '길이 효과' 와 '이 링커가 좋다' 는 구분할 수 없는 같은 문장이다.")
    print("       rho(길이, 순도) 가 아무리 깨끗해도 **길이 때문이라고 말하면 안 된다.**")
    print("       같은 길이에 조성이 다른 칸이 하나도 없기 때문이다 (4-B절이 유일한 출구).")
    print("     → 7절 ③ 증분 검정의 귀무모형 '길이' 도 사실 '링커 정체성' 이다.")
    print("       거기서 증분이 0 이면 '앙상블이 링커 이름 위에 더한 게 없다' 는 뜻이지")
    print("       '길이 말고는 없다' 가 아니다.")
else:
    print("  길이가 링커를 식별하지 못하는 항체가 있다 — 같은 길이에 다른 조성이 있다.")
    print("  그 항체에서만은 길이와 조성을 (약하게나마) 가를 수 있다.")

# ── 같은 이름의 링커가 블록마다 길이가 다른가 ──────────────────────────────
#   이름이 같으면 서열도 같아야 한다. 다르면 '같은 링커' 라는 전제가 깨지고,
#   블록 간 비교와 합성 패널의 숙주 선택이 전부 흔들린다.
_real = FEAT[~FEAT.합성] if "합성" in FEAT.columns else FEAT
_nl = _real.groupby("링커").길이.nunique()
_badnm = _nl[_nl > 1]
if len(_badnm):
    print(f"\n  ★★ 이름은 같은데 길이가 다른 링커: {list(_badnm.index)}")
    print("     같은 이름이면 같은 서열이어야 한다. 엑셀의 Linker 열을 확인하라 —")
    print("     서열이 정말 다르면 **다른 링커로 이름을 갈라야** 한다. 안 그러면")
    print("     블록 간 '같은 링커' 대조가 성립하지 않는다.")
    display(_real[_real.링커.isin(_badnm.index)]
            .groupby(["링커", "길이"])[GROUP].apply(lambda v: sorted(set(v))).reset_index())

# ── ★ 분산 분해 — 이 특징의 변동이 '항체 사이' 인가 '링커 사이' 인가 ────────
#   링커 신호는 **항체 안** 변동에만 있다. 타깃을 항체 안에서 중심화하므로
#   항체 간 성분은 통째로 지워진다. 그러면 항체내몫이 작은 특징은
#   **라벨을 보기도 전에** 링커 신호를 나를 수 없다는 것이 정해져 있다.
#   ※ 이건 y 를 안 쓰는 사전점검이라 몇 번을 봐도 다중비교가 안 생긴다.
#   ※ 실측에서 실제로 나온 값: 열림분율 0.61 · OCD5_p95 0.76 인데
#     붙음중_중앙 0.23 · 안붙음_심도 0.20 · 말단간_중앙 0.29 였다.
#     뒤 셋은 '링커를 재는' 게 아니라 **어느 항체인지를 재고** 있었다.
print("="*76); print("★ 분산 분해 — 변동이 항체 사이인가 링커 사이인가 (y 를 안 본다)"); print("="*76)
print("  항체내몫 = 항체내분산 / (항체간분산 + 항체내분산).")
print("  0.3 미만이면 그 특징은 **링커가 아니라 항체 정체성**을 재고 있다 —")
print("  항체 고정효과가 나머지를 지우므로 검정력이 아니라 **정보가** 없다.\n")
_vd = []
for _c in [x for x in ALLF + TAIL_FEATS if x in USE.columns]:
    _v = pd.to_numeric(USE[_c], errors="coerce")
    if _v.notna().sum() < 4 or _v.std(ddof=1) < 1e-12: continue
    _m = _v.groupby(USE[GROUP]).transform("mean")
    _sb = float(_v.groupby(USE[GROUP]).mean().std(ddof=1))
    _sw = float((_v - _m).std(ddof=1))
    _sh = _sw**2/(_sb**2 + _sw**2) if (_sb**2 + _sw**2) > 1e-30 else np.nan
    _vd.append(dict(특징=_c, 항체간SD=round(_sb, 3), 항체내SD=round(_sw, 3),
                    항체내몫=round(_sh, 2),
                    판정=("★ 항체를 재고 있다" if _sh < 0.3 else
                          "치우쳐 있다" if _sh < 0.5 else "쓸 만하다")))
VDEC = pd.DataFrame(_vd).sort_values("항체내몫", ascending=False)
display(VDEC)
_dead = list(VDEC[VDEC.항체내몫 < 0.3].특징)
if _dead:
    print(f"  ★★ 항체내몫 < 0.3: {_dead}")
    print("     이 특징들은 이 설계에서 링커 신호를 **원리적으로** 못 나른다.")
    print("     p 가 크게 나와도 '효과 없음' 이 아니라 '정보가 없음' 이다.")

# ── ★ 수율 축 — 상관이 아니라 **파국 탐지**다 ──────────────────────────────
#   실측에서 나온 구조가 이렇다 (그래서 회귀로 물으면 안 된다):
#     · 항체 사이 수율 중앙값은 60배 차이 난다 (4 ~ 240 mg/L)
#     · 항체 **안** 링커 간 배수는 1.1 · 1.3 · 1.4 · 1.6 배 — 거의 안 변한다
#     · 그런데 딱 하나, 블록3 Whitlow218 이 **49배** 무너졌다 (222 → 4.7 mg/L)
#   즉 링커가 수율에 주는 효과는 **연속적인 축이 아니라 사실상 이진**이다:
#   멀쩡하거나(±60%), 아예 안 나오거나. 상관계수는 이런 구조를 못 본다 —
#   순위로 보면 파국 1건이 그냥 '제일 낮은 값' 한 칸일 뿐이다.
#   그래서 회귀 대신 **형제 대비 배수**로 보고, 임계 아래를 이름으로 지목한다.
#   ※ 파국은 링커 고유 성질이 아니다. 같은 Whitlow218 이 다른 세 항체에서는
#     형제 중앙값의 0.75 · 1.07 · 1.13 배로 멀쩡했다. **그 도메인과만** 안 맞는다.
#     이것이 '링커 × 도메인 상호작용' 의 가장 날것 그대로의 증거다.
_yc = [c for c in OBS if ("수율" in str(c) or "생산" in str(c)
                          or "titer" in str(c).lower() or "yield" in str(c).lower())]
if _yc and _yc[0] in FEAT.columns:
    YC = _yc[0]
    print("="*76); print(f"★ 수율 축 — 파국 탐지 ({YC})"); print("="*76)
    print("  링커가 수율에 주는 효과는 연속적이지 않고 **이진**일 수 있다 —")
    print("  멀쩡하거나, 아예 안 나오거나. 그래서 상관이 아니라 형제 대비 배수로 본다.\n")
    _Y = FEAT[(~FEAT.합성) & FEAT[YC].notna()][[GROUP, "링커", "길이", YC]].copy()
    if len(_Y):
        _med = _Y.groupby(GROUP)[YC].transform("median")
        _Y["형제중앙"] = _med.round(1)
        _Y["형제대비"] = (_Y[YC]/_med).round(3)
        CATASTROPHE_CUT = 0.25          # 형제 중앙값의 1/4 미만이면 파국
        _Y["파국"] = _Y.형제대비 < CATASTROPHE_CUT
        _sp = _Y.groupby(GROUP)[YC].agg(["min", "max", "median"])
        _sp["안쪽배수"] = (_sp["max"]/_sp["min"]).round(1)
        print("  항체별 수율 규모와 항체 **안** 링커 간 배수")
        display(_sp.round(1))
        print(f"  항체 간 중앙값 배수 = "
              f"{_sp['median'].max()/max(_sp['median'].min(), 1e-9):.0f}배 "
              f"· 항체 안 최대 배수 = {_sp['안쪽배수'].max():.1f}배")
        CATA = _Y[_Y.파국]
        if len(CATA):
            print(f"\n  ★★ 파국 {len(CATA)}건 — 형제 중앙값의 {CATASTROPHE_CUT:.0%} 미만")
            display(CATA[[GROUP, "링커", "길이", YC, "형제중앙", "형제대비"]])
            for _lk in CATA.링커.unique():
                _o = _Y[(_Y.링커 == _lk) & (~_Y.파국)]
                print(f"     '{_lk}' 는 다른 항체 {len(_o)}개에서는 형제 대비 "
                      f"{sorted(_o.형제대비.round(2))} 로 멀쩡하다.")
                print(f"     → 링커 고유 성질이 아니라 **그 도메인과의 조합**이 문제다.")
            print("     ※ 파국은 앙상블 특징으로 예측되지 않았다 (실측 1건에서 전 특징 |z|<1.1).")
            print("       실패 지점이 **접힌 상태의 형태 분포 밖**이라는 뜻이다 —")
            print("       공동번역 폴딩·ER 품질관리·분비 쪽을 봐야 한다. BioEmu 는 거길 안 본다.")
            print("     ※ 먼저 확인할 것: 이 구성체를 **다시 만들어 봤는가?** 50배짜리")
            print("       한 점은 생물학일 수도 있고 그냥 실패한 트랜스펙션일 수도 있다.")
        else:
            print(f"\n  파국 없음 — 모든 구성체가 형제 중앙값의 {CATASTROPHE_CUT:.0%} 이상이다.")
        _thin = _sp[_sp["median"] < 0.2*_sp["median"].max()]
        if len(_thin):
            print(f"\n  ★ 수율이 전체 최고의 20% 미만인 항체: {list(_thin.index)}")
            print("     이 항체들의 물성값은 **극소량에서 잰 값**이다. 다른 항체와")
            print("     나란히 놓고 읽을 때 그 사실을 같이 말해야 한다.")
        YIELD_RESULT = dict(표=_Y, 파국=CATA, 규모=_sp, 임계=CATASTROPHE_CUT)

print("="*76); print("★ 특징 잡음 원장 — 특징 하나가 150 프레임의 통계량이다"); print("="*76)
print("  관측 상관 ≈ 참 상관 × 감쇠배율.  '상관이 없다' 와 '특징이 시끄럽다' 를 여기서 가른다.")
NOISE = bootstrap_feature_se(E[~E.합성.astype(bool)], USE, ALLF)
display(NOISE)
RELY = dict(zip(NOISE.특징, NOISE.신뢰도)) if len(NOISE) else {}
_rp0 = RELY.get(PRIMARY, np.nan)
if np.isfinite(_rp0) and _rp0 < 0.7:
    print(f"  ★★ 확증 특징 '{PRIMARY}' 의 신뢰도가 {_rp0:.2f} 다 — 프레임 {N_BE if 'N_BE' in dir() else 150}개로는")
    print(f"     이 양이 제대로 안 잡힌다. 관측 상관이 참값의 {np.sqrt(_rp0):.2f}배로 눌리므로")
    print(f"     7절 ① 이 귀무로 나와도 그것은 **'효과가 없다' 가 아니라 '측정이 시끄럽다'** 다.")
    print(f"     고치는 법: N_BE 를 올려라 (오차는 √n 로 준다 → 4배 뽑으면 절반).")
# ── 잡음 바닥 — 같은 서열을 두 번 돌렸을 때의 차이 ────────────────────────
_rp = [c for c in FEAT.링커 if str(c).endswith("__rep2")]
if _rp:
    print("="*76); print("★ 잡음 바닥 — 같은 서열을 두 번 돌리면 얼마나 다른가"); print("="*76)
    print("  링커를 안 바꿨으니 여기 나오는 차이는 전부 **BioEmu 실행 잡음**이다.")
    print("  링커 간 차이가 이보다 작으면 그건 링커 효과가 아니다.\n")
    rows = []
    for lk2 in _rp:
        lk1 = lk2[:-len("__rep2")]
        a = FEAT[FEAT.링커 == lk1]; b = FEAT[FEAT.링커 == lk2]
        if not (len(a) and len(b)): continue
        a, b = a.iloc[0], b.iloc[0]
        for c in ALLF:
            if c not in FEAT.columns or not np.isfinite(a.get(c, np.nan)): continue
            sd = FEAT[~FEAT.합성][c].std(ddof=1)
            rows.append(dict(항체=a.항체, 특징=c, 실행1=round(float(a[c]), 3),
                             실행2=round(float(b[c]), 3),
                             실행간차=round(float(abs(a[c]-b[c])), 3),
                             구성체간SD=round(float(sd), 3),
                             비=round(float(abs(a[c]-b[c])/sd), 2) if sd > 1e-12 else np.nan))
    if rows:
        NZ = pd.DataFrame(rows); display(NZ)
        _bad = NZ[NZ.비 > 0.5]
        if len(_bad):
            print(f"  ★ 실행 간 차이가 구성체 간 흩어짐의 절반을 넘는 특징: "
                  f"{sorted(set(_bad.특징))}")
            print("     이 특징으로 링커를 구분한다고 말할 수 없다. 프레임을 더 뽑아라.")
        else:
            print("  → 실행 잡음이 구성체 간 흩어짐보다 충분히 작다. 링커 차이를 읽어도 된다.")
else:
    # ★★ frames.csv 에 씨앗 반복이 없어도 **geom.csv 에는 있을 수 있다.**
    #   실측에서 정확히 그랬다 — BioEmu 가 rep2 프레임을 만들었는데 기준점 ABangle 이
    #   실패해서 frames 에 한 줄도 안 들어갔다 ("쌍 PDB 0개 → 성공 0/0").
    #   그런데 도메인Rg·말단간·링커 기하는 **좌표에서 바로** 나온다 — ABangle 도,
    #   ANARCI 도, 번호매김도 안 거친다. 그러니 잡음 바닥은 여기서 낼 수 있다.
    #   세션 내내 없다고 하던 분모가 사실 디스크에 있었다.
    _rep_g = pd.DataFrame()
    if len(G):
        _gg = G.copy()
        _gg["기준링커"] = _gg.링커.astype(str).str.replace(r"__rep\d+$", "", regex=True)
        _gg["실행"] = np.where(_gg.링커.astype(str).str.contains("__rep"), "rep", "orig")
        _n = _gg.groupby(["항체", "기준링커"]).실행.nunique()
        _pairs = _n[_n >= 2].index
        if len(_pairs):
            _rep_g = _gg[_gg.set_index(["항체", "기준링커"]).index.isin(_pairs)]
    if len(_rep_g):
        print("※ frames.csv 엔 씨앗 반복이 없지만 **geom.csv 에는 있다** —")
        print("  좌표 기하는 ABangle 을 안 거치므로 여기서 잡음 바닥을 낸다.")
        GF = [c for c in ("도메인Rg", "말단간", "링커접촉_잔기당", "링커밀착율",
                          "링커신장도", "링커Rg_잔기당") if c in _rep_g.columns]
        _rows = []
        for c in GF:
            _m = (_rep_g.groupby(["항체", "기준링커", "실행"])[c]
                  .median().unstack("실행").dropna())
            if len(_m) < 1 or not {"orig", "rep"} <= set(_m.columns): continue
            _d = (_m["rep"] - _m["orig"]).values
            _run = float(np.sqrt(np.mean(_d**2)/2))       # 실행 간 SD
            _bet = float(USE[c].std(ddof=1)) if c in USE.columns else np.nan
            _rel = max(1 - (_run/_bet)**2, 0.0) if _bet and _bet > 1e-12 else np.nan
            _rows.append(dict(특징=c, 실행간SD=round(_run, 3),
                              구성체간SD=round(_bet, 3) if _bet == _bet else np.nan,
                              신뢰도=round(_rel, 2) if _rel == _rel else np.nan,
                              감쇠배율=round(np.sqrt(_rel), 2) if _rel == _rel else np.nan,
                              n쌍=len(_m),
                              판정=("★ 실행 잡음이 링커 차이만큼 크다" if _rel == _rel
                                    and _rel < 0.5 else "쓸 만하다")))
        if _rows:
            SEEDN = pd.DataFrame(_rows); display(SEEDN)
            print(f"  씨앗 반복 쌍 {len(_m)}개로 냈다. **실행 잡음이 분모다** —")
            print("  구성체 간 흩어짐이 이것보다 작으면 우리가 재는 것은 링커가 아니라")
            print("  BioEmu 를 두 번 돌린 차이다. 신뢰도 0.5 미만이면 그 특징은 못 쓴다.")
            _bad = [r["특징"] for r in _rows if r["신뢰도"] == r["신뢰도"] and r["신뢰도"] < 0.5]
            if _bad:
                print(f"  ★★ 못 쓰는 특징: {_bad}")
        else:
            print("  ※ 짝이 맞는 씨앗 반복을 못 찾았다.")
    else:
        print("※ 씨앗 반복이 없다 — **BioEmu 의 잡음 바닥이 측정되지 않았다.**")
        print("  4절의 N_SEED_REP 을 켜고 한 번 더 돌려라. 그것 없이는 링커 간 차이가")
        print("  링커 때문인지 실행마다 달라서인지 구분할 방법이 없다 (구성체당 20분이면 된다).")
        print("  ※ 이미 돌렸는데 여기까지 안 왔다면 5절 로그에서 '기준점 ABangle 실패' 를")
        print("    찾아보라 — 프레임은 만들어졌는데 ABangle 이 못 재서 버려졌을 수 있다.")

print(f"\n  ML 이 쓰는 것: 구성체 {len(USE)} · **독립 단위(항체) {USE[GROUP].nunique()}개** "
      f"· 블록 {sorted(USE.블록.unique())}")
print(f"  자유도는 26 이 아니라 {USE[GROUP].nunique()} 개분이다. 합성 데이터로도 안 늘어난다.")
display(USE.groupby([GROUP, "블록"]).agg(링커수=("링커", "nunique"),
                                        링커=("링커", lambda x: list(x))).reset_index())
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 6절 C·D·E · **채널 판정** — BioEmu 가 조성을 보는가

세 셀이 같은 질문을 서로 다른 자리에서 묻는다. **순서가 있다.**

| 셀 | 어디서 읽나 | 무엇을 답하나 |
|---|---|---|
| **6-D** | 링커 **안** (나선도) | 조성이 모델에 **닿기는 하는가**. 논문이 서열 민감도를 실제로 보인 자리 |
| **6-C** | 도메인 **배향** (ΔABangle) | 그 조성이 **전역 배향까지 전달되는가** |
| **6-E** | 프레임 상관 (ICC) | 위 두 p 가 서 있는 땅이 단단한가 |

★ 6-D 가 음성이면 6-C 는 볼 것도 없다. 반대로 **6-D 양성 · 6-C 음성**은 실패가 아니라
결과다 — "조성은 링커 국소 구조를 바꾸지만 도메인 배향까지는 안 간다" 는 뜻이고,
그건 이 연구가 낼 수 있는 가장 구체적인 기전 진술이다.

---

### 6절-C · 전역 배향에서의 채널 판정
합성 요인패널의 28칸에 2원 분산분석을 건다. 이 표 한 장이 연구의 분기점이다.

| 나오는 모양 | 뜻 | 그 다음 |
|---|---|---|
| η²(길이) 만 크다 | BioEmu 는 사실상 `\|i−j\|` 만 본다 | 7절의 상관은 **전부 "긴 링커가 나쁘다"** 다. 그렇게 보고한다 |
| η²(조성) 도 크다 | **조성 채널이 있다** | 어느 특징이 그 채널을 나르는지 본다. 7절이 의미를 갖는다 |
| 둘 다 작다 | 앙상블이 링커에 아예 안 반응한다 | 접는다. BioEmu 로는 이 질문을 못 본다 |

### 반복 없는 요인설계의 함정 — 피해 간다
칸당 구성체가 하나뿐이면 **잔차 자유도가 0** 이다. 상호작용과 오차를 가를 수 없고
잔차 SS 가 기계적으로 0 이 나온다. 그 표의 F 와 p 는 숫자가 아니라 착시다.

→ 각 구성체의 150 프레임을 **무작위 3등분**해 칸 안 반복을 만든다. 그러면 칸 안
흩어짐이 **BioEmu 표집 잡음** 그 자체가 되고, 그게 "이 칸과 저 칸의 차이가
표집 잡음보다 큰가" 를 물을 때 쓸 올바른 오차항이다.

★ 생물학적 반복이 아니다. 이 오차항으로 말할 수 있는 것은 **"BioEmu 의 앙상블이
조성에 따라 다르다"** 까지이고, "실제 단백질이 다르다" 는 아니다.

### 그리고 음성대조 — 이 데이터로 얻을 수 있는 가장 강한 것
씨앗 반복 두 개(같은 서열, 다른 실행)를 **"서로 다른 링커 2종" 인 척** 같은 검정에
넣는다. 링커를 안 바꿨으니 여기서 '유의' 가 나오면 **검정의 위양성률이 깨진 것**이고,
그러면 위의 조성 판정도 전부 무효다. 셀이 자동으로 돌린다.
''')

code(r'''
# ── 6절-C · 2원 분산분석 — 길이축 vs 조성축 ────────────────────────────────
def split_replicates(frames, F, feats, k=3, seed=0, gen=GEN):
    """구성체마다 프레임을 k 덩이로 쪼개 **칸 안의 반복**을 만든다.
    칸 안 흩어짐 = BioEmu 표집 잡음 = 올바른 오차항."""
    rng = np.random.default_rng(seed)
    Gf = frames[frames.생성기 == gen].copy()
    if "OCD5" not in Gf.columns:
        Gf["OCD5"] = sum(Gf[f"Δ{k}"].abs()/AB_SD[k] for k in AB6 if k != "dc")
    Gf["짝지음"] = paired(Gf)
    geo = [c for c in feats if c in GEOM_FEATS]
    if geo and len(G):
        Gf = Gf.merge(G.drop(columns=["링커"], errors="ignore"),
                      on=["항체", "태그"], how="left", validate="many_to_one")
    rows = []
    for (ab, lk), g in Gf.groupby(["항체", "링커"], sort=False):
        if len(g) < 3*k: continue
        for j, part in enumerate(np.array_split(rng.permutation(len(g)), k)):
            s = g.iloc[part]; c = s[s.짝지음]
            src = c if len(c) >= 7 else s
            d = dict(항체=ab, 링커=lk, 반복=j, n=len(s))
            d[PRIMARY]   = float(1 - s.짝지음.mean())
            d["OCD5중앙"] = float(np.median(src.OCD5))
            d["OCD5_MAD"] = mad(src.OCD5)
            for c2 in geo:
                d[c2] = float(np.median(s[c2].dropna())) if s[c2].notna().any() else np.nan
            rows.append(d)
    return pd.DataFrame(rows)

def two_way_anova(df, value, a="조성", b="설계길이"):
    """η² + F 검정. 칸당 반복이 없으면 **F 를 내지 않는다** (잔차 자유도 0)."""
    d = df[[value, a, b]].dropna()
    if len(d) < 6: return pd.DataFrame(), "관측이 모자란다"
    y = d[value].values.astype(float); gm = y.mean()
    sst = float(((y-gm)**2).sum())
    na, nb, N = d[a].nunique(), d[b].nunique(), len(d)
    cell = d.groupby([a, b])[value].agg(["mean", "size"])
    ss = {k: float(sum(len(g)*(g[value].mean()-gm)**2 for _, g in d.groupby(k)))
          for k in (a, b)}
    ss_cells = float((cell["size"]*(cell["mean"]-gm)**2).sum())
    inter = max(ss_cells - ss[a] - ss[b], 0.0); resid = max(sst - ss_cells, 0.0)
    dfr = N - na*nb
    e2 = lambda v: round(v/sst, 3) if sst > 1e-12 else np.nan
    if int(cell["size"].min()) < 2 or dfr <= 0:
        T = pd.DataFrame([dict(요인=a, SS=round(ss[a], 4), eta2=e2(ss[a]), F=np.nan, p=np.nan),
                          dict(요인=b, SS=round(ss[b], 4), eta2=e2(ss[b]), F=np.nan, p=np.nan),
                          dict(요인="상호작용+오차", SS=round(inter+resid, 4),
                               eta2=e2(inter+resid), F=np.nan, p=np.nan)])
        return T, "★ 칸당 관측이 1개 — 상호작용과 오차를 못 가른다. F·p 를 내지 않는다."
    dfk = {a: na-1, b: nb-1, "상호작용": (na-1)*(nb-1)}
    ss["상호작용"] = inter; mse = resid/dfr
    out = []
    for k in (a, b, "상호작용"):
        F_ = (ss[k]/dfk[k])/mse if mse > 1e-12 and dfk[k] else np.nan
        out.append(dict(요인=k, SS=round(ss[k], 4), eta2=e2(ss[k]), 자유도=dfk[k],
                        F=round(F_, 2) if np.isfinite(F_) else np.nan,
                        p=round(float(st_.f.sf(F_, dfk[k], dfr)), 5)
                          if np.isfinite(F_) else np.nan))
    out.append(dict(요인="잔차", SS=round(resid, 4), eta2=e2(resid), 자유도=dfr,
                    F=np.nan, p=np.nan))
    return pd.DataFrame(out), f"칸당 반복 {int(cell['size'].min())}개 · 잔차 자유도 {dfr}"

SYN = FEAT[FEAT.합성].copy()
if len(SYN) >= 8 and SYN.조성.nunique() >= 2 and SYN.설계길이.nunique() >= 2:
    REP = split_replicates(E[E.합성.astype(bool)], FEAT, ALLF, k=3)
    REP = REP.merge(SYN[["링커", "조성", "설계길이"]], on="링커", how="left").dropna(subset=["조성"])
    print("="*76)
    print("★ 채널 판정 — 합성 요인패널 2원 분산분석 (칸 안 반복 3)")
    print("="*76)
    print(f"  칸 {SYN.조성.nunique()}조성 × {SYN.설계길이.nunique()}길이 = "
          f"{SYN.조성.nunique()*SYN.설계길이.nunique()} · 반복 포함 {len(REP)}행\n")
    VERD = {}
    for c in ALLF:
        if c not in REP.columns or REP[c].isna().all(): continue
        T, note = two_way_anova(REP, c)
        if not len(T): continue
        t = T.set_index("요인")
        eC = float(t.loc["조성", "eta2"]); eL = float(t.loc["설계길이", "eta2"])
        pC = t.loc["조성", "p"] if "p" in t.columns else np.nan
        VERD[c] = dict(특징=c, 조성eta2=eC, 길이eta2=eL,
                       조성p=pC, 비=round(eC/max(eL, 1e-9), 2))
        print(f"  [{c}]  {note}")
        display(T)
    # ── ★ 음성대조 — 씨앗 반복 두 개를 '서로 다른 링커 2종' 인 척 넣어 본다 ───
    #   링커를 안 바꿨으니 여기서 '유의' 가 나오면 **검정의 위양성률이 깨진 것**이고,
    #   그러면 위의 조성 판정도 전부 무효다. 이 데이터로 얻을 수 있는 가장 강한 음성대조다.
    _rp2 = [c for c in FEAT.링커 if str(c).endswith("__rep2")]
    if _rp2:
        print("\n" + "─"*76)
        print("음성대조 — 같은 서열의 두 실행을 '다른 링커' 인 척 검정에 넣는다")
        print("─"*76)
        nz = []
        for lk2 in _rp2:
            lk1 = lk2[:-len("__rep2")]
            ab = FEAT[FEAT.링커 == lk2].항체.iloc[0]
            g = E[(E.항체 == ab) & (E.링커.isin([lk1, lk2])) & (E.생성기 == GEN)].copy()
            if g.링커.nunique() < 2: continue
            if "OCD5" not in g.columns:
                g["OCD5"] = sum(g[f"Δ{k}"].abs()/AB_SD[k] for k in AB6 if k != "dc")
            for c in ("ΔHL", "Δdc", "OCD5"):
                if c not in g.columns: continue
                a_ = g[g.링커 == lk1][c].values; b_ = g[g.링커 == lk2][c].values
                sd = np.sqrt((a_.var(ddof=1)+b_.var(ddof=1))/2)
                nz.append(dict(항체=ab, 지표=c,
                               효과크기=round(float(abs(np.median(a_)-np.median(b_))/sd), 2)
                                         if sd > 1e-9 else np.nan,
                               p=round(float(st_.mannwhitneyu(a_, b_).pvalue), 4)))
        if nz:
            NZ2 = pd.DataFrame(nz); display(NZ2)
            _fp = int((NZ2.효과크기 > 0.3).sum())
            if _fp:
                print(f"  ★★ 같은 서열의 두 실행이 효과크기 0.3 을 넘는 경우가 {_fp}건 있다.")
                print("      링커를 안 바꿨는데 '차이' 가 나왔다 = 실행 잡음이 효과크기 눈금보다 크다.")
                print("      **위의 조성 판정을 믿지 마라.** N_BE 를 크게 올린 뒤 다시 재라.")
            else:
                print("  → 같은 서열의 두 실행은 구분되지 않는다. 효과크기 눈금이 유효하다.")
    else:
        print("\n  ※ 씨앗 반복이 없어 **음성대조를 못 했다.** 4절의 N_SEED_REP 을 켜라 —")
        print("    그것 없이는 위 η² 가 링커 때문인지 실행마다 달라서인지 알 수 없다.")

    if VERD:
        V = pd.DataFrame(VERD.values())
        print("\n" + "="*76); print("★ 요약 — 조성 채널이 있는가"); print("="*76)
        display(V)
        _best = V.loc[V.조성eta2.idxmax()]
        if (_best.조성p == _best.조성p) and _best.조성p < ALPHA and _best.조성eta2 > 0.15:
            print(f"  → **조성 채널이 있다.** 가장 크게 반응한 특징: {_best.특징} "
                  f"(η²조성 {_best.조성eta2:.2f} vs η²길이 {_best.길이eta2:.2f})")
            print("     7절의 상관이 '길이의 다른 말' 이 아닐 수 있다. 읽어도 된다.")
        else:
            print(f"  → **조성 채널이 없다.** 가장 큰 것도 η²조성 {_best.조성eta2:.2f} "
                  f"(p={_best.조성p}).")
            print("     BioEmu 는 이 링커들에서 사실상 길이만 본다.")
            print("     ★ 7절에서 상관이 나오더라도 그것은 **길이 상관**이다. 그렇게 보고하라.")
            print("       (인공 링커는 MSA 가 비어 있다 — BioEmu 논문이 검증한 범위 밖이다.)")
else:
    print("합성 패널이 없다 (RUN_PANEL=False 이거나 4-B절이 아직 안 돌았다).")
    print("★ 채널 판정 없이 7절을 읽으면, 나오는 상관이 조성 때문인지 길이 때문인지")
    print("  **구조적으로 알 수 없다.** 4-B절을 먼저 돌려라 — 하룻밤이면 끝난다.")
''')

code(r'''
# ── 6절-D · ★ 국소 채널 시험 — 이 노트북에서 제일 결정적인 다섯 줄 ──────────
# BioEmu 논문(Lewis et al. 2025)이 **서열 민감도**를 보여준 결과는 전부 국소 이차구조다:
#   Fig 3A "excellent agreement of the predicted secondary structure propensities"
#   Fig 4F  단일 치환 Ile7→Pro 가 **그 잔기가 있는 나선의** 헬릭스 함량을 떨어뜨린다
#   ΔΔG 점돌연변이 50만 건에 Spearman > 0.6 — 한 잔기 차이가 모델에 닿는다는 직접 증거
# 반대로 "링커 조성이 두 도메인의 **전역 배향**을 바꾼다" 는 논문 어디에도 없다.
#
# → 그러니 조성 채널을 ΔHL/Δdc 로만 재면 **모델이 잘하는 자리에서 두 단계 떨어진 곳**에서
#   재는 것이다. 여기서는 링커 **안**에서 직접 읽는다.
#
# 시험: 길이가 같은 (EAAAK)₃ 와 (G4S)₃ 를 준다. EAAAK 는 교과서적 강직 α-나선 링커이고
#       G4S 는 교과서적 유연 코일이다. 이보다 강한 조성 대비는 만들 수 없다.
#
# ★ 사전 등록 판정 규칙 — 결과 보기 전에 박는다
#   같은 길이에서 EAAAK 가 G4S 보다 **링커 나선도를 못 올리면**, 이 구성체류에서
#   BioEmu 의 조성 채널은 죽어 있는 것이다. 그러면 조성 관련 해석을 전부 접는다 —
#   전역 배향에서 무엇이 나오든 그건 길이이거나 잡음이다.
HELIX_PAIRS = [("P_EAAAK10", "P_G4S10"), ("P_EAAAK15", "P_G4S15"),
               ("P_EAAAK20", "P_G4S20"), ("P_EAAAK25", "P_G4S25")]

def _frame_vals(col, ab, lk):
    """구성체 하나의 프레임별 값 (geom.csv 에서)."""
    q = G[(G.항체 == ab) & (G.링커 == lk)] if len(G) else pd.DataFrame()
    return q[col].dropna().values.astype(float) if col in q.columns else np.array([])

if len(FEAT[FEAT.합성]) and "링커나선도" in FEAT.columns:
    _ab = FEAT[FEAT.합성].항체.iloc[0]
    print("="*76)
    print("★ 국소 채널 시험 — 같은 길이에서 강직 나선(EAAAK) 대 유연 코일(G4S)")
    print("="*76)
    print("  논문이 서열 민감도를 실제로 보인 자리는 **국소 이차구조**다. 여기서 읽는다.")
    print("  CA 기반 나선 판별: d(i,i+4) 가 5.0~7.5 Å 이면 나선 (α-나선 ≈ 6.2, 신장 ≈ 13).\n")
    rows = []
    for hx, gs in HELIX_PAIRS:
        a = _frame_vals("링커나선도", _ab, hx); b = _frame_vals("링커나선도", _ab, gs)
        if len(a) < 20 or len(b) < 20: continue
        d4a = _frame_vals("링커i_i4", _ab, hx); d4b = _frame_vals("링커i_i4", _ab, gs)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(dict(
            길이=int("".join(c for c in gs if c.isdigit())),
            EAAAK_나선도=round(float(np.median(a)), 3),
            G4S_나선도=round(float(np.median(b)), 3),
            차=round(float(np.median(a) - np.median(b)), 3),
            효과크기=round(float((np.median(a)-np.median(b))/sd), 2) if sd > 1e-9 else np.nan,
            p=round(float(st_.mannwhitneyu(a, b, alternative="greater").pvalue), 5),
            EAAAK_i_i4=round(float(np.median(d4a)), 1) if len(d4a) else np.nan,
            G4S_i_i4=round(float(np.median(d4b)), 1) if len(d4b) else np.nan))
    if rows:
        HX = pd.DataFrame(rows); display(HX)
        # ★ 프레임 단위 p 는 자유도가 부풀어 있다 (150 프레임을 독립으로 센다).
        #   그래서 판정은 p 가 아니라 **길이 4개에서 부호가 일관되는가 + 효과크기**로 한다.
        _pos = int((HX.차 > 0).sum()); _big = int((HX.효과크기 > 0.3).sum())
        print(f"\n  EAAAK 가 더 나선인 길이: {_pos}/{len(HX)} · "
              f"효과크기 0.3 초과: {_big}/{len(HX)}")
        if _pos == len(HX) and _big >= max(2, len(HX)//2):
            print("  → **국소 조성 채널이 살아 있다.** BioEmu 가 링커 조성을 본다.")
            print("     전역 배향(6절-C)에서 조성이 안 잡히면 그건 '조성을 못 본다' 가 아니라")
            print("     '조성이 전역 배향까지는 전달되지 않는다' 는 뜻이다. 다른 결론이다.")
        else:
            print("  → **국소 조성 채널이 죽어 있다.**")
            print("     교과서적 강직 나선조차 코일과 구분되지 않는다. 이 구성체류에서 BioEmu 의")
            print("     조성 채널은 없는 것이고, 조성에 관한 해석은 여기서 전부 접는 것이 맞다.")
            print("     (전역 배향에서 무엇이 나오든 그건 길이이거나 잡음이다.)")
    else:
        print("  EAAAK/G4S 짝을 못 찾았다 — 4-B절 합성 패널을 먼저 돌려라")
else:
    print("국소 채널 시험 건너뜀 — 합성 패널이나 geom.csv 가 없다")
''')

code(r'''
# ── 6절-E · 표본 독립성 진단 — 순열 p 가 서 있는 땅을 확인한다 ─────────────
# BioEmu 논문은 "thousands of **statistically independent** structure samples" 라고
# 말한다 (Abstract). 우리 검정이 프레임을 독립으로 세는 근거가 그것이다.
#
# ★ 그런데 우리는 `physical_steering.yaml` (SMC 입자 5개) 로 돌렸다. **이건 논문에 없다.**
#   저장소의 수정판이고, SMC 재표집은 한 입자군 안의 표본에 공통 조상을 만든다 —
#   즉 같은 배치 안의 프레임은 서로 **양의 상관**을 갖는다.
#   그러면 유효 표본수가 150보다 작고, 우리 p 값은 낙관적(anti-conservative)이 된다.
#
# 배치 번호는 프레임 색인에서 복원한다 (드라이버가 순서대로 저장한다).
# ICC 가 크면 유효 표본수 n_eff = n / (1 + (m−1)·ICC) 가 확 준다.
BATCH = 5      # BE_BATCH 와 같아야 한다

def icc_by_batch(v, batch, m):
    """일원 분산성분 분해로 급내상관(ICC)을 낸다. v 는 프레임 값, batch 는 배치 번호."""
    v = np.asarray(v, float); batch = np.asarray(batch)
    ks, idx = np.unique(batch, return_inverse=True)
    k = len(ks)
    if k < 3: return np.nan, np.nan
    cnt = np.bincount(idx, minlength=k)
    mu = np.bincount(idx, weights=v, minlength=k) / np.maximum(cnt, 1)
    gm = v.mean()
    msb = float((cnt * (mu - gm)**2).sum() / (k - 1))
    msw = float(((v - mu[idx])**2).sum() / max(len(v) - k, 1))
    n0 = (len(v) - (cnt**2).sum()/len(v)) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1)*msw) if (msb + (n0-1)*msw) > 1e-12 else np.nan
    icc = float(np.clip(icc, 0, 1)) if np.isfinite(icc) else np.nan
    neff = len(v) / (1 + (m - 1)*icc) if np.isfinite(icc) else np.nan
    return icc, neff

_Ei = E[E.생성기 == GEN].copy() if (E is not None and len(E)) else pd.DataFrame()
if len(_Ei):
    _Ei["프레임"] = _Ei.태그.str.extract(r"_s(\d+)$")[0].astype(float)
    _Ei = _Ei[_Ei.프레임.notna()]
if len(_Ei) and BATCH > 1:
    _Ei["배치"] = (_Ei.프레임 // BATCH).astype(int)
    rows = []
    for c in ("ΔHL", "Δdc", "OCD5"):
        if c not in _Ei.columns: continue
        ii, nn = [], []
        for (ab, lk), g in _Ei.groupby(["항체", "링커"], sort=False):
            if len(g) < 30: continue
            i_, n_ = icc_by_batch(g[c].values, g.배치.values, BATCH)
            if np.isfinite(i_): ii.append(i_); nn.append(n_)
        if ii:
            rows.append(dict(지표=c, ICC중앙=round(float(np.median(ii)), 3),
                             ICC최대=round(float(np.max(ii)), 3),
                             n_eff중앙=round(float(np.median(nn)), 1),
                             판정=("독립으로 봐도 된다" if np.median(ii) < 0.05 else
                                   "★ 배치 상관이 있다 — p 가 낙관적이다")))
    print("="*76)
    print(f"★ 표본 독립성 — SMC 입자 {BATCH}개가 배치 안 프레임을 묶어 놓았나")
    print("="*76)
    print("  n_eff = n / (1 + (m−1)·ICC).  ICC 가 0 이면 150 이 그대로 150 이다.")
    print("  ※ physical_steering 은 **논문에 없는 저장소 수정판**이다 —")
    print("     논문의 '통계적으로 독립' 보장이 여기까지 따라오지 않는다. 그래서 직접 잰다.\n")
    if rows:
        display(pd.DataFrame(rows))
        _w = [r for r in rows if "★" in r["판정"]]
        if _w:
            print(f"  ★ {[r['지표'] for r in _w]} 에 배치 상관이 있다.")
            print("     7절의 순열 검정은 **프레임이 아니라 구성체 단위**로 돌기 때문에 안전하다")
            print("     (구성체당 특징 하나로 접은 뒤 항체 안에서 라벨을 섞는다).")
            print("     영향을 받는 것은 **6절-C·6절-D 의 프레임 단위 p** 다 — 그쪽은 효과크기와")
            print("     부호 일관성으로 읽고 p 는 참고로만 보라. 위 셀들이 그렇게 돼 있다.")
        else:
            print("  → 배치 상관이 무시할 만하다. 프레임을 독립으로 세도 된다.")
    else:
        print("  구성체마다 프레임이 모자라 ICC 를 못 낸다")
else:
    print("표본 독립성 진단 건너뜀 (프레임 색인을 못 읽거나 BATCH=1)")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 7절 · ML — 작게, 그러나 제대로

"아주 살짝만 머신러닝" 이 맞다. 여기 필요한 것은 **딥러닝이 아니라 올바른 검정**이다.
데이터가 이렇게 생겼기 때문이다:

```
구성체 26개  ·  그중 링커 3종 이상인 블록 5개  ·  독립 단위는 구성체가 아니라 블록 5개
블록끼리 항체도 정제 공정도 달라서 절대값이 비교 불가
블록 안에서는 링커 길이와 조성이 완전히 얽힘
특징 하나하나가 150 프레임의 통계량이라 자체 오차가 있음
```

### 네 조각이 전부다

**1. 타깃 — 블록 안 중심화한 로짓**
$$y_{ij} = \mathrm{logit}(p_{ij}) - \overline{\mathrm{logit}(p_{\cdot j})}$$
블록 평균을 빼면 남는 것은 "같은 항체에서 링커를 바꿨을 때의 차이" 뿐이고,
그게 이 실험이 답할 수 있는 유일한 질문이다. (계량경제의 within-transformation
= 블록 고정효과와 정확히 같다.) 로짓을 먼저 씌우는 이유: 2% → 4% 와 50% → 52% 는
같은 2%p 지만 응집 평형에서는 전자가 훨씬 큰 변화다.

**2. 검증 — Leave-One-Block-Out**
블록 하나를 통째로 빼고 학습해 그 블록을 예측한다. **표준화도 학습 폴드 안에서만**
적합한다 — 전체로 표준화하면 시험 블록의 평균·분산이 새어 들어가고, n=26 에서는
그것만으로 성능이 올라간다.

**3. 지표 — 블록 안 쌍 일치도**
블록당 링커가 4종이면 Spearman 이 `{±1, ±0.8, ±0.4, ±0.2, 0}` 밖에 못 낸다.
대신 블록 안 쌍(4종이면 6쌍)을 다 모으면 5×6 = **30쌍**이 되고, 귀무가설에서
각 쌍이 1/2 로 맞으므로 **이항분포**라는 해석이 붙는다.

**4. 유의성 — 블록 안 순열 검정. 이것 말고는 없다**
- **귀무 A (신호)** : 블록 안에서 `y` 를 섞는다 → "링커와 물성 사이에 아무것도 없다"
- **귀무 B (증분)** : `y` 는 그대로 두고 **앙상블 특징 행만** 블록 안에서 섞는다
  → "앙상블은 **길이 위에** 아무것도 더하지 않는다"

★ **B 가 진짜 질문이다.** A 만 통과한 상관은 십중팔구 "긴 링커가 나쁘다" 를
다시 발견한 것이다. BioEmu 를 밤새 돌린 값어치는 B 에서만 나온다.

### 그리고 검정력 곡선 — 귀무 결과를 읽는 유일한 방법
라벨을 **알려진 효과크기로 심어** 진짜 분석을 통째로 다시 돌린다.
β=0 행이 거짓양성률(0.05 근처여야 정직하다), 나머지가 검출률이다.

**이 표가 없으면 "상관이 없었다" 는 아무 뜻도 없다.** 없는 것인지 못 본 것인지
가를 방법이 없기 때문이다. 블록 5개 × 링커 4종이면 검출 가능한 효과는
대략 **특징 1 SD 당 타깃 1.3 σ** 부터다 — 작지 않은 효과다.
''')

code(r'''
# ── 7절 · ML 핵심 ──────────────────────────────────────────────────────────
# ★ 안쪽 고리에 pandas 를 두지 않는다. 순열 2000회 × LOBO 5폴드 = 1만 번 적합이고
#   검정력 곡선은 그 200배다. 블록 색인과 쌍 목록을 Design 에 한 번만 굳혀 둔다.

def logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return np.log(p/(1-p))

def as_fraction(v):
    """퍼센트로 들어왔으면 분율로 내린다."""
    v = np.asarray(v, float)
    return v/100.0 if np.nanmax(np.abs(v)) > 1.5 else v

def make_target(F, col, block="항체", higher_is_worse=True):
    """**그룹(항체) 안**에서 중심화한 로짓 타깃. higher_is_worse 면 부호를 뒤집어
    클수록 좋음으로 맞춘다 (HMW 는 낮을수록 좋다)."""
    y = pd.Series(logit(as_fraction(F[col].values)), index=F.index)
    if higher_is_worse: y = -y
    return y.groupby(F[block]).transform(lambda v: v - v.mean())

class Design:
    """그룹 구조를 numpy 로 굳혀 둔 것. 쌍은 그룹을 **가로지르지 않는다**.
    그룹 = 항체 = ABodyBuilder2 기준점 하나를 공유하는 단위다."""
    def __init__(self, F, block="항체"):
        codes = pd.factorize(F[block].values)[0]
        self.blocks = [b for b in (np.where(codes == k)[0]
                                   for k in range(codes.max()+1)) if len(b) >= 2]
        self.pairs = np.array([(i, j) for b in self.blocks
                               for i, j in combinations(b, 2)], dtype=int)
        assert len(self.pairs), "그룹 안 쌍이 하나도 없다 — 항체마다 구성체가 1개뿐인가?"
    def center(self, y):
        o = np.asarray(y, float).copy()
        for b in self.blocks: o[b] -= o[b].mean()
        return o
    def permute(self, y, rng):
        """그룹 **안에서만** 섞는다. 그룹 구조와 주변 분포는 그대로."""
        o = np.asarray(y, float).copy()
        for b in self.blocks: o[b] = o[b][rng.permutation(len(b))]
        return o
    def permute_rows(self, X, rng):
        """특징 **행**을 그룹 안에서 섞는다 (증분 검정용)."""
        o = np.asarray(X, float).copy()
        for b in self.blocks: o[b] = o[b][rng.permutation(len(b))]
        return o

def concordance(pred, y, D):
    """그룹 안 모든 쌍에서 예측 순서가 실측 순서와 맞는가. NaN·동점은 안 센다."""
    i, j = D.pairs[:, 0], D.pairs[:, 1]
    dp, dy = pred[i]-pred[j], y[i]-y[j]
    ok = np.isfinite(dp) & np.isfinite(dy) & (np.abs(dp) > 1e-12) & (np.abs(dy) > 1e-12)
    n = int(ok.sum())
    if not n: return 0, 0, np.nan
    a = int((np.sign(dp[ok]) == np.sign(dy[ok])).sum())
    return a, n, a/n

def _ridge(Xtr, ytr, Xte, alpha):
    """학습 폴드에서 **표준화까지 적합**하고 시험 폴드를 예측한다.
    ★ 전체로 표준화하면 시험 블록이 학습에 샌다."""
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Xtr-mu)/sd
    coef = np.linalg.solve(Z.T @ Z + alpha*np.eye(Z.shape[1]), Z.T @ (ytr-ytr.mean()))
    return ((Xte-mu)/sd) @ coef + ytr.mean(), coef

def lobo_predict(X, y, D, alpha=1.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim == 1: X = X[:, None]
    pred, coefs = np.full(len(y), np.nan), []
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    for b in D.blocks:
        s  = set(b.tolist())
        te = b[ok[b]]
        tr = np.array([i for i in range(len(y)) if ok[i] and i not in s], dtype=int)
        if len(te) < 2 or len(tr) < X.shape[1] + 2: continue
        p, c = _ridge(X[tr], y[tr], X[te], alpha); pred[te] = p; coefs.append(c)
    return pred, (np.array(coefs) if coefs else np.zeros((0, X.shape[1])))

def lobo_score(X, y, D, alpha=1.0):
    return concordance(lobo_predict(X, y, D, alpha)[0], np.asarray(y, float), D)

def perm_test_signal(X, y, D, alpha=1.0, n_perm=N_PERM, seed=0):
    """귀무 A: 링커와 물성 사이에 아무 관계도 없다. 블록 안에서 y 를 섞는다."""
    X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X
    y = np.asarray(y, float)
    a, n, obs = lobo_score(X, y, D, alpha)
    rng = np.random.default_rng(seed)
    null = np.array([lobo_score(X, D.permute(y, rng), D, alpha)[2] for _ in range(n_perm)])
    fin = np.isfinite(null)
    return dict(일치=f"{a}/{n}", 일치도=round(float(obs), 3),
                귀무평균=round(float(np.nanmean(null)), 3),
                귀무95=round(float(np.nanpercentile(null[fin], 95)), 3),
                순열p=round(float((np.sum(null[fin] >= obs)+1)/(fin.sum()+1)), 4),
                이항p=round(float(st_.binomtest(a, n, 0.5, alternative="greater").pvalue), 4)
                       if n else np.nan,
                _null=null, _obs=obs)

def perm_test_increment(Xb, Xa, y, D, alpha=1.0, n_perm=N_PERM, seed=0):
    """귀무 B: 앙상블 특징은 **길이 위에** 아무것도 더하지 않는다.
    y 는 그대로 두고 앙상블 특징 **행만** 블록 안에서 섞는다 — 길이와 y 의 관계는
    보존한 채 앙상블 특징이 링커에 붙어 있다는 사실만 끊는다."""
    Xb = np.atleast_2d(np.asarray(Xb, float).T).T
    Xa = np.atleast_2d(np.asarray(Xa, float).T).T
    y  = np.asarray(y, float)
    b = lobo_score(Xb, y, D, alpha)[2]
    f = lobo_score(np.hstack([Xb, Xa]), y, D, alpha)[2]
    obs = f - b
    rng = np.random.default_rng(seed)
    null = np.array([lobo_score(np.hstack([Xb, D.permute_rows(Xa, rng)]), y, D, alpha)[2] - b
                     for _ in range(n_perm)])
    fin = np.isfinite(null)
    return dict(길이만=round(float(b), 3), 길이_앙상블=round(float(f), 3),
                증분=round(float(obs), 3), 귀무평균=round(float(np.nanmean(null)), 3),
                순열p=round(float((np.sum(null[fin] >= obs)+1)/(fin.sum()+1)), 4),
                _null=null, _obs=obs)

def power_curve(x, D, betas=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0), n_sim=200, n_perm=300,
                alpha=1.0, seed=0, sigma=1.0):
    """실제 특징값은 그대로 두고 **라벨만** 알려진 효과크기로 심어 만든다.
    β=0 행이 거짓양성률. β 단위: 특징 1 SD 당 타깃 몇 σ."""
    x = np.asarray(x, float)
    z = (x - np.nanmean(x))/(np.nanstd(x) if np.nanstd(x) > 1e-12 else 1.0)
    rng, rows = np.random.default_rng(seed), []
    for b in betas:
        hit = 0
        for _ in range(n_sim):
            y = D.center(b*z + rng.normal(0, sigma, len(z)))
            hit += int(perm_test_signal(z[:, None], y, D, alpha, n_perm,
                                        int(rng.integers(1 << 30)))["순열p"] < ALPHA)
        rows.append(dict(효과크기=b, 검출률=round(hit/n_sim, 3), 시행=n_sim))
    return pd.DataFrame(rows)

def min_detectable_effect(P, target=0.8):
    P = P.sort_values("효과크기"); b, d = P.효과크기.values, P.검출률.values
    h = np.where(d >= target)[0]
    if not len(h): return np.inf
    k = h[0]
    if k == 0: return float(b[0])
    return (float(b[k-1] + (target-d[k-1])*(b[k]-b[k-1])/(d[k]-d[k-1]))
            if d[k] > d[k-1] else float(b[k]))

def holm(ps):
    """Holm-Bonferroni 보정 p. 탐색 특징 3개를 동시에 읽을 때 쓴다."""
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m-i)*ps[k]); adj[k] = min(run, 1.0)
    return adj

print("7절 함수 준비 완료")
''')

code(r'''
# ── 7절-B · 판정 — 확증 · 탐색 · 증분 · 검정력 ─────────────────────────────
# 6절-B 를 안 돌리고 이 셀만 다시 돌려도 죽지 않게 한다 (원장 결과가 없으면 빈 채로)
DEGENERATE = DEGENERATE if "DEGENERATE" in dir() else set()
RELY       = RELY       if "RELY"       in dir() else {}

if not (RUN_ML and HC and len(USE) >= 6 and USE[GROUP].nunique() >= 3):
    print("7절 건너뜀 —", "RUN_ML=False" if not RUN_ML else
          ("실측 열이 없다" if not HC else
           f"링커 3종 이상 항체가 {USE[GROUP].nunique()}개뿐이다 (3개 이상 필요)"))
else:
    D = Design(USE, block=GROUP)
    Y = make_target(USE, HC, block=GROUP, higher_is_worse=HIGHER_IS_WORSE).values
    print("="*76)
    print(f"★ 7절 판정 — 타깃 {HC} (항체 안 로짓 중심화, "
          f"{'낮을수록 좋은 값이라 부호를 뒤집었다' if HIGHER_IS_WORSE else '클수록 좋은 값이다'})")
    print("="*76)
    print(f"  구성체 {len(USE)} · **독립 단위(항체) {USE[GROUP].nunique()}개** · "
          f"항체 안 쌍 {len(D.pairs)}개 · 순열 {N_PERM}회")
    print(f"  한 쌍이 우연히 맞을 확률이 1/2 이므로 귀무 일치도는 0.50 이다.\n")

    # ── ① 확증 검정 — 사전 등록 특징 하나. 보정 없음 ──────────────────────
    print("─"*76); print(f"① 확증 검정 — {PRIMARY} (사전 등록, 다중비교 보정 없음)")
    print("─"*76)
    # ★ 확증 특징이 **상수**면 검정 자체가 정의되지 않는다. 그런데 이건 사고가 아니라
    #   결과다 — 짝지음 게이트가 한 번도 안 걸렸다는 뜻이고(모든 프레임이 붙어 있다),
    #   그러면 '열림' 이라는 축 위에 구성체를 가를 것이 없다. 그냥 두면 이 사실이
    #   'NaN 을 int 로 못 바꾼다' 라는 무관한 문장으로 보고되고 7절 전체가 멈춘다.
    _pv = pd.to_numeric(USE[PRIMARY], errors="coerce").values
    PRIM_OK = bool(np.isfinite(_pv).sum() >= len(USE) - 1
                   and np.nanstd(_pv) > 1e-12)
    _npair = len(D.pairs)
    if not PRIM_OK:
        r1 = None
        _v = _pv[np.isfinite(_pv)]
        print(f"  ★ {PRIMARY} 가 구성체 전체에서 **상수다** "
              f"(값 {(_v[0] if len(_v) else float('nan')):.3g}, 유효 {len(_v)}/{len(USE)}).")
        print(f"     짝지음 게이트(|dc−{DC0}| < {PAIR_NSD}σ)가 한 번도 안 걸렸다 —")
        print(f"     즉 **모든 프레임이 붙어 있다.** '효과가 없다' 가 아니라")
        print(f"     '이 축에는 잴 변동 자체가 없다' 이다. ② 탐색 특징으로 간다.")
    else:
        r1 = perm_test_signal(USE[PRIMARY].values, Y, D, seed=1)
        _need = int(np.ceil(r1["귀무95"] * _npair))
        print(f"  일치도 {r1['일치도']:.3f} ({r1['일치']})  vs  귀무평균 {r1['귀무평균']:.3f} "
              f"· 귀무 95% {r1['귀무95']:.3f}")
        print(f"  → p ≤ 0.05 에 닿으려면 **{_npair}쌍 중 {_need}쌍**을 맞혀야 한다.")
        print(f"  순열 p = {r1['순열p']:.4f}   (참고: 이항 p = {r1['이항p']:.4f})")
        print(f"  → {'**신호가 있다.**' if r1['순열p'] < ALPHA else '신호 없음.'}")
    if r1 is not None and r1["순열p"] >= ALPHA and np.isfinite(RELY.get(PRIMARY, np.nan)) \
            and RELY[PRIMARY] < 0.7:
        print(f"     ※ 단, 6절-B 가 이 특징의 신뢰도를 {RELY[PRIMARY]:.2f} 로 쟀다 —")
        print(f"       '효과가 없다' 보다 **'측정이 시끄럽다'** 가 먼저다. 프레임을 더 뽑아라.")

    # ── ② 탐색 검정 — Holm 보정 ───────────────────────────────────────────
    print("\n" + "─"*76); print("② 탐색 검정 — 나머지 3개, Holm 보정")
    print("─"*76)
    ex = [c for c in EXPLORATORY if USE[c].notna().sum() >= len(USE)-1
          and USE[c].std(ddof=1) > 1e-12]
    if ex:
        rs = [perm_test_signal(USE[c].values, Y, D, seed=10+i) for i, c in enumerate(ex)]
        T = pd.DataFrame([dict(특징=c, 일치=r["일치"], 일치도=r["일치도"],
                               순열p=r["순열p"]) for c, r in zip(ex, rs)])
        T["Holm_p"] = holm(T.순열p.values).round(4)
        T["판정"]   = np.where(T.Holm_p < ALPHA, "유의 (탐색)", "—")
        display(T)
    else:
        print("  쓸 수 있는 탐색 특징이 없다 (전부 NaN 이거나 변화가 없다)")
        T = pd.DataFrame()

    # ── ③ 증분 검정 — 이게 진짜 질문이다 ──────────────────────────────────
    print("\n" + "─"*76)
    print("③ 증분 검정 — 앙상블이 **링커 길이 위에** 무언가 더하는가")
    print("─"*76)
    print("  y 는 안 섞는다. 앙상블 특징 **행만** 항체 안에서 섞어, 길이와 y 의 관계는")
    print("  그대로 둔 채 앙상블 특징이 링커에 붙어 있다는 사실만 끊는다.\n")
    inc = []
    for nm, cols in [("확증 특징만", [PRIMARY] if PRIM_OK else []),
                     ("사전 등록 4개", [c for c in ALLF if c in USE.columns and
                                       USE[c].notna().all() and USE[c].std(ddof=1) > 1e-12])]:
        if not cols: continue
        r = perm_test_increment(USE[BASELINE].values, USE[cols].values, Y, D,
                                seed=100+len(inc))
        inc.append(dict(앙상블특징=nm, 길이만=r["길이만"], 길이_앙상블=r["길이_앙상블"],
                        증분=r["증분"], 순열p=r["순열p"],
                        판정="**길이 위에 더한다**" if r["순열p"] < ALPHA else "더하지 않는다"))
    INC = pd.DataFrame(inc); display(INC)
    if DEGENERATE >= set([PRIMARY]):
        print(f"  ★★ '{PRIMARY}' 이(가) 항체마다 길이와 rho = ±1 이다 — 증분은 **수학적으로 0 에**")
        print("     묶여 있다. 위 p 는 '효과 없음' 이 아니라 **'이 설계로는 답할 수 없음'** 이다.")
        print("     4-B절 합성 패널(같은 길이 × 다른 조성)이 유일한 탈출구다.")
    else:
        print("  → 증분이 유의하지 않으면, 나온 상관은 **'긴 링커가 나쁘다'의 다른 말**이다.")
        print("     BioEmu 를 밤새 돌린 값어치는 이 줄에서만 나온다.")

    # ── ④ 검정력 — 귀무 결과를 읽는 유일한 방법 ───────────────────────────
    print("\n" + "─"*76); print("④ 검정력 — '상관이 없었다' 가 무슨 뜻인지")
    print("─"*76)
    print("  실제 특징값은 그대로 두고 **라벨만** 알려진 효과크기로 심어 진짜 분석을")
    print("  통째로 다시 돌린다. β=0 행이 거짓양성률이다 (0.05 근처여야 정직하다).\n")
    # 곡선은 **특징값의 실제 분포** 위에 라벨을 심어 그린다. 상수 특징으로는 그릴 수
    # 없으므로, 그럴 때만 변동이 있는 다른 사전 등록 특징으로 갈아타고 그 사실을 적는다.
    _pwf = PRIMARY if PRIM_OK else next(
        (c for c in EXPLORATORY if c in USE.columns
         and USE[c].notna().all() and USE[c].std(ddof=1) > 1e-12), None)
    if _pwf is None:
        PW, mde = pd.DataFrame(), float("nan")
        print("  ★ 사전 등록 특징 중 변동이 있는 것이 하나도 없다 — 검정력 곡선을 못 그린다.")
    else:
        if _pwf != PRIMARY:
            print(f"  ※ {PRIMARY} 이(가) 상수라 **{_pwf}** 의 분포 위에서 그린다 "
                  f"(설계의 검정력이지 그 특징의 성능이 아니다).\n")
        PW = power_curve(USE[_pwf].values, D, n_sim=200, n_perm=300, seed=7)
        display(PW)
        mde = min_detectable_effect(PW, 0.8)
        print(f"  거짓양성률(β=0) = {PW.검출률.iloc[0]:.3f}")
        print(f"  검출률 80% 에 닿는 최소 효과 ≈ β = {mde:.2f}  "
              f"(특징 1 SD 가 타깃을 {mde:.1f} σ 움직여야 한다)")
    if r1 is not None and np.isfinite(mde) and r1["순열p"] >= ALPHA:
        print(f"\n  ★ 확증 검정이 귀무였다. 위 곡선을 같이 읽어라 —")
        print(f"     β = {mde:.1f} 보다 작은 효과는 이 설계(항체 {USE[GROUP].nunique()}개 ×"
              f" 링커 {int(USE.groupby(GROUP).size().median())}종)로는 **애초에 못 본다.**")
        print(f"     '효과가 없다' 가 아니라 '{mde:.1f} σ 보다 작은 효과는 못 봤다' 가 맞다.")

    # ── ⑥ 그래서 다음에 뭘 만들어야 하나 — 측정해서 답한다 ────────────────
    print("\n" + "─"*76); print("⑥ 다음 설계 — 항체를 늘릴까, 항체당 링커를 늘릴까")
    print("─"*76)
    print("  검정 통계량이 **항체 안 쌍**이라 쌍 수 = 항체 × C(링커,2) 다.")
    print("  링커에는 **제곱으로**, 항체에는 **선형으로** 는다. 직관과 다르다.\n")
    _g0, _k0 = USE[GROUP].nunique(), int(round(USE.groupby(GROUP).size().mean()))
    from math import comb
    plans = [(f"지금:   항체{_g0} × 링커{_k0}", _g0, _k0),
             (f"링커+2: 항체{_g0} × 링커{_k0+2}", _g0, _k0+2),
             (f"링커+4: 항체{_g0} × 링커{_k0+4}", _g0, _k0+4),
             (f"항체+3: 항체{_g0+3} × 링커{_k0}", _g0+3, _k0),
             (f"항체+7: 항체{_g0+7} × 링커{_k0}", _g0+7, _k0)]
    display(pd.DataFrame([dict(설계=nm, 새구성체=g*k - _g0*_k0,
                               총구성체=g*k, 항체안쌍=g*comb(k, 2))
                          for nm, g, k in plans]))
    print("  ★ 같은 수를 만든다면 **기존 항체에 링커를 더하는 쪽**이 쌍을 훨씬 많이 준다.")
    print("    (모의 검정력, β=1.0 기준: 항체5×링커4 = 0.52 → 항체5×링커6 = 0.93")
    print("     vs 항체8×링커4 = 0.85. 구성체 수는 30 대 32 로 비슷한데도 그렇다.)")
    print("  ※ 단, 항체를 늘리면 **일반화 범위**가 는다 — 링커를 늘리면 그 5개 항체에서의")
    print("    검출력만 는다. 결론을 몇 개 항체까지 주장하고 싶은가가 그 선택을 정한다.")
    print("  ※ 합성 링커는 여기 안 든다. 라벨이 없으므로 쌍을 만들지 못한다 —")
    print("    합성 패널이 사는 것은 **검출력이 아니라 4-B절의 설계 해상도**다.")

    # ── ⑤ 계수 — 부호가 블록마다 같은가 ───────────────────────────────────
    _cols = [c for c in ALLF if c in USE.columns and USE[c].notna().all()
             and USE[c].std(ddof=1) > 1e-12]
    if _cols:
        _, CF = lobo_predict(USE[BASELINE+_cols].values, Y, D)
        if len(CF):
            C = pd.DataFrame(CF, columns=BASELINE+_cols)
            print("\n" + "─"*76); print("⑤ LOBO 폴드별 계수 — 부호가 폴드마다 뒤집히면 못 믿는다")
            print("─"*76)
            display(C.round(3).assign(폴드=[f"{b} 제외" for b in
                        USE[GROUP].unique()[:len(C)]]).set_index("폴드"))
            S = pd.DataFrame(dict(평균=C.mean().round(3), SD=C.std(ddof=1).round(3),
                                  부호일치=[f"{max((C[c]>0).sum(), (C[c]<0).sum())}/{len(C)}"
                                            for c in C.columns]))
            display(S)
    ML_RESULT = dict(확증=r1, 탐색=T, 증분=INC, 검정력=PW, MDE=mde)
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 7절-C · 계면 상호작용 — **같은 링커인데 블록마다 다른 이유**

### 묻는 것 하나
> 같은 만큼 열렸을 때, **계면이 센 Fv 가 그 열림을 더 많은 응집으로 바꾸는가.**

7절 ① 은 "링커가 열면 나빠지는가" 를 물었다. 여기서는 **그 기울기 자체가 항체마다
다른가, 그리고 그 차이를 계면 강도가 설명하는가** 를 묻는다.

```
y_ij = α_j  +  β·열림_ij  +  γ·(계면강도_j × 열림_ij)  +  ε        사전 등록: γ < 0
       ↑         ↑                    ↑
   항체 절편   링커 효과        계면이 셀수록 기울기가 가팔라진다
   (중심화가    (7절 ①)          (여기가 새로 묻는 것)
    지운다)
```

### 왜 상호작용이어야만 하는가 — 주효과로는 원리적으로 못 넣는다
계면 강도는 **항체마다 상수**다. 7절의 항체 안 중심화는 항체마다 상수인 것을
정의상 **완전히** 지운다. 모의로 확인하면 중심화 뒤 항체별 평균의 폭이 3×10⁻¹⁶ 다.

그런데 **상호작용은 통과한다.** 열림분율이 항체 **안에서** 변하기 때문이다.
모의에서 상호작용을 심으면 항체별 기울기가 계면강도를 r = 0.96 으로 따라간다.
그래서 이 축은 상호작용으로만 물을 수 있고, 다행히 그게 물어야 할 형태이기도 하다.

### 검정통계량과 귀무가설
**통계량**은 항체별 기울기를 계면강도에 회귀한 **정밀도가중 t** 다.

Δ일치도를 쓰면 안 된다. 항체 안에서 예측이 특징의 **양의 재척도**일 뿐이라
일치도가 `sign(β + γS_j)` 만 보고, 통계량이 항체 수만큼의 **이진 부호**로 붕괴한다.
모의에서 γ=3 일 때 Δ일치도 0.27 대 기울기회귀 0.92 다. (앞서 본 "γ=0 에서 검출률
0.00" 도 보수적인 게 아니라 이 붕괴의 증상이었다 — 정상적인 정확검정이면 0.05 다.)
Spearman 상관도 그보다는 낫지만 기울기회귀보다 25~30% 약하다.

**귀무가설**: 항체별 기울기 벡터의 결합분포가 **S 라벨 치환에 불변**이다.
→ 항체 수준 계면강도를 항체끼리 재배치한다. 항체 **안** 링커 배열은 안 건드린다
(섞으면 β 까지 지워져 귀무가 γ=0 이 아니라 β=γ=0 이 되고, 주효과만 있어도 기각한다).

### ★★ 작동점 교란 — 통제 안 하면 거짓양성이 0.41 까지 간다
S 라벨 치환이 정확하려면 S 가 항체의 **작동점**과 무관해야 한다. 그런데 열역학이
말하듯 **센 계면일수록 덜 열린다** — S 와 항체별 평균 열림분율이 기계적으로 얽혀 있다.
그리고 y 가 x 에 조금이라도 곡선으로 붙으면 (로짓이 보장한다) 국소 기울기가
작동점의 함수가 되어 **γ=0 인데도 기각된다.**

모의 거짓양성 (γ를 정확히 0 으로 두고):

| ρ(S, 평균열림) | 통제 없음 | Freedman–Lane 통제 |
|---|---|---|
| 0.0 | 0.054 | 0.053 |
| 0.7 | **0.205** | 0.049 |
| 0.9 | **0.331** | 0.051 |
| 1.0 | **0.417** | 0.052 |

→ **Freedman–Lane** 로 S 를 평균열림에 잔차화한 뒤 잔차만 섞는다. 검정력을 25~45%
치르지만 이걸 안 하면 검정이 아니다.

→ 그리고 **평균열림 자체를 경쟁 축으로 반드시 본다.** AF2 도 BSA 도 필요 없는 공짜
경쟁자이고, 곡률만 있어도 귀무 자료의 48.5% 에서 발화한다. 그게 유의하면 계면강도가
설명하는 것이 "계면" 인지 "항체마다 작동점이 다르다" 인지 가를 수 없다.

### ★ 측정오차가 만드는 **검정력 천장**
상호작용은 두 신뢰도의 **곱**으로 눌린다: `γ̂ = γ · λ_S · λ_O`.
(주효과는 하나만 들어온다. 상호작용이 훨씬 가혹하다.)

λ_S = λ_O = 0.7 이면 항체 5개에서 **어떤 효과크기로도 80% 검정력에 못 닿는다.**
셀이 두 신뢰도와 유효 γ 를 찍고, 곱이 0.5 미만이면 그렇게 인쇄한다.

**귀무가설**: 계면강도는 그 항체의 (log 열림 → 물성) 민감도와 무관하다.
→ 항체 수준 계면강도 값을 **항체끼리 재배치**한다. 항체 안의 링커 배열은 안 건드린다.
기울기는 관측된 그대로 두고, 거기에 어느 계면강도가 붙느냐만 끊는 것이다.

항체가 k 개면 가짓수가 k! 이고, k ≤ 7 이면 **전부 세어 정확 p** 를 낸다.
k=5 면 120가지, 최소 p 는 1/120 = 0.0083 이다. **양측**이므로 사실상 그 두 배다.

그리고 이 분해능이 곧 **다중비교 예산**이다. Bonferroni 로 K 개를 보려면
p ≤ 0.05/K 여야 하는데, K=2 만 넘어도 어떤 효과크기로도 기각이 불가능해진다.
→ **확증 검정은 정확히 하나다**: 특징은 `열림분율`, 계면축은 y-무관 규칙이 고르는
단 하나. 나머지(다른 계면 지표 · 다른 앙상블 특징 · Pearson r)는 **p 값 없이**
표로만 찍는다. 숫자는 보여 주되 유의성은 주장하지 않는다.
항체가 6개면 720가지, 7개면 5040가지라 이 제약이 빠르게 풀린다.

### ★ 항체 5개에서 제일 중요한 것은 **반증 원장**이다
항체가 k 개면 가능한 순서가 k! 가지다. k=5 면 **120뿐**이고, 완벽한 순서(|rho|=1)가
우연히 나올 확률이 축 하나당 2/120 ≈ 1.7% 다. **축을 열 개 재면 대략 하나는 우연히
완벽하게 정렬된다.** 계면강도 하나만 재고 "맞았다" 고 하면 그 하나를 본 것이다.

→ 그래서 계면과 무관한 축들 — CDR-H3 길이, Fv 순전하, Fv 소수성, **난수 축 둘**,
항체 이름 알파벳순 — 에도 **똑같은 검정**을 걸어 나란히 세운다.

**사전 등록된 읽는 법**: 무관한 축 **3개 이상**이 |rho| ≥ 0.9 를 찍으면, 계면강도의
p 가 얼마든 정보가 없다. 그때 유의한 것은 축이 아니라 **n=5 라는 사실**이다.

### 그리고 지표가 시끄러우면 있어도 못 쓴다
ABB2 4모델의 흩어짐이 곧 "같은 항체를 다시 재면 얼마나 다른가" 다.
3절-C 가 지표마다 신뢰도 = 1 − SD_항체내²/SD_항체간² 를 재고,
**신뢰도 < 0.7 이면 7절-C 에 못 들어간다.** 라벨을 만지기 전에 정하는 규칙이다.

### 순서가 있다 — ★ **관문은 y 를 안 쓰는 것만이다**
실측(y)을 쓰는 검정은 전부 **보고용이지 관문이 아니다.** 그래야 "결과를 보고 규칙을
바꿨다" 는 의심이 원천적으로 없다.

1. **y-무관 사전점검** (유일한 관문): S 가 실제로 변하는가, 레버리지가 한 항체에
   쏠렸는가, S 가 항체별 정밀도·링커수·**평균 열림분율**과 얽혔는가, λ_S ≥ 0.7 인가.
2. **확증 검정**: 정밀도가중 기울기회귀 + Freedman–Lane, 정확 순열, 양측. **하나뿐.**
3. **취약성**: 항체 하나를 빼면 결론이 바뀌는가. 바뀌면 그렇게 보고한다.
4. **반증 원장**: 무관한 축들과 **공짜 경쟁자(평균열림)** 도 같은 순서를 만드는가.
5. **맥락**: 기울기 이질성 F. ★ 이건 **관문이 아니다** — 전방위 검정이라 방향을 아는
   ② 보다 둔하다 (γ=1 에서 0.39 대 0.52). 게이트로 쓰면 0.27 로 떨어진다.

### 이 검정만 포함 규칙을 완화한다
7절 ① 의 쌍 일치도는 항체 안 **쌍**이 필요해 링커 3종 이상이었다. 여기서는 항체마다
기울기 **하나**만 있으면 되므로 **링커 2종 이상**으로 넓힌다. 항체가 5개에서 7개로만
늘어도 순열 가짓수가 120 → 5040 이라 **최소 양측 p 가 0.0165 → 0.0004** 로 내려간다.
이 검정에서 가장 값싼 개선이다.
5. **검정력**과 **다음 설계**: 지금 설계로 무엇이 보이는지 재고, 항체를 늘리는 쪽과
   항체당 링커를 늘리는 쪽을 **직접 모의해서** 비교한다. 상호작용은 두 축 모두에
   반응한다 — 항체를 늘리면 기울기 개수와 순열 가짓수가 늘고, 링커를 늘리면 기울기
   하나하나가 정밀해져 상관의 감쇠가 준다. 어느 쪽이 값싼지는 셀이 표로 답한다.
   7절 ⑥ 의 주효과 표와 답이 다를 수 있고, 다른 게 정상이다.

### ★ 함정 하나 — 날 기울기를 쓰면 로짓 곡률이 가짜 신호를 만든다
타깃이 로짓이라 `d(logit p)/dp = 1/(p(1−p))` 다. 즉 **기준선 HMW 가 낮은 항체는
같은 %p 변화에도 로짓에서 훨씬 큰 기울기**가 나온다. 기준선이 계면 지표와 조금이라도
상관되면 (충분히 그럴 수 있다 — 끈끈한 계면은 기준선 HMW 자체도 높일 테니)
이 검정은 생물이 아니라 **변환의 곡률**을 잡는다.

모의 데이터로 확인했다. 상호작용을 **안 심었는데** 날 기울기로는
`rho = −0.9, 단측 p = 0.05` 가 나왔다. 기준선이 우연히 계면 지표를 따라갔기 때문이다.

→ 그래서 민감도를 **항체 안 Pearson 상관** (= 기울기 × SD(x)/SD(y)) 으로 쓴다.
SD(y) 가 같은 곡률 인자로 커지므로 인자가 정확히 상쇄된다. 검정통계량이 Spearman 이라
단조변환에 둔감하니 잃는 것도 없다. 셀이 기준선과 계면강도의 상관도 같이 찍는다.

### 미리 인정할 것 넷
- **계면 강도의 대리지표는 친화도가 아니다.** ipTM 은 자기확신이고 BSA 는 면적이다.
  둘 다 ΔG 가 아니다. 순위 정도로만 읽는다. 3절-B 에 이유를 자세히 적었다.
- **조건화가 새면 부호가 뒤집힌다.** 평형만 놓고 보면 계면 친화도는 이량체 분율에서
  정확히 **상쇄된다**(3절-B). 조건부로만 살아남는데, 그 조건화는 BioEmu 의 열림분율이
  친화도를 반영할 때만 성립한다. 그래서 **양측** 검정이고, 부호가 곧 해석이다.
- **경쟁 설명을 못 가른다.** 계면강도가 CDR-H3 길이 같은 알려진 응집 인자와 겹치면,
  항체 5개로는 둘을 분리할 수 없다. 3절-C 가 그 표를 찍고, 겹치면 그렇게 보고한다.
- **항체 5개다.** 아주 큰 상호작용만 보인다. 귀무가 나와도 대개는 "없다" 가 아니라
  "이 표본으로는 못 본다" 다. ④ 가 그 구분을 숫자로 준다.
''')

code(r'''
# ── 7절-C 핵심 · 항체별 기울기 → 계면강도 회귀 ─────────────────────────────
# ★ Δ일치도로 하면 안 된다. 항체 안에서 예측이 특징의 **양의 재척도**일 뿐이라
#   일치도가 sign(β+γS_j) 만 본다 — 통계량이 항체 수만큼의 **이진 부호**로 붕괴한다.
#   모의: γ=3 에서 Δ일치도 0.27 vs 정밀도가중 기울기회귀 0.92.
#   (우리가 앞서 본 "γ=0 에서 검출률 0.00" 도 보수적인 게 아니라 이 붕괴의 증상이었다.)
from itertools import permutations
from math import factorial, sqrt

def antibody_slopes(y, x, groups):
    """항체별 기울기 b_j 와 정밀도. y 는 이미 항체 안 중심화된 타깃.
    σ² 는 **항체끼리 풀링**한다 — 링커 3종이면 잔차 자유도가 1 뿐이라
    항체별 SE 는 못 믿는다."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    g = pd.Series(groups).astype(str).values
    rows, rss, dof = [], 0.0, 0
    for ab in pd.unique(g):
        m = (g == ab) & np.isfinite(y) & np.isfinite(x)
        if m.sum() < 3: continue          # 기울기+절편+잔차 1개는 있어야 한다
        yy = y[m] - y[m].mean(); xx = x[m] - x[m].mean()
        sxx = float(xx @ xx)
        if sxx < 1e-12: continue          # 항체 안에서 특징이 상수
        b = float(xx @ yy / sxx); r = yy - b*xx
        rss += float(r @ r); dof += m.sum() - 2
        sy = yy.std(ddof=0)
        rows.append(dict(항체=ab, k=int(m.sum()), b=b, Sxx=sxx,
                         r_within=float(b*xx.std(ddof=0)/sy) if sy > 1e-12 else np.nan,
                         mean_x=float(x[m].mean())))
    T = pd.DataFrame(rows)
    assert len(T), "기울기를 낼 수 있는 항체가 없다 (항체당 링커 3종 이상 필요)"
    sig2 = rss/dof if dof > 0 else np.nan
    T["se"] = np.sqrt(sig2/T.Sxx.values)
    return T

def _wls_t(b, S, w):
    sw = w.sum(); d = S - (w*S).sum()/sw
    den = float((w*d*d).sum())
    if den <= 1e-300: return 0.0, 0.0
    gam = float((w*d*(b - (w*b).sum()/sw)).sum()/den)
    return gam, gam*sqrt(den)

def _resid(v, C):
    C = C - C.mean(); v = v - v.mean(); ss = float(C @ C)
    return v - (float(C @ v)/ss)*C if ss > 1e-12 else v

def interaction_test(slopes, S, covariate=None, sided="two", max_exact=5040,
                     n_rand=20000, seed=0):
    """항체 수준 S 라벨만 재배치하는 정확 순열 검정.

    H0: 항체별 기울기 벡터의 결합분포가 **S 라벨 치환에 불변**이다.
    ★ 항체 **안** 링커 라벨은 절대 안 섞는다. 섞으면 β 까지 지워져 귀무가
      γ=0 이 아니라 β=γ=0 이 되고, 주효과만 있어도 기각한다.

    ★★ covariate 를 주면 **Freedman–Lane** 로 그 항체 수준 교란을 통제한다.
       여기서 반드시 줘야 하는 것은 **항체별 평균 열림분율**이다. 이유:
       센 계면일수록 덜 열리므로 S 와 작동점이 기계적으로 얽혀 있고,
       y 가 x 에 조금이라도 곡선으로 붙으면(로짓이 보장한다) 국소 기울기가
       작동점의 함수가 되어 γ=0 인데도 기각된다.
       모의 거짓양성: 통제 없이 ρ(S, 평균열림)=0.9 에서 **0.33**, 1.0 에서 **0.41**.
       Freedman–Lane 를 걸면 전 구간 0.049~0.053 으로 돌아온다.
    """
    b = slopes.b.values.astype(float); S = np.asarray(S, float); m = len(b)
    assert len(S) == m, "S 의 길이가 항체 수와 다르다"
    w = 1.0/(slopes.se.values**2)          # y 정보에 의존하지 않는 가중
    C = None if covariate is None else np.asarray(covariate, float)
    exact = factorial(m) <= max_exact
    P = (np.array(list(permutations(range(m)))) if exact else
         np.vstack([np.arange(m),
                    [np.random.default_rng(seed+t).permutation(m) for t in range(n_rand-1)]]))
    if C is None:
        bb = b; make = lambda p: S[p]
    else:
        Cc = C - C.mean()
        fit = (float(Cc @ (S - S.mean()))/float(Cc @ Cc))*Cc if float(Cc @ Cc) > 1e-12 else 0*Cc
        Sr = (S - S.mean()) - fit
        bb = _resid(b, C); make = lambda p: Sr[p] + fit
    T = np.empty(len(P)); G = np.empty(len(P))
    for t, pm in enumerate(P):
        Sp = make(pm)
        if C is not None: Sp = _resid(Sp, C)
        G[t], T[t] = _wls_t(bb, Sp, w)
    og, ot = G[0], T[0]
    p = (float((T >= ot - 1e-12).mean()) if sided == "greater" else
         float((T <= ot + 1e-12).mean()) if sided == "less" else
         float((np.abs(T) >= abs(ot) - 1e-12).mean()))
    d = S - S.mean(); Sxx = float(d @ d)
    return dict(항체수=m, gamma=round(og, 4),
                se_gamma=round(abs(og/ot), 4) if abs(ot) > 1e-12 else np.nan,
                t=round(ot, 3),
                spearman=round(float(st_.spearmanr(b, S).statistic), 3) if m > 2 else np.nan,
                순열p=round(p, 5), 정확=bool(exact), 순열수=len(P),
                최소가능p=round(1.0/len(P), 5),
                통제=("Freedman–Lane" if C is not None else "없음"),
                레버리지=np.round((d**2)/Sxx, 2) if Sxx > 1e-12 else None,
                _null=T, _obs=ot)

def slope_heterogeneity(y, x, groups):
    """항체별 기울기가 애초에 다른가 — 공통기울기 대 개별기울기 정확 F.
    ★ 이걸로 상호작용 검정을 **막지 마라.** 모의에서 γ=1 일 때 상호작용 0.52 vs
      이 F 0.39 이고, 게이트를 걸면 0.27 로 떨어진다. 보고용 맥락이지 관문이 아니다."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    g = pd.Series(groups).astype(str).values
    parts = []
    for ab in pd.unique(g):
        msk = (g == ab) & np.isfinite(y) & np.isfinite(x)
        if msk.sum() < 3: continue
        yy = y[msk]-y[msk].mean(); xx = x[msk]-x[msk].mean()
        if float(xx @ xx) > 1e-12: parts.append((yy, xx))
    m = len(parts); n = sum(len(a) for a, _ in parts)
    if m < 2 or n - 2*m <= 0: return dict(F=np.nan, p=np.nan, 메모="자유도가 없다")
    bc = sum(float(xx @ yy) for yy, xx in parts)/sum(float(xx @ xx) for _, xx in parts)
    rss_r = sum(float((yy-bc*xx) @ (yy-bc*xx)) for yy, xx in parts)
    rss_f = sum(float((yy-(xx @ yy/(xx @ xx))*xx) @ (yy-(xx @ yy/(xx @ xx))*xx))
                for yy, xx in parts)
    df1, df2 = m-1, n-2*m
    F = ((rss_r-rss_f)/df1)/(rss_f/df2)
    return dict(F=round(float(F), 3), df=(df1, df2), p=round(float(st_.f.sf(F, df1, df2)), 4))
print("7절-C 함수 준비 완료")
''')

code(r'''
# ── 7절-C 판정 ─────────────────────────────────────────────────────────────
IFACE_RESULT = None
_ok = ("ML_RESULT" in dir() and "IFACE_SRC" in dir() and IFACE_SRC
       and "IFACE" in dir() and len(IFACE) and "계면강도" in IFACE.columns)
if not _ok:
    print("7절-C 건너뜀 —", "7절이 안 돌았다" if "ML_RESULT" not in dir()
          else "쓸 수 있는 계면 지표가 없다 (3절-B·3절-C 를 보라)")
else:
    # ★ 이 검정만 포함 규칙을 **링커 3종 이상**으로 유지한다. 링커 2종이면 항체 안
    #   자유도가 0 이라 기울기의 오차를 못 재고, 정밀도 가중이 성립하지 않는다.
    U = USE.merge(IFACE[["항체", "계면강도"]], on="항체", how="left")
    U = U[U.계면강도.notna() & U[HC].notna()].reset_index(drop=True)
    _bb = U.groupby(GROUP).링커.nunique()
    U = U[U[GROUP].isin(_bb[_bb >= 3].index)].reset_index(drop=True)
    if U[GROUP].nunique() < 4:
        print(f"7절-C 건너뜀 — 쓸 수 있는 항체가 {U[GROUP].nunique()}개뿐이다 (4개 이상 필요)")
    else:
        Yi = make_target(U, HC, block=GROUP, higher_is_worse=HIGHER_IS_WORSE).values
        # ★ x 는 **로그 열림분율**이다. 선형을 쓰면 축의 곡률만으로 가짜 신호가 난다.
        _op = U[PRIMARY].values.astype(float)
        _fl = max(0.5/float(np.nanmedian(U.n프레임)) if "n프레임" in U else 1/300., 1e-4)
        _lx = np.log(np.clip(_op, _fl, None))
        SL = antibody_slopes(Yi, _lx, U[GROUP].values)
        _S  = U.drop_duplicates(GROUP).set_index(GROUP).계면강도
        Sv  = np.array([float(_S[a]) for a in SL.항체])
        k   = len(SL)
        print("="*76)
        print(f"★ 7절-C · 계면 상호작용 — 지수 = {IFACE_SRC}")
        print("="*76)
        print(f"  {IFACE_WHY}")
        print(f"  항체 {k}개 · 구성체 {len(U)} · 순열 {factorial(k) if factorial(k)<=5040 else '무작위'}가지")
        print("  모형:  y = α_항체 + β·log(열림) + γ·(계면강도 × log(열림))")
        print("  ★ 계면강도 **주효과는 못 넣는다** — 항체마다 상수라 중심화가 지운다.")
        print("    자료가 말할 수 있는 건 **항체별 기울기 b_j = β + γ·S_j** 뿐이고,")
        print("    γ 는 그 기울기 5개를 S 5개에 회귀해서만 나온다.")
        _L = sorted(U.길이.unique()); _below = [x for x in _L if x < 13]
        print(f"\n  링커 길이: {_L}")
        print("  ★ 전부 13 aa 이상 — Holliger 1993 의 diabody 임계(3~12 aa) **위**다."
              if not _below else f"  ★ {_below} 는 diabody 임계 아래 — 강제 이량체 영역이다.")
        if not _below:
            print("    즉 강제 이량체 영역이 아니다. 도메인 교환은 **부수 경로**일 것이고,")
            print("    그만큼 γ 가 작을 것을 각오해야 한다. 귀무가 나와도 놀랄 일이 아니다.")

        # ── ① y-무관 사전점검 — **관문은 여기뿐이다** ──────────────────────
        print("\n" + "─"*76)
        print("① y-무관 사전점검 — 실측을 보기 전에 통과해야 하는 것")
        print("─"*76)
        print("  ★ 실측(y) 을 쓰는 검정은 **보고용이지 관문이 아니다.** 관문은 y 를 안 쓰는")
        print("    것만이다 — 그래야 '결과를 보고 규칙을 바꿨다' 는 의심이 원천적으로 없다.\n")
        d_ = Sv - Sv.mean(); Sxx_ = float(d_ @ d_)
        lev = (d_**2)/Sxx_ if Sxx_ > 1e-12 else np.full(k, np.nan)
        _mx = SL.mean_x.values
        w_ = 1.0/SL.se.values**2
        chk = [dict(항목="S 변동계수", 값=round(float(np.std(Sv, ddof=1)/abs(np.mean(Sv))), 4)
                    if abs(np.mean(Sv)) > 1e-12 else np.nan, 판정=""),
               dict(항목="γ 유효정보 Σw·d²", 값=round(float((w_*d_*d_).sum()), 3),
                    판정=f"se(γ) = {1/sqrt(max(float((w_*d_*d_).sum()),1e-300)):.3f}"),
               dict(항목="최대 레버리지", 값=f"{SL.항체.values[int(np.argmax(lev))]} ({lev.max():.2f})",
                    판정="★ 한 항체가 S축을 지배한다 — 5점 회귀가 사실상 그 점 하나다"
                         if lev.max() > 0.6 else ""),
               dict(항목="ρ(S, 항체별 SE)", 값=round(float(st_.spearmanr(Sv, SL.se).statistic), 3),
                    판정="★ 정밀도가 S 를 따라간다 — 순열 교환가능성이 의심스럽다"
                         if abs(st_.spearmanr(Sv, SL.se).statistic) > 0.8 else ""),
               dict(항목="★ ρ(S, 항체별 평균 열림)",
                    값=round(float(st_.spearmanr(Sv, _mx).statistic), 3),
                    판정="★★ 작동점 교란이 크다 — Freedman–Lane 통제가 **필수**다"),
               # ★ 항체 **안** 열림분율 범위가 좁으면 기울기 자체가 안 잡힌다.
               #   그러면 상호작용은 볼 수도 없다. 모의: 범위를 1/3 로 줄이면
               #   같은 γ=2 에서 검출률 0.78 → 0.18.
               dict(항목="항체 안 log(열림) 범위 중앙",
                    값=round(float(np.median([np.ptp(_lx[U[GROUP].values == a])
                                              for a in SL.항체])), 2),
                    판정="★★ 링커가 열림분율을 거의 안 벌려 놓았다 — 기울기가 안 잡힌다"
                         if np.median([np.ptp(_lx[U[GROUP].values == a])
                                       for a in SL.항체]) < 0.7 else "")]
        display(pd.DataFrame(chk))
        print("  마지막 줄이 이 검정의 급소다. 센 계면일수록 덜 열리므로 S 와 작동점이")
        print("  기계적으로 얽혀 있고, y 가 x 에 조금이라도 곡선으로 붙으면 국소 기울기가")
        print("  작동점의 함수가 되어 **γ=0 인데도 기각된다.** 모의 거짓양성 0.33~0.41.")

        # ── ② 확증 검정 — **정확히 하나다** ────────────────────────────────
        print("\n" + "─"*76)
        print("② 확증 검정 — 정밀도가중 기울기회귀 + Freedman–Lane, 양측")
        print("─"*76)
        IT = interaction_test(SL, Sv, covariate=_mx, sided="two")
        SLp = SL.assign(계면강도=np.round(Sv, 3), 레버리지=IT["레버리지"])
        display(SLp[["항체", "k", "b", "se", "r_within", "mean_x", "계면강도", "레버리지"]].round(3))
        print(f"  γ̂ = {IT['gamma']}  ± {IT['se_gamma']}   (t = {IT['t']})")
        print(f"  Spearman(기울기, 계면강도) = {IT['spearman']}   ← 5점 순서. 이게 결과의 실체다")
        print(f"  {'정확' if IT['정확'] else '무작위'} 순열 {IT['순열수']}가지 · 통제 {IT['통제']}")
        print(f"  **양측 p = {IT['순열p']}**   (최소 가능 {IT['최소가능p']})")
        print("")
        print("  ── 부호를 어떻게 읽나 ───────────────────────────────────────")
        print("  γ < 0 : 조건부 예측대로. 같은 만큼 열려도 센 계면이 더 많은 응집으로 간다")
        print("          ([D] ∝ [M_open]²/Kd² 의 Kd 항).")
        print("  γ > 0 : 조건화가 샜다. 주변부 관계가 새어 들어온 것이고 그쪽은 방향이 반대다")
        print("          (약한 계면 → 더 열림 → 더 응집. Arndt/Plückthun 1998 계열).")
        if IT["순열p"] < ALPHA:
            print(f"\n  → **유의하다.** 부호는 {'조건부' if IT['gamma'] < 0 else '주변부(조건화가 샌)'} 방향.")
        else:
            print(f"\n  → 유의하지 않다." + (f"  ★ 단 최소 가능 p 가 {IT['최소가능p']} 라"
                  " 애초에 못 닿았을 수도 있다." if IT["최소가능p"] > ALPHA else ""))

        # ── ②-B 한 항체를 빼면 무너지는가 ─────────────────────────────────
        print("\n" + "─"*76)
        print("②-B 취약성 — 항체 하나를 빼면 결론이 바뀌는가")
        print("─"*76)
        loo = []
        for i in range(k):
            m_ = np.ones(k, bool); m_[i] = False
            if m_.sum() < 4: continue
            r_ = interaction_test(SL[m_].reset_index(drop=True), Sv[m_],
                                  covariate=_mx[m_], sided="two")
            loo.append(dict(뺀항체=SL.항체.values[i], 남은항체=int(m_.sum()),
                            gamma=r_["gamma"], 양측p=r_["순열p"]))
        if loo:
            LOO = pd.DataFrame(loo); display(LOO)
            if IT["순열p"] < ALPHA and (LOO.양측p > 0.20).any():
                print("  ★ 한 항체를 빼면 p 가 0.20 을 넘는다 — **취약한 결론**이다. 그렇게 보고하라.")
            elif IT["순열p"] < ALPHA:
                print("  → 어느 항체를 빼도 버틴다.")

        # ── ②-C ★ 반증 원장 ───────────────────────────────────────────────
        print("\n" + "─"*76)
        print("②-C 반증 원장 — 무관한 축들도 같은 순서를 만드는가")
        print("─"*76)
        print(f"  항체 {k}개면 가능한 순서가 {factorial(k)}가지다. |rho|=1 이 우연히 나올")
        print(f"  확률이 축 하나당 {2/max(factorial(k),1):.3f} — 축을 열 개 재면 대략 하나는 맞는다.")
        print("  ★ 무관한 축 3개 이상이 |rho| ≥ 0.9 를 찍으면, 계면강도의 p 가 얼마든 정보가 없다.")
        print("  ★ 그리고 **평균열림** 축을 반드시 본다 — 이건 AF2 도 BSA 도 필요 없는 공짜")
        print("    경쟁자이고, 곡률만 있어도 48.5% 의 귀무 자료에서 발화한다.\n")
        IFX = IFACE.set_index("항체")
        _rg = np.random.default_rng(12345)
        cand = {}
        for c in ("ipTM", "계면PAE", "계면_BSA", "계면_접촉밀도", "계면_소수성",
                  "기준점_흔들림", "CDRH3길이", "Fv순전하", "Fv소수성", "Fv길이"):
            if c in IFACE.columns and IFACE[c].notna().sum() >= k:
                cand[c] = np.array([float(IFX[c].get(a, np.nan)) for a in SL.항체])
        cand["★평균열림"] = _mx.copy()                       # 공짜 경쟁자
        cand["난수축1"] = _rg.normal(0, 1, k)
        cand["난수축2"] = _rg.normal(0, 1, k)
        cand["이름순"]  = np.argsort(np.argsort(SL.항체.values)).astype(float)
        rows = []
        for nm, v in cand.items():
            if not np.isfinite(v).all() or np.std(v) < 1e-12: continue
            # ★ 평균열림 자체를 볼 때는 자기 자신을 통제할 수 없다
            r_ = interaction_test(SL, v, covariate=(None if nm == "★평균열림" else _mx),
                                  sided="two")
            rows.append(dict(축=nm,
                             종류=("계면" if nm in ("ipTM","계면PAE","계면_BSA",
                                                   "계면_접촉밀도","계면_소수성")
                                   else "공짜경쟁자" if nm == "★평균열림"
                                   else "음성대조" if nm in ("난수축1","난수축2","이름순")
                                   else "경쟁설명"),
                             rho=r_["spearman"], 양측p=r_["순열p"]))
        if rows:
            FLG = pd.DataFrame(rows).sort_values("양측p").reset_index(drop=True)
            display(FLG)
            _o = FLG[(FLG.rho.abs() >= 0.9) & (~FLG.종류.isin(["계면"]))]
            _fr = FLG[FLG.축 == "★평균열림"]
            if len(_fr) and _fr.양측p.iloc[0] < ALPHA:
                print("  ★★ **평균열림만으로도 유의하다.** 그러면 계면강도가 설명하는 것이")
                print("     '계면' 인지 '항체마다 작동점이 다르다' 인지 가를 수 없다.")
                print("     후자는 ipTM 도 BSA 도 필요 없는 설명이다. 그렇게 보고하라.")
            if len(_o) >= 3:
                print("  ★★ 무관한 축 3개 이상이 같은 순서를 만들었다.")
                print("     **계면강도의 결과는 정보가 없다.** 항체를 늘리기 전에는 읽지 마라.")
            elif len(_o):
                print(f"     겹치는 축: {list(_o.축)} — 계면과 이것들을 못 가른다.")
            else:
                print("     무관한 축들은 조용하다.")

        # ── ③ 맥락 · 기울기가 애초에 다른가 (관문 아님) ────────────────────
        print("\n" + "─"*76)
        print("③ 맥락 — 항체별 기울기가 애초에 다른가 (보고용, **관문 아님**)")
        print("─"*76)
        HT = slope_heterogeneity(Yi, _lx, U[GROUP].values)
        print(f"  공통기울기 대 개별기울기 F = {HT['F']} · df = {HT.get('df')} · p = {HT['p']}")
        print("  ★ 이게 귀무여도 ② 를 버리지 않는다. 이 F 는 전방위(m−1 자유도)라")
        print("    방향을 아는 ② 보다 둔하다 — 모의에서 γ=1 일 때 0.39 vs 0.52 이고,")
        print("    게이트로 쓰면 0.27 로 떨어진다. 둔한 검정으로 예민한 검정을 막는 셈이다.")

        # ── ④ 신뢰도와 **검정력 천장** ────────────────────────────────────
        print("\n" + "─"*76)
        print("④ 검정력 — 측정오차가 만드는 **천장**")
        print("─"*76)
        print("  상호작용은 두 신뢰도의 **곱**으로 눌린다:  γ̂ = γ · λ_S · λ_O.")
        print("  (주효과는 하나만 들어온다. 상호작용이 훨씬 가혹하다.)\n")
        lamS = float(REL.set_index("지표").신뢰도.get(IFACE_SRC, np.nan)) if "REL" in dir() else np.nan
        lamO = float(RELY.get(PRIMARY, np.nan)) if "RELY" in dir() else np.nan
        print(f"  λ_S ({IFACE_SRC}) = {lamS if lamS==lamS else '미측정'}   "
              f"λ_O ({PRIMARY}) = {lamO if lamO==lamO else '미측정'}")
        if np.isfinite(lamS) and np.isfinite(lamO):
            print(f"  → **유효 γ = 실제 γ × {lamS*lamO:.2f}**")
            if lamS*lamO < 0.5:
                print("     ★★ 곱이 0.5 미만이다. 이 설계에서는 **어떤 효과크기로도**")
                print("       80% 검정력에 못 닿는다. 귀무가 나와도 '없다' 가 아니라")
                print("       '측정이 이만큼 시끄러우면 애초에 못 본다' 가 맞다.")
        rows = []
        rng = np.random.default_rng(0)
        for g_ in (0.0, 1.0, 2.0, 4.0):
            hit = 0; NS = 60
            for _ in range(NS):
                yy = np.empty(len(Yi))
                for t, ab in enumerate(SL.항체):
                    msk = (U[GROUP].values == ab)
                    xx = _lx[msk] - _lx[msk].mean()
                    yy[msk] = (1.0 + g_*(Sv[t]-Sv.mean())/max(Sv.std(ddof=1),1e-9))*xx \
                              + rng.normal(0, 1.0, msk.sum())
                    yy[msk] -= yy[msk].mean()
                try:
                    T_ = antibody_slopes(yy, _lx, U[GROUP].values)
                    hit += int(interaction_test(T_, Sv, covariate=_mx,
                                                sided="two")["순열p"] <= ALPHA)
                except Exception: pass
            rows.append(dict(효과크기=g_, 검출률=round(hit/NS, 3),
                             유효효과=round(g_*lamS*lamO, 2)
                             if (np.isfinite(lamS) and np.isfinite(lamO)) else np.nan))
        PW2 = pd.DataFrame(rows); display(PW2)
        print("  ★ '유효효과' 열로 읽어라. 명목 γ 가 아니라 감쇠된 값이 실제로 보이는 것이다.")
        print("  ★ 이 질문에서 가장 값싼 개선은 **항체를 늘리는 것**이다 —")
        print(f"    지금 {k}개면 순열 {factorial(k)}가지, 최소 p {1/factorial(k):.4f}.")
        print(f"    {k+1}개면 {factorial(k+1)}가지, {k+2}개면 {factorial(k+2)}가지다.")
        print("    (7절 ⑥ 의 주효과는 링커를 늘리는 쪽이었다. 다른 질문이라 답이 다르다.)")
        IFACE_RESULT = dict(지수=IFACE_SRC, 이유=IFACE_WHY, 기울기표=SLp,
                            상호작용=IT, 이질성=HT, 검정력=PW2,
                            반증원장=(FLG if rows else None), lamS=lamS, lamO=lamO)
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 7절-D · 배향 — VL-링커-VH 와 VH-링커-VL 이 **반대로** 가는가

v09 의 실측 결과를 보고 넣었다. 항체별 **길이 → HMW** 기울기가 이랬다.

| 배향 | 항체 | rho(길이, HMW) |
|---|---|---|
| LH | 실측4_B1 | **−0.80** |
| LH | 실측2_B2 | **−0.80** |
| LH | 실측3_B3 | **−0.87** |
| LH | 실측1_B4 | **−0.80** |
| HL | 실측9_B5 | **+0.80** |

LH 넷이 전부 n=4 가 낼 수 있는 거의 최대값으로 **같은 방향**이고, HL 하나만
**정확히 반대**다. 그런데 전체로 묶으면 27쌍 중 18쌍(0.69, p=0.12) 밖에 안 된다 —
HL 항체의 쌍들이 전부 '틀린' 쪽으로 세어지기 때문이다. **묶는 행위 자체가
신호를 지운다.** 그러면 묶지 말고 배향을 **조절변수**로 두고 물어야 한다.

### 왜 말이 되는가
scFv 는 대칭이 아니다. VH 의 C 말단과 VL 의 N 말단이 만나는 기하와, VL 의 C 말단과
VH 의 N 말단이 만나는 기하는 서로 다르다. 같은 길이의 링커가 한쪽에서는 남고
다른 쪽에서는 모자랄 수 있다 — 두 말단 사이의 직선거리가 배향마다 다르기 때문이다.
Fv 에서 그 거리는 대략 VH C말단–VL N말단이 **더 멀다**고 알려져 있다.

### 그래도 이건 **가설이지 결과가 아니다**
항체 5개 중 HL 이 **하나**다. 한쪽 수준에 표본이 하나면 그 기울기가 배향 때문인지
그 항체 때문인지 **원리적으로 못 가른다.** 아래 셀은 그 사실을 먼저 찍는다.

### 그런데 이미 가진 데이터에 **맞짝**이 있다
v09 출력에 블록 6·7·8 이 각각 **같은 도메인 · 같은 링커 · 배향만 반대**인 항체 쌍이다
(실측10_B6_HL ↔ 실측6_B6_LH, 실측11_B7_HL ↔ 실측7_B7_LH, 실측8_B8_HL ↔ 실측5_B8_LH).
링커가 1종뿐이라 기울기는 못 내지만, **배향 하나만 다른 직접 대조 3쌍**이다.
v09 는 이걸 블록으로 묶어 평균 내 버렸다 (그래서 안 보였다). v12 는 항체로 키를
잡으므로 살아 있다. 아래에서 그 3쌍을 짝지어 뺀다.
''')

code(r'''
# ── 7절-D · 배향 조절 시험 ─────────────────────────────────────────────────
from math import comb
_ok_d = ("USE" in dir() and len(USE) and HC and "배향" in USE.columns
         and "Y" in dir())
if not _ok_d:
    print("7절-D 건너뜀 —", "7절이 안 돌았다" if "Y" not in dir() else "배향 열이 없다")
else:
    print("="*76); print("★ 7절-D · 배향이 링커 효과의 **부호**를 뒤집는가")
    print("="*76)
    # ── ① 항체별 길이 기울기 + 배향 ────────────────────────────────────────
    SLd = antibody_slopes(Y, USE["길이"].values.astype(float), USE[GROUP].values)
    _od = USE.drop_duplicates(GROUP).set_index(GROUP)["배향"]
    SLd["배향"] = SLd.항체.map(_od)
    SLd["부호"] = np.where(SLd.b > 0, "+", "−")
    display(SLd[["항체", "배향", "k", "b", "se", "r_within", "부호"]].round(3))
    print("  b 는 '링커가 1 aa 길어질 때 (부호 뒤집은) 항체 안 로짓' 이다.")
    print("  ※ y 는 낮을수록 나쁜 값을 뒤집어 놓았으므로, b > 0 = 길수록 HMW 가 준다.\n")

    # ── ② 배향 수준별 표본 — 먼저 물을 수 있는 질문인지부터 본다 ───────────
    _cnt = SLd.배향.value_counts()
    print("  배향별 항체 수:", dict(_cnt))
    if len(_cnt) < 2 or _cnt.min() < 2:
        if len(_cnt) < 2:
            # 배향이 한 종류뿐 — 조절변수가 상수다. 검정이 **정의되지 않는다.**
            _why = (f"배향이 '{_cnt.index[0]}' 한 종류뿐이다 ({int(_cnt.iloc[0])}개)"
                    if len(_cnt) else "배향 정보가 없다")
            print(f"  ★ {_why}. 조절변수가 상수라 검정이 정의되지 않는다.")
        else:
            _thin = _cnt.idxmin()
            _why = f"'{_thin}' 배향 항체가 {int(_cnt.min())}개"
            print(f"  ★ {_why}뿐이다.")
            print("     그 항체의 기울기가 **배향 때문인지 그 항체 때문인지 못 가른다.**")
        print("     순열 p 는 내지 않는다 — 낼 수 있는 척하면 안 된다.")
        print("     위 기울기표의 부호만 읽고, 다음에 무엇을 만들지 정하는 데 써라.")
        ORIENT_RESULT = dict(기울기표=SLd, 검정=None, 이유=_why)
    else:
        # 양쪽에 2개 이상 있을 때만 검정한다. 통계량은 7절-C 와 **같은 것**이다 —
        # 항체 수준 라벨(여기서는 배향)만 재배치하는 정확 순열.
        _S = (SLd.배향.values == SLd.배향.value_counts().index[0]).astype(float)
        ITd = interaction_test(SLd, _S, sided="two", seed=11)
        print(f"  γ = {ITd['gamma']:+.4f}  ·  순열 p = {ITd['순열p']:.4f}  "
              f"({'정확' if ITd['정확'] else '무작위'} 순열 {ITd['순열수']}회, 양측, "
              f"최소가능 p = {ITd['최소가능p']:.4f})")
        print(f"  → {'**배향이 부호를 가른다.**' if ITd['순열p'] < ALPHA else '배향으로 갈린다고 못 한다.'}")
        ORIENT_RESULT = dict(기울기표=SLd, 검정=ITd, 이유="")

    # ── ③ 맞짝 대조 — 같은 도메인 · 같은 링커 · 배향만 반대 ────────────────
    #   기울기가 필요 없는 유일한 증거다. 링커 1종짜리 항체도 여기엔 들어온다.
    print("\n" + "─"*76)
    print("③ 맞짝 대조 — 같은 블록 · 같은 링커 · 배향만 반대인 항체 쌍")
    print("─"*76)
    _A = FEAT[~FEAT.합성].copy() if "합성" in FEAT.columns else FEAT.copy()
    _A = _A[_A[HC].notna()] if HC in _A.columns else _A.iloc[0:0]
    pr = []
    for (blk, lk), g in _A.groupby(["블록", "링커"]):
        o = g.drop_duplicates("배향")
        if o.배향.nunique() < 2: continue
        for a in o[o.배향 == "LH"].itertuples():
            for b in o[o.배향 == "HL"].itertuples():
                pr.append(dict(블록=blk, 링커=lk, LH항체=a.항체, HL항체=b.항체,
                               LH=getattr(a, HC, np.nan), HL=getattr(b, HC, np.nan)))
    PR = pd.DataFrame(pr)
    if len(PR):
        PR["차_HL빼기LH"] = (PR.HL - PR.LH).round(2)
        display(PR.round(2))
        _d = PR.차_HL빼기LH.dropna().values
        if len(_d) >= 2:
            # 부호검정 — n 이 2~3 이라 정규근사를 쓰면 안 된다. 정확 이항이다.
            _pos = int((_d > 0).sum()); _n = int((_d != 0).sum())
            _pb = 2*min(sum(comb(_n, i) for i in range(_pos, _n+1)),
                        sum(comb(_n, i) for i in range(0, _pos+1)))/2**_n
            print(f"  차의 중앙값 {np.median(_d):+.2f} %p · "
                  f"{_pos}/{_n} 쌍에서 HL 이 크다 · 정확 이항 p = {min(_pb,1.0):.3f}")
            print(f"  ★ n={_n} 쌍이면 전부 같은 방향이어도 p ≥ {2/2**_n:.3f} 다 —")
            print("     '유의하지 않다' 가 아니라 **'이 쌍 수로는 유의할 수 없다'** 이다.")
            print("     방향만 읽고, 쌍을 더 만들지 말지를 정하는 데 써라.")
    else:
        print("  배향이 양쪽 다 있는 (블록, 링커) 칸이 없다 — 대조할 짝이 없다.")
        print("  ※ 만들 수 있다면 이게 제일 싸다: 도메인·링커를 그대로 두고")
        print("     배향만 뒤집은 구성체 3~4개. 기울기가 필요 없어 링커 1종이면 된다.")
''')


# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 8절 · 대리모형 — "가상 데이터셋" 의 정직한 형태

### 먼저, 하면 안 되는 것부터 — 이유를 셋으로 적는다
라벨(순도)을 **만들어 내서** 학습 표본을 늘리는 것 — SMOTE, 잡음 주입, 생성모형으로
찍어낸 가짜 물성값. **이 노트북은 그걸 안 한다.** 왜인지 정확히:

1. **SMOTE 는 데이터를 더하지 않는다. 데이터 모양에 대한 *가정*을 더한다.**
   보간으로 만든 점은 정의상 "기술자와 실측이 매끄럽게 선형" 인 면 위에 정확히 놓인다.
   그 다음 모델이 "선형 관계가 있다" 고 보고하면, 그건 발견이 아니라
   **방금 손으로 그려 넣은 선을 다시 읽은 것**이다.
2. **교차검증이 통째로 깨진다.** 합성점의 부모 둘이 학습 폴드와 시험 폴드로 갈리면
   시험 폴드는 학습 폴드의 보간이다. CV 점수가 1 쪽으로 뜨는데 실제 일반화는 그대로다.
   26점에서는 부모 누출을 피할 방법이 아예 없다 (모든 점이 모든 점의 이웃이다).
3. **신뢰구간이 √(m/n) 만큼 좁아진다.** 26점을 260점으로 늘리면 구간이 √10 배 좁아진다.
   정보는 한 톨도 안 늘었는데. 결과는 **자신 있게 틀린 답**이다.
   이것이 "없는 상관을 제조한다" 의 정확한 기전이다.

대신 이 노트북이 하는 것: **부트스트랩으로 불확실도를 정직하게 적고** (6절-B, 8절),
**그룹 안에서 순열해 유의성을 내고** (7절), **합성 패널로 설계 해상도를 산다** (4-B절).

### 대신, 정직하게 데이터를 늘리는 자리는 따로 있다 — 2단 구조

```
Stage A   링커 서열 기술자  →  앙상블 기술자        라벨 불필요!
          학습 표본 = BioEmu 를 돌린 **모든** 구성체 (실측 26 + 합성 24 = 50)
          합성 패널이 여기를 키운다. 물성 측정이 필요 없으니까.

Stage B   앙상블 기술자  →  실측 순도               라벨 필요
          학습 표본 = 26. **여기는 안 늘어난다.** 늘릴 방법도 없다.

A ∘ B     임의의 링커 서열  →  예측 순도            in-silico 스크리닝
          BioEmu 를 안 돌리고 수천 개 링커를 훑는다.
```

**핵심**: Stage A 가 아무리 커져도 Stage B 의 n=26 은 그대로다. 합성 데이터는
**추정 정밀도가 아니라 설계 해상도**를 산다. A∘B 의 예측은 **후보 정렬용**이고
검증이 아니다 — 거기서 고른 링커는 결국 **실제로 만들어 재봐야** 한다.

### Stage A 가 왜 값진가
- 라벨이 필요 없으니 BioEmu 를 돌리는 만큼 계속 커진다.
- 보통의 교차검증이 그대로 통한다 (n=50, 조성별 leave-one-composition-out).
- **Stage A 가 안 맞으면 거기서 끝난다** — 링커 서열로 앙상블을 못 맞히면
  애초에 BioEmu 가 링커를 안 보는 것이고, 그건 6절-C 의 판정과 같은 이야기다.

### 유보 — 셀이 직접 숫자로 찍는다
합성 예측의 분산은 두 조각이다.

```
Var[예측] = (Stage A 몫)  서열→기술자 사상의 오차   ← 패널을 늘리면 준다
          + (Stage B 몫)  기술자→실측 기울기의 오차  ← n=26 이 정한다. 안 준다
```

셀이 두 몫을 부트스트랩으로 따로 재서 `StageB몫` 열에 찍는다.
그 값이 0.7 을 넘으면 **"이 표는 순위표이지 예측값이 아니다"** 를 스스로 인쇄한다.

한 문장으로: **자를 나노미터까지 정밀하게 만들어도, 그 자를 26개 물체로만
보정했다면 밀리미터 환산은 여전히 26개짜리다.**

**가상 데이터는 f(서열→기술자)를 산다. g(기술자→물성)는 실험실에서만 산다.**
''')

code(r'''
# ── 8절 · Stage A (라벨 불필요) → Stage B (라벨 필요) → in-silico 스크리닝 ──
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

_KD  = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
            I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
            Y=-1.3, V=4.2)                                      # Kyte-Doolittle
_HEL = dict(A=1.42, R=0.98, N=0.67, D=1.01, C=0.70, Q=1.11, E=1.51, G=0.57, H=1.00,
            I=1.08, L=1.21, K=1.16, M=1.45, F=1.13, P=0.57, S=0.77, T=0.83, W=1.08,
            Y=0.69, V=1.06)                                     # Chou-Fasman 나선성향

def linker_descriptors(seq):
    """링커 서열만으로 나오는 기술자. 구조가 필요 없다.
    ★ 조성은 전부 **분율**로 낸다 = 길이에 대해 불변. 길이는 별도 열 하나로만 들어간다.
      실제 패널에서는 둘이 완전히 얽혀 있고, 합성 요인패널이 그걸 푸는 유일한 장치다."""
    s = str(seq).upper(); n = max(len(s), 1)
    f = lambda aa: sum(s.count(a) for a in aa)/n
    return dict(길이=len(s), G분율=f("G"), S분율=f("S"), P분율=f("P"), A분율=f("A"),
                양전하분율=f("KR"), 음전하분율=f("DE"),
                순전하_잔기당=f("KR")-f("DE"), 전하밀도=f("KRDE"),
                극성분율=f("STNQY"), 소수성분율=f("AVLIMFW"),
                소수성=float(np.mean([_KD.get(a, 0.0) for a in s])) if s else 0.0,
                나선성향=float(np.mean([_HEL.get(a, 1.0) for a in s])) if s else 1.0)

SEQ_FEATS = ["길이", "G분율", "S분율", "P분율", "A분율", "양전하분율", "음전하분율",
             "순전하_잔기당", "전하밀도", "극성분율", "소수성분율", "소수성", "나선성향"]

if not RUN_ML:
    print("8절 건너뜀 (RUN_ML=False)")
else:
    SEQMAP = CONS.drop_duplicates(["항체", "링커"]).set_index(["항체", "링커"]).링커서열
    A = FEAT.copy()
    A["링커서열"] = pd.MultiIndex.from_arrays([A.항체, A.링커]).map(SEQMAP)
    A = A[A.링커서열.notna() & (A.링커서열.str.len() > 0)].reset_index(drop=True)
    A = pd.concat([A.drop(columns=[c for c in SEQ_FEATS if c in A.columns]),
                   pd.DataFrame([linker_descriptors(s) for s in A.링커서열],
                                index=A.index)], axis=1)
    print(f"Stage A 학습 표본: {len(A)}건 "
          f"(실측 {int((~A.합성).sum())} + 합성 {int(A.합성.sum())})")
    print("  ★ 라벨(순도)이 필요 없다. 그래서 BioEmu 를 돌린 만큼 계속 커진다.\n")

    # ── Stage A — 링커 서열 → 앙상블 기술자 ───────────────────────────────
    # 검증: 조성별 leave-one-composition-out. 안 본 조성으로 일반화되는가를 본다.
    #       (합성 패널이 없으면 조성이 하나뿐이라 무작위 5겹으로 내려간다.)
    grp = A.조성.fillna("실측").values
    logo = LeaveOneGroupOut() if len(set(grp)) >= 3 else None
    SA, rows = {}, []
    for tgt in ALLF:
        d = A[A[tgt].notna()]
        if len(d) < 8 or d[tgt].std(ddof=1) < 1e-12: continue
        X, y, g = d[SEQ_FEATS].values, d[tgt].values, d.조성.fillna("실측").values
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 24)))
        if logo is not None and len(set(g)) >= 3:
            yh = cross_val_predict(mdl, X, y, groups=g, cv=LeaveOneGroupOut())
            cvn = f"조성별 LOCO ({len(set(g))}겹)"
        else:
            from sklearn.model_selection import KFold
            yh = cross_val_predict(mdl, X, y, cv=KFold(5, shuffle=True, random_state=0))
            cvn = "무작위 5겹"
        r2 = 1 - ((y-yh)**2).sum()/max(((y-y.mean())**2).sum(), 1e-12)
        rows.append(dict(앙상블기술자=tgt, n=len(d), 검증=cvn,
                         교차검증R2=round(float(r2), 3),
                         Spearman=round(float(st_.spearmanr(y, yh)[0]), 3),
                         판정="쓸 만하다" if r2 > 0.3 else "★ 서열로 못 맞힌다"))
        SA[tgt] = mdl.fit(X, y)
    print("="*76); print("★ Stage A — 링커 서열로 앙상블 기술자를 맞힐 수 있나")
    print("="*76)
    display(pd.DataFrame(rows))
    print("  R² 가 전부 낮으면 링커 서열 → 앙상블 사상이 없다는 뜻이고,")
    print("  그것은 6절-C 의 '조성 채널 없음' 과 같은 이야기다. 그때는 8절을 접어라.")

    # ── Stage B — 앙상블 기술자 → 실측. n=26, **여기는 안 늘어난다** ──────
    # ★ 실측값이 **없는** 구성체가 있다. 그것은 결측이 아니라 **결과**다 —
    #   블록3 Whitlow218 처럼 수율이 무너져 물질이 없으면 HMW 를 잴 수가 없다.
    #   Stage A(라벨 불필요)에는 그대로 두고, 라벨이 필요한 Stage B 에서만 뺀다.
    #   빼면서 **몇 개를 왜 뺐는지** 반드시 찍는다 — 조용히 빠지면 표본 수가
    #   달라진 것을 아무도 모른다. (sklearn 은 NaN 타깃에 ValueError 를 낸다.)
    SB = None
    USE_LAB = USE[pd.to_numeric(USE[HC], errors="coerce").notna()] if HC else USE.iloc[0:0]
    _drop = len(USE) - len(USE_LAB)
    if _drop:
        print(f"\n  ※ 실측 {HC} 가 없는 구성체 {_drop}건을 Stage B 에서 뺀다:")
        for r in USE[~USE.index.isin(USE_LAB.index)].itertuples():
            print(f"     {getattr(r, GROUP):<18} {r.링커}")
        print("     결측이 아니라 **결과**다 — 물질이 안 나오면 물성을 못 잰다.")
        print("     6절-B 수율 파국 표와 같이 읽어라. Stage A 에는 그대로 남는다.")
    if HC and len(USE_LAB) >= 6:
        _c = [c for c in ALLF
              if USE_LAB[c].notna().all() and USE_LAB[c].std(ddof=1) > 1e-12]
        if _c:
            Yb = make_target(USE_LAB, HC, block=GROUP,
                             higher_is_worse=HIGHER_IS_WORSE).values
            _ok = np.isfinite(Yb)
            assert _ok.all(), "타깃에 아직 NaN 이 있다 — 필터가 샜다"
            SB = make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 3, 24))).fit(
                                   USE_LAB[_c].values, Yb)
            print(f"\nStage B 학습 표본: {len(USE_LAB)}건 — **합성 데이터로 늘릴 수 없다.**")
            print(f"  쓰는 기술자: {_c}")
            print(f"  성능은 7절의 LOBO 일치도로 이미 냈다 (여기서 다시 안 낸다 — "
                  f"같은 데이터로 두 번 재면 낙관적으로 나온다).")

    # ── A ∘ B — in-silico 스크리닝. **후보 정렬용이지 검증이 아니다** ──────
    if SA and SB is not None and set(_c) <= set(SA):
        SCREEN = {}
        for nm, m in MOTIF.items(): SCREEN[nm] = m
        SCREEN.update({"GSG": "GGSGS", "EK교대": "EKEKE", "PG": "PGPGP",
                       "TS": "TTSSG", "GD": "GGDGS", "QN": "QGNGS"})
        cand = [dict(조성=nm, 길이=L, 서열=(m*(L//len(m)+1))[:L])
                for nm, m in SCREEN.items() for L in range(8, 41, 2)]
        Cd = pd.DataFrame(cand)
        Xs = pd.DataFrame([linker_descriptors(s) for s in Cd.서열])[SEQ_FEATS].values
        for t in _c: Cd[t] = SA[t].predict(Xs)
        Cd["예측_상대순도"] = SB.predict(Cd[_c].values)
        Cd["돌려봤나"] = Cd.서열.isin(set(A.링커서열))

        # ── ★ 불확실성이 어디서 오는가 — 이 표가 8절의 진짜 결론이다 ─────────
        # 합성 예측의 분산은 두 조각이다:
        #   Stage A 몫 : 서열→기술자 사상의 오차. **패널을 늘리면 준다.**
        #   Stage B 몫 : 기술자→실측 기울기의 오차. **n=26 이 정한다. 안 준다.**
        # 두 몫을 부트스트랩으로 따로 잰다.
        rng = np.random.default_rng(0)
        nB = 300
        # Stage B 부트스트랩 — 항체 단위로 재표집 (구성체 단위로 하면 낙관적이다)
        # ★ 라벨이 있는 행만 재표집한다 (Stage B 는 라벨이 필요하다)
        gb = list(USE_LAB.groupby(GROUP).groups.values())
        predsB, _bfail = [], 0
        for _ in range(nB):
            pick = np.concatenate([np.asarray(gb[i]) for i in
                                   rng.integers(0, len(gb), len(gb))])
            d = USE_LAB.loc[pick]
            if d[_c].std(ddof=0).min() < 1e-12: continue
            yb = make_target(d, HC, block=GROUP, higher_is_worse=HIGHER_IS_WORSE).values
            _m = np.isfinite(yb)
            if _m.sum() < 6: continue
            try:
                mB = make_pipeline(StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(
                                       d[_c].values[_m], yb[_m])
                predsB.append(mB.predict(Cd[_c].values))
            except Exception as e:
                # ★ 조용히 삼키지 않는다. 몇 번 실패했는지 아래에서 찍는다 —
                #   전부 실패하면 구간이 비는데 그걸 모르고 읽으면 안 된다.
                _bfail += 1
                if _bfail == 1:
                    print(f"  ※ Stage B 부트스트랩 실패 1회: {type(e).__name__}: {e}")
        if _bfail:
            print(f"  ※ Stage B 부트스트랩 {_bfail}/{nB}회 실패 — 구간이 그만큼 얇다.")
        # Stage A 부트스트랩 — 구성체 단위 재표집
        predsA = []
        for _ in range(nB):
            d = A.sample(len(A), replace=True, random_state=int(rng.integers(1 << 30)))
            try:
                xs = pd.DataFrame([linker_descriptors(t) for t in Cd.서열])[SEQ_FEATS].values
                cols = {}
                for t in _c:
                    dd = d[d[t].notna()]
                    if len(dd) < 8: raise ValueError
                    mA = make_pipeline(StandardScaler(),
                                       RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(
                                           dd[SEQ_FEATS].values, dd[t].values)
                    cols[t] = mA.predict(xs)
                predsA.append(SB.predict(pd.DataFrame(cols)[_c].values))
            except Exception: pass
        vB = np.var(np.array(predsB), axis=0) if len(predsB) > 20 else np.zeros(len(Cd))
        vA = np.var(np.array(predsA), axis=0) if len(predsA) > 20 else np.zeros(len(Cd))
        tot = vA + vB
        Cd["StageB몫"] = np.where(tot > 1e-12, vB/np.maximum(tot, 1e-12), np.nan)
        if len(predsB) > 20:
            P_ = np.array(predsB)
            Cd["lo95"] = np.percentile(P_, 2.5, axis=0)
            Cd["hi95"] = np.percentile(P_, 97.5, axis=0)
            Cd["구간이0을가로지름"] = (Cd.lo95 < 0) & (Cd.hi95 > 0)

        print("\n" + "="*76)
        print("★ A∘B — 안 돌려본 링커까지 훑는다 (후보 정렬용, **검증 아님**)")
        print("="*76)
        print(f"  후보 {len(Cd)}개 (조성 {Cd.조성.nunique()} × 길이 {Cd.길이.nunique()}) · "
              f"그중 실제로 BioEmu 를 돌린 것 {int(Cd.돌려봤나.sum())}개\n")
        _sb = float(np.nanmedian(Cd.StageB몫)) if Cd.StageB몫.notna().any() else np.nan
        print("  ── 불확실성이 어디서 오는가 ─────────────────────────────────")
        print(f"  Stage B 몫 중앙값 = **{_sb:.2f}**   (나머지가 Stage A 몫)")
        print("  Stage A 몫은 패널을 늘리면 준다. Stage B 몫은 **n=26 이 정하고 안 준다.**")
        if np.isfinite(_sb) and _sb > 0.7:
            print(f"  → 불확실성의 {_sb:.0%} 가 라벨 26개에서 온다.")
            print("     **이 표는 순위표이지 예측값이 아니다.** 숫자 자체를 인용하지 마라.")
            print("     자를 나노미터까지 정밀하게 만들어도, 그 자를 26개로만 보정했으면")
            print("     밀리미터 환산은 여전히 26개짜리다. 합성 데이터로는 못 고친다.")
        if "구간이0을가로지름" in Cd:
            print(f"  95% 구간이 0 을 가로지르는 후보: "
                  f"{int(Cd['구간이0을가로지름'].sum())}/{len(Cd)}"
                  f"  ← 많으면 순위만 읽어라")
        print()
        _show = ["조성", "길이", "서열", "예측_상대순도"] + \
                (["lo95", "hi95"] if "lo95" in Cd else []) + ["StageB몫", "돌려봤나"]
        display(Cd.nlargest(12, "예측_상대순도")[_show].round(3))
        Cd.to_csv(f"{OUT}/insilico_screen.csv", index=False, encoding="utf-8-sig")
        print(f"  전체 저장 → {OUT}/insilico_screen.csv")
        print("  다음 할 일: 상위 후보 3~5개를 **실제로 만들어 재라.** 그게 검증이다.")
    else:
        print("\nA∘B 건너뜀 — Stage A 나 Stage B 중 하나가 안 섰다")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 9절 · 그림 — 판정에 들어가는 것만

v11 은 그림이 여섯 장이었고 그중 RMSF 5단 패널과 PC1 영화는 **판정에 안 들어갔다.**
숫자로 쓰이는 것만 남긴다.

| 그림 | 답하는 질문 |
|---|---|
| `fig_landscape` | BioEmu 논문 Fig 2A 형식. 링커별 VH-VL 배향 지형 · 겹쳐보기 · 누적곡선 |
| `fig_channel` | 4-B절 요인패널. 길이축 대 조성축 — **채널 판정의 그림판** |
| `fig_ml` | 순열 귀무분포 · 증분 · 검정력 곡선 — **7절의 그림판** |

★ Fig 2A 용어 주의. 논문의 "coverage" 는 *참조 구조 중 몇 %가 표본의 0.1% 이상으로
샘플링됐나* 이고, 우리 곡선은 *한 구성체의 프레임 중 몇 %가 기준점에서 t 이내인가* 다.
**다른 양이다.** 그래서 축 이름을 `누적 비율` 로 쓰고 coverage 라 부르지 않는다.

★ 지형의 **깊이는 읽지 마라.** BioEmu 에는 에너지 함수가 없어 재가중이 불가능하고,
본문 결과는 10,000 샘플로 낸 것이다 (우리는 150). 읽을 것은 골짜기의 **위치**,
링커 간 **이동**, 그리고 자연 2σ 타원과의 관계다.
''')

code(r'''
# ── 9절 · 그림 ─────────────────────────────────────────────────────────────
plt = setup_font()
from matplotlib.patches import Ellipse
# 앞 셀을 안 돌리고 이 셀만 다시 돌려도 죽지 않게 한다
# ★ 빈 표를 만들 때 **열까지** 만든다. 열 없는 빈 DataFrame 은 `FEAT.합성` 에서
#   'DataFrame 에 합성 속성이 없다' 로 죽는데, 그 문장은 원인(앞 절을 안 돌렸다)과
#   아무 상관 없어 보인다. 방어선이 방어선 때문에 죽는 꼴이다.
FEAT = FEAT if "FEAT" in dir() else pd.DataFrame(
    {c: pd.Series(dtype=t) for c, t in (("블록", "object"), ("항체", "object"),
                                        ("링커", "object"), ("길이", "float"),
                                        ("조성", "object"), ("설계길이", "float"),
                                        ("합성", "bool"))})
USE  = USE  if "USE"  in dir() else FEAT[~FEAT.합성]   # FEAT 다음이어야 한다
SIG = {f"Δ{k}": v for k, v in AB_SD.items()}
LAND_X, LAND_Y = "ΔHL", "Δdc"       # ΔHC1 · ΔHC2 등으로 바꿔도 된다

# ── 그림 1 · BioEmu 논문 Fig 2A "Domain motions" 형식 ──────────────────────
#   그들: 기준구조 A 까지의 RMSD  vs  기준구조 B 까지의 RMSD
#   우리: ΔHL (VH-VL 꼬임각)     vs  Δdc (도메인 중심간 거리)
#   기준점(ABB2)은 정의상 원점 (0,0) 이라 그들의 '기준 구조 점' 자리에 그대로 온다.
E = E if ("E" in dir() and E is not None) else pd.DataFrame()
_blk = (sorted(USE.블록.unique())[0] if len(USE)
        else sorted(E.블록.dropna().astype(str).unique())[0] if len(E) else None)
_G = (E[(E.블록 == _blk) & (E.생성기 == GEN) & (~E.합성.astype(bool))].copy()
      if _blk is not None else pd.DataFrame())
_lk = ([l for l in CONS[(CONS.블록 == _blk) & (~CONS.합성)].링커 if l in set(_G.링커)]
       if len(_G) else [])
if len(_G) and _lk:
    cols = dict(zip(_lk, plt.cm.tab10.colors))
    xr = (min(float(_G[LAND_X].quantile(.005)), -3*SIG[LAND_X]),
          max(float(_G[LAND_X].quantile(.995)),  3*SIG[LAND_X]))
    yr = (min(float(_G[LAND_Y].quantile(.005)), -3*SIG[LAND_Y]),
          max(float(_G[LAND_Y].quantile(.995)),  3*SIG[LAND_Y]))
    n = len(_lk)
    fig = plt.figure(figsize=(3.1*n + 6.8, 3.6))
    gs = fig.add_gridspec(1, n+2, wspace=.36)
    # ★ N=150 에서 26×26 격자는 칸당 0.22 개다. −RT ln(H/N) 로 칸 하나짜리를 그리면
    #   −0.593·ln(1/150) = **2.97 kcal/mol** — 즉 색눈금 상한(3)이 통째로
    #   **표본 하나짜리 잡음 바닥**이다. 그 그림의 '3 kcal/mol 등고선' 은 물리가 아니다.
    #   (논문은 모든 지형을 10,000 표본으로 그린다. 우리는 그 1.5% 다.)
    #   → 격자를 14로 줄이고, 칸 수가 MIN_CNT 미만이면 아예 안 그린다.
    #     그리고 눈금을 kcal/mol 로 부르지 않는다 — BioEmu 에는 에너지 함수가 없어
    #     재가중이 불가능하므로 이건 **보정 안 된 상대 로그밀도**다.
    NB, MIN_CNT = 14, 5
    for j, lk in enumerate(_lk):                        # ① 링커별 2D 지형
        g = _G[_G.링커 == lk]; a = fig.add_subplot(gs[0, j])
        H, xe, ye = np.histogram2d(g[LAND_X], g[LAND_Y], bins=NB, range=[xr, yr])
        with np.errstate(divide="ignore"):
            Fm = -np.log(H.T/max(H.sum(), 1))           # 상대 로그밀도 (kT 단위)
        Fm[H.T < MIN_CNT] = np.nan                      # ← 잡음 바닥은 안 그린다
        Fm[~np.isfinite(Fm)] = np.nan
        if np.isfinite(Fm).any(): Fm -= np.nanmin(Fm)
        im = a.imshow(Fm, origin="lower", aspect="auto", cmap="turbo",
                      extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=3)
        a.add_patch(Ellipse((0, 0), 4*SIG[LAND_X], 4*SIG[LAND_Y], fill=False,
                            ec="white", lw=1.4, ls="--"))      # SAbDab 자연 2σ
        a.plot(0, 0, marker="*", ms=17, c="white", mec="k", mew=.9, zorder=6)
        a.set_title(lk, fontsize=10)
        a.set_xlabel(f"{LAND_X} (°)", fontsize=8)
        if j == 0: a.set_ylabel(f"{LAND_Y} (Å)", fontsize=8)
        if j == n-1: fig.colorbar(im, ax=a, label="−ln P (상대, 보정 안 됨)")
    a = fig.add_subplot(gs[0, n])                       # ② 겹쳐 보기
    for lk in _lk:
        g = _G[_G.링커 == lk]
        H, xe, ye = np.histogram2d(g[LAND_X], g[LAND_Y], bins=NB, range=[xr, yr])
        a.contour((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, H.T,
                  levels=[H.max()*.25, H.max()*.6], colors=[cols[lk]], linewidths=1.3)
        a.plot(*np.median(g[[LAND_X, LAND_Y]].values, axis=0), "o", c=cols[lk],
               ms=7, mec="k", mew=.6, label=lk)
    a.add_patch(Ellipse((0, 0), 4*SIG[LAND_X], 4*SIG[LAND_Y], fill=False,
                        ec="gray", ls="--"))
    a.plot(0, 0, marker="*", ms=15, c="k", zorder=6)
    a.set_xlim(*xr); a.set_ylim(*yr); a.legend(fontsize=7)
    a.set_xlabel(LAND_X, fontsize=8); a.set_ylabel(LAND_Y, fontsize=8)
    a.set_title("겹쳐 보기 — 겹치면 축퇴다\n(별 = ABB2 기준점, 점선 = 자연 2σ)", fontsize=9)
    a = fig.add_subplot(gs[0, n+1])                     # ③ 누적 곡선
    xs = np.linspace(0, float(_G.OCD6.quantile(.99)), 160)
    for lk in _lk:
        v = _G[_G.링커 == lk].OCD6.values
        a.plot(xs, [(v <= t).mean() for t in xs], lw=1.6, c=cols[lk], label=lk)
    a.axvline(6.6, c="gray", ls=":", lw=1)
    a.text(6.7, .06, "무관한 두 항체", fontsize=7, color="gray", rotation=90)
    a.set_xlabel("OCD6 (자연 σ 6축 합)", fontsize=8)
    a.set_ylabel("누적 비율", fontsize=8)      # ★ 논문의 coverage 와 다른 양이다
    a.set_ylim(0, 1.02); a.legend(fontsize=7, loc="lower right")
    a.set_title("기준점에서 얼마나 벗어나 있나", fontsize=9)
    fig.suptitle(f"블록 {_blk} · {GEN} · 링커별 VH-VL 배향 지형 "
                 f"(BioEmu 논문 Fig 2A 형식)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_landscape.png", bbox_inches="tight")
    plt.show()
    print(f"  ※ 격자 {NB}×{NB} · 칸 수 {MIN_CNT} 미만은 안 그렸다. N=150 에서 칸 하나짜리는")
    print("     −ln(1/150) = 5.0 (kT) 로 색눈금 상한을 그냥 채운다 — 물리가 아니라 잡음이다.")
    print("  ※ 깊이는 읽지 마라. BioEmu 에는 에너지 함수가 없어 재가중이 불가능하고,")
    print("     논문은 모든 지형을 10,000 표본으로 그린다 (우리는 그 1.5%).")
    print("     읽을 것은 골짜기의 **위치**와 링커 간 **이동**이다.")

# ── 그림 2 · 채널 판정 — 길이축 vs 조성축 ──────────────────────────────────
SYN = FEAT[FEAT.합성]
if len(SYN) >= 8 and SYN.조성.nunique() >= 2:
    ks = [c for c in ALLF if SYN[c].notna().sum() >= 6]
    fig, ax = plt.subplots(1, max(len(ks), 1), figsize=(4.3*max(len(ks), 1), 3.6),
                           squeeze=False)
    for j, c in enumerate(ks):
        a = ax[0][j]
        for nm, g in SYN.groupby("조성"):
            g = g.sort_values("설계길이")
            a.plot(g.설계길이, g[c], "o-", ms=6, label=nm, lw=1.4)
        a.set_xlabel("링커 길이 (aa)", fontsize=9); a.set_ylabel(c, fontsize=9)
        a.set_title(f"{c}\n세로로 벌어지면 조성 채널, 가로로만 기울면 길이뿐", fontsize=9)
        if j == 0: a.legend(fontsize=7, ncol=2)
    fig.suptitle("합성 요인패널 — 같은 길이에서 조성만 바꿨을 때 (조성 6 × 길이 4)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_channel.png", bbox_inches="tight")
    plt.show()
    print("  같은 x(길이)에서 선들이 **세로로 벌어지면** 조성 채널이 있는 것이다.")
    print("  선들이 겹친 채 같이 기울기만 하면 BioEmu 는 길이만 본 것이다.")
else:
    print("그림 2 건너뜀 — 합성 패널이 없다")

# ── 그림 3 · ML 판정 ───────────────────────────────────────────────────────
if "ML_RESULT" in dir():
    R = ML_RESULT
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    a = ax[0]                                           # 순열 귀무분포
    if R.get("확증") is None:      # 확증 특징이 상수여서 검정이 정의되지 않았다
        a.text(.5, .5, f"① 확증 없음\n{PRIMARY} 가 상수", ha="center", va="center",
               fontsize=10, transform=a.transAxes)
        a.set_xticks([]); a.set_yticks([])
    else:
        nl = R["확증"]["_null"]; nl = nl[np.isfinite(nl)]
        a.hist(nl, bins=25, color="lightsteelblue", ec="w")
        a.axvline(R["확증"]["_obs"], c="crimson", lw=2,
                  label=f"관측 {R['확증']['일치도']:.2f}")
        a.axvline(0.5, c="k", ls=":", lw=1, label="우연 0.50")
        a.set_xlabel("항체 안 쌍 일치도", fontsize=9); a.set_ylabel("순열 횟수", fontsize=9)
        a.legend(fontsize=7)
        a.set_title(f"① 확증 — {PRIMARY}\n순열 p = {R['확증']['순열p']:.4f}", fontsize=10)

    a = ax[1]                                           # 증분
    I = R["증분"]
    if len(I):
        y = np.arange(len(I))
        a.barh(y-.18, I.길이만, .34, color="gray", label="길이만")
        a.barh(y+.18, I.길이_앙상블, .34, color="steelblue", label="길이 + 앙상블")
        a.axvline(0.5, c="k", ls=":", lw=1)
        a.set_yticks(y); a.set_yticklabels(I.앙상블특징, fontsize=8); a.invert_yaxis()
        a.set_xlabel("LOGO 항체 안 쌍 일치도", fontsize=9); a.legend(fontsize=7)
        a.set_xlim(0, 1)
        for k, r in I.iterrows():
            a.text(max(r.길이만, r.길이_앙상블)+.02, k, f"p={r.순열p:.3f}",
                   va="center", fontsize=7)
    a.set_title("② 증분 — 앙상블이 길이 위에 더하나", fontsize=10)

    a = ax[2]                                           # 검정력
    P = R["검정력"]
    a.plot(P.효과크기, P.검출률, "o-", c="crimson", ms=5)
    a.axhline(0.8, ls="--", c="gray", lw=1); a.axhline(ALPHA, ls=":", c="gray", lw=1)
    if np.isfinite(R["MDE"]):
        a.axvline(R["MDE"], ls="--", c="steelblue", lw=1)
        a.text(R["MDE"], .1, f" MDE ≈ {R['MDE']:.1f}", fontsize=8, color="steelblue")
    a.set_xlabel("심은 효과크기 β (특징 1 SD 당 타깃 σ)", fontsize=9)
    a.set_ylabel("검출률", fontsize=9); a.set_ylim(0, 1.03)
    a.set_title(f"③ 검정력 — 거짓양성 {P.검출률.iloc[0]:.3f}\n"
                f"이보다 작은 효과는 못 본다", fontsize=10)
    fig.suptitle(f"7절 판정 · 항체(독립 단위) {USE[GROUP].nunique()}개 × "
                 f"구성체 {len(USE)}개 · 순열 {N_PERM}회", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml.png", bbox_inches="tight")
    plt.show()
else:
    print("그림 3 건너뜀 — 7절이 안 돌았다")

# ── 앙상블을 눈으로 — MODEL 여러 개짜리 PDB (PyMOL/ChimeraX 용) ────────────
# v11 의 py3Dmol 격자 렌더링은 뺐다. 파일만 남기면 PyMOL 에서 더 잘 본다.
VIEW_N = 20
for _, q in CONS[~CONS.합성].groupby("항체").head(1).iterrows():
    for lk in CONS[CONS.항체 == q.항체].링커.unique()[:4]:
        fr = sorted(glob.glob(f"{OUT}/{q.항체}/flow/{lk}__{GEN}_s*.pdb"))[:VIEW_N]
        if len(fr) < 2: continue
        ref = ca_xyz(fr[0])[:int(q.b1)]                 # D1 기준으로 겹친다
        out = f"{OUT}/{q.항체}/{lk}__{GEN}_x{len(fr)}.pdb"
        with open(out, "w") as fo:
            for i, p in enumerate(fr, 1):
                L = open(p).readlines()
                ca = ca_xyz(p)[:int(q.b1)]
                if len(ca) != len(ref): continue
                cm, cr = ca.mean(0), ref.mean(0); Rk = kabsch(ca-cm, ref-cr)
                fo.write(f"MODEL {i:>8}\n")
                for l in L:
                    if l.startswith("ATOM"):
                        x = (np.array([float(l[30:38]), float(l[38:46]),
                                       float(l[46:54])])-cm) @ Rk + cr
                        l = l[:30] + f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}" + l[54:]
                    fo.write(l)
                fo.write("ENDMDL\n")
            fo.write("END\n")
        print(f"  {lk:<12} MODEL {len(fr)}개 → {os.path.basename(out)}")
    break
print("\n※ 백본만 있다 (N·CA·C·CB·O). 곁사슬은 BioEmu 출력에 없다 —")
print("  bioemu.sidechain_relax 로 복원할 수 있지만 그건 사후 패킹이지 샘플링이 본 것이 아니다.")
print(f"{elapsed()} 9절 끝")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 10절 · CALVADOS — **두 사슬**을 실제로 만나게 한다

### 왜 여기까지 와야 했나
BioEmu 는 단일 사슬만 다룬다. 지금까지 잰 것은 전부 *"단량체 하나가 어떤 모양인가"*
이고, HMW 는 *"둘이 만나서 붙었나"* 다. 그 사이는 모델이 아니라 **가설**이었다.

그리고 단일사슬 기술자로는 무엇을 정의해도 결국 **링커 길이의 함수로 환원된다.**
`링커신장도`(말단간/윤곽길이)가 제일 셌지만 길이와 rho **−0.99** 였다 — 말단간은
도메인 기하가 정하니 거의 상수라서 신장도 ≈ 상수/길이가 된다. 증분 +0.000.
**두 사슬을 만나게 해야 새 축이 나온다.**

### 물리 하나 — "만날 확률" 은 거의 안 변한다
Smoluchowski 충돌 속도 `k_D ∝ R·D` 인데 `D ∝ 1/R` 이라 **R 이 상쇄된다.**
링커가 펴져서 분자가 커져도 **만나는 빈도 자체는 거의 그대로**다.
바뀌는 것은 **만나서 붙을 확률**이다. 그래서 겨냥할 양은

- **B22** — 만남당 알짜 인력. 음수면 응집 경향. SLS·CG-MALS 로 **실측도 되는 양**이다.
- **끈끈한 면 노출** — 링커가 그 면을 덮고 있나 비켜 있나.

후자가 "링커가 자유분방하면 표면이 더 자주 드러난다" 의 직접 대응물이다.

### 모형 설정 — 왜 Fv 를 통째로 강체로 두나
CALVADOS 3 은 접힌 도메인을 탄성 네트워크로 묶고 IDR 만 유연하게 둔다.
여기서는 **VH+VL 을 하나의 강체**로 둔다 (ABodyBuilder2 의 짝지은 구조 그대로).
이유: CALVADOS 의 λ 척도는 IDP 용이라 **VH-VL 같은 특이적 계면을 못 만든다.**
도메인을 따로 풀어 놓으면 짝지음 여부를 비특이적 끈끈함이 정해 버려서 못 믿는다.
Fv 를 강체로 고정하면 묻는 것이 깨끗해진다 — **링커가 바깥 표면을 얼마나 가리고,
두 분자가 얼마나 붙는가.**

### ★ 이것으로 증명되지 않는 것
CALVADOS 는 **비특이적·콜로이드적 회합**만 본다. VH(1)-VL(2) **도메인 교환(diabody)
은 재현되지 않는다** — 모양 상보성이 만드는 특이적 결합이기 때문이다.
B22 가 나와도 "도메인 교환 가설을 검정했다" 고 쓰면 안 된다. 다른 기전이다.

### 표본 문제를 어떻게 피하나
B22 와 노출은 **라벨이 필요 없다.** 합성 링커 패널 32개에 그대로 돌릴 수 있고,
조성 × 길이 직교 설계에서 "조성이 B22 를 움직이는가" 를 n=5 제약 없이 물을 수 있다.
그래서 **항체내몫을 먼저 본다** — 0.3 미만이면 또 항체를 재는 것이고,
그때는 라벨 검정까지 갈 필요가 없다. 그 판정에 라벨은 안 쓴다.
''')

code(r'''
# ── 10절-A · CALVADOS 설치와 입력 만들기 ──────────────────────────────────
if not RUN_CALVADOS:
    print("10절 건너뜀 (RUN_CALVADOS=False)")
    print("  켜기 전에 읽어라: 구성체당 수십 분이고 GPU 가 사실상 필요하다.")
    print("  먼저 실측 4구성체만 돌려 **항체내몫**을 보라 — 0.3 미만이면 거기서 접는다.")
    CALV = pd.DataFrame()
else:
    import importlib
    HAVE_CALV = False
    try:
        importlib.import_module("calvados"); HAVE_CALV = True
    except Exception:
        # ★ calvados 는 **PyPI 에 없다.** GitHub 저장소에서 받아야 한다
        #   (직접 확인했다: `pip install calvados` → No matching distribution).
        print("openmm + calvados 설치 중 (몇 분)…", flush=True)
        sh(["pip", "-q", "install", "openmm", "mdtraj"])
        sh(["pip", "-q", "install",
            "git+https://github.com/KULL-Centre/CALVADOS.git"])
        try:
            importlib.import_module("calvados"); HAVE_CALV = True
        except Exception as e:
            print(f"★ calvados 를 못 올렸다: {type(e).__name__}: {e}")
    if HAVE_CALV:
        import calvados
        print(f"calvados {getattr(calvados, '__version__', '?')} 준비")

    # ── 잔기별 파라미터는 **패키지에서 읽는다** ─────────────────────────────
    #   λ 끈끈함 척도를 여기에 손으로 적으면 안 된다. 그건 지어내는 것이다.
    #   패키지가 들고 있는 표를 그대로 쓰고, 못 찾으면 그 사실을 말하고 멈춘다.
    LAM, RES_CSV = None, None
    if HAVE_CALV:
        # ★ `calvados.data` 에는 __init__.py 가 없다 → **네임스페이스 패키지**라
        #   `calvados.data.__file__` 이 None 이다. 거기서 dirname 을 부르면
        #   TypeError: expected str … not NoneType 가 난다 (실측에서 그랬다).
        #   그래서 `calvados.__file__` 을 기준으로 찾는다.
        import calvados as _cv, glob as _glob
        _root = os.path.dirname(os.path.abspath(_cv.__file__))
        _cand = (sorted(_glob.glob(f"{_root}/data/residues*.csv"))
                 + sorted(_glob.glob(f"{_root}/data/*.csv"))
                 + sorted(_glob.glob(f"{_root}/../residues*.csv"))
                 + sorted(_glob.glob(f"{OUT}/residues*.csv")))
        for _f in _cand:
            try:
                _t = pd.read_csv(_f)
            except Exception:
                continue
            _c = [c for c in _t.columns if str(c).lower() in ("lambdas", "lambda")]
            _o = [c for c in _t.columns if str(c).lower() in ("one", "onelettercode")]
            if _c and _o:
                LAM, RES_CSV = dict(zip(_t[_o[0]].astype(str),
                                        _t[_c[0]].astype(float))), os.path.abspath(_f)
                print(f"  λ 척도: {os.path.basename(_f)} ({len(LAM)} 잔기) → {RES_CSV}")
                print(f"    열: {list(_t.columns)}")
                break
        if LAM is None:
            print(f"  ★ residues.csv 를 못 찾았다. 뒤진 곳: {_root}/data/ 등 {len(_cand)}개")
            print(f"    아래 한 줄이면 받아진다 — {OUT} 에 두면 다음 실행부터 자동으로 읽는다:")
            print(f"      !wget -qO {OUT}/residues.csv https://raw.githubusercontent.com"
                  f"/KULL-Centre/CALVADOS/main/calvados/data/residues.csv")
    if HAVE_CALV and LAM is None:
        print("  ★★ λ 끈끈함 척도를 패키지에서 못 찾았다.")
        print("     **여기에 손으로 적어 넣지 마라** — 지어낸 척도로 낸 B22 는 숫자일 뿐이다.")
        print("     calvados 버전을 확인하고 데이터 파일 경로를 맞춰라. 10절은 멈춘다.")
        HAVE_CALV = False

    CALV_IN = f"{OUT}/calvados"; os.makedirs(CALV_IN, exist_ok=True)
    print(f"  작업 폴더 {CALV_IN}")
    print(f"  조건: 이온강도 {CALV_IONIC} M · pH {CALV_PH} · {CALV_TEMP} K · "
          f"사슬 {CALV_NCHAIN}개 · {CALV_STEPS:,} 스텝")
    print("  ※ 이온강도를 **실제 제형에 맞춰라.** Debye 스크리닝이 명시적이라")
    print("    Whitlow218 의 K/E 가 여기서 실제로 작동한다.")
''')

code(r'''
# ── 10절-A2 · 어떤 앙상블 구조를 넣을까 — **좌표로 고른다** ────────────────
# ★ ABangle 로 고르면 안 된다. 무성 오염되면 dc 가 실제 열림과 무관해진다
#   (실측 rho(도메인Rg, |Δdc|) = +0.06). 진짜 열린 프레임이 '붙음' 으로,
#   안 열린 프레임이 '열림' 으로 들어가 있었다. **좌표에서 직접 고른다.**
def pick_conformers(ab, lk, n_closed=None, n_open=None, q=None):
    """구성체 하나에서 닫힌 대표 구조와 열린 대표 구조를 고른다.

    고르는 기준은 `도메인Rg` — CA 좌표에서 바로 나오므로 번호매김과 무관하다.
    각 무리 안에서는 **골고루 퍼지게** 뽑는다 (분위수 등간격). 연속 프레임이
    서로 복제인 경우가 많아서(실측에서 76% 가 |Δdc| 차이 0.5 미만) 이웃을
    연달아 집으면 같은 구조를 여러 번 넣게 된다.
    """
    n_closed = CALV_N_CLOSED if n_closed is None else n_closed
    n_open   = CALV_N_OPEN   if n_open   is None else n_open
    q        = CALV_OPEN_Q   if q        is None else q
    if not len(G):
        return {}, "geom.csv 가 없다 — 5절-B 를 먼저 돌려라"
    g = G[(G.항체 == ab) & (G.링커 == lk)].dropna(subset=["도메인Rg"])
    if len(g) < (n_closed + n_open) * 2:
        return {}, f"프레임이 {len(g)}개뿐이다"
    g = g.sort_values("도메인Rg").reset_index(drop=True)
    cut = float(g.도메인Rg.quantile(q))
    op = g[g.도메인Rg >= cut]
    cl = g[g.도메인Rg <= float(g.도메인Rg.quantile(0.5))]
    def _spread(d, k):
        if len(d) <= k: return d
        idx = np.linspace(0, len(d) - 1, k).round().astype(int)
        return d.iloc[np.unique(idx)]
    sel = {"닫힘": _spread(cl, n_closed), "열림": _spread(op, n_open)}
    out = {}
    for kind, d in sel.items():
        rows = []
        for r in d.itertuples():
            p = f"{OUT}/{ab}/flow/{r.태그}.pdb"
            if os.path.isfile(p):
                rows.append(dict(종류=kind, 태그=r.태그, 경로=p,
                                 도메인Rg=round(float(r.도메인Rg), 2),
                                 말단간=round(float(getattr(r, "말단간", np.nan)), 1)))
        out[kind] = rows
    n_ok = sum(len(v) for v in out.values())
    return out, ("" if n_ok else "PDB 파일을 못 찾았다 — 4절 출력이 있나 확인하라")

if RUN_CALVADOS:
    CONF = {}
    print("="*76); print("★ CALVADOS 에 넣을 구조 고르기 — 닫힌 것과 열린 것 둘 다")
    print("="*76)
    print("  기전이 '대부분은 얌전하고 일부만 열려서 붙는다' 이므로")
    print("  닫힌 것만 재도 틀리고 열린 것만 재도 틀리다. **둘 다 재서 섞는다.**\n")
    for r in CONS[~CONS.합성].drop_duplicates(["항체", "링커"]).itertuples():
        sel, why = pick_conformers(r.항체, r.링커)
        if why:
            print(f"  {r.항체:<16} {r.링커:<12} 건너뜀 — {why}"); continue
        CONF[(r.항체, r.링커)] = sel
        _c = sel.get("닫힘", []); _o = sel.get("열림", [])
        print(f"  {r.항체:<16} {r.링커:<12} 닫힘 {len(_c)}개 "
              f"(Rg {min([x['도메인Rg'] for x in _c], default=0):.1f}"
              f"~{max([x['도메인Rg'] for x in _c], default=0):.1f}) · "
              f"열림 {len(_o)}개 "
              f"(Rg {min([x['도메인Rg'] for x in _o], default=0):.1f}"
              f"~{max([x['도메인Rg'] for x in _o], default=0):.1f})")
    if CONF:
        _gap = [max([x["도메인Rg"] for x in v.get("열림", [])], default=np.nan) -
                min([x["도메인Rg"] for x in v.get("닫힘", [])], default=np.nan)
                for v in CONF.values()]
        _gap = [x for x in _gap if np.isfinite(x)]
        print(f"\n  닫힘↔열림 도메인Rg 차이 중앙값 {np.median(_gap):.1f} Å")
        print("  ★ 이 차이가 작으면 두 무리가 사실 같은 것이다 — 그때는 섞을 이유가 없고")
        print("    B22 를 한 번만 재면 된다. 2~3 Å 미만이면 그렇게 보고하라.")
        print(f"\n  총 시뮬레이션 {sum(len(v) for s in CONF.values() for v in s.values())}건 "
              f"× 구성체당 수십 분. **먼저 한 구성체만 돌려 보라.**")
else:
    CONF = {}
''')

code(r'''
# ── 10절-B · 끈끈한 패치 정의 — 무엇이 '붙을 면' 인가 ──────────────────────
# ABodyBuilder2 기준 구조에서 **표면에 드러난 소수성 잔기**를 고른다.
# 이것이 링커가 가릴 수도 있고 안 가릴 수도 있는 면이고, 다른 분자가 붙을 면이다.
# ★ 항체마다 다르다 — 그래서 '링커 × 도메인' 상호작용이 여기로 들어온다.
_KD_HYDRO = set("AVLIMFWYC")

def sticky_patch(ref_pdb, q=None):
    """기준 구조 → 끈끈한 패치 잔기 색인 (사슬 안 0부터).

    표면 노출(상대 SASA)이 중앙값 위이고 소수성인 잔기 중 상위 q 분위.
    곁사슬이 있는 **ABB2 구조**에서 뽑는다 — BioEmu 백본으로는 SASA 가 안 나온다.
    """
    q = CALV_PATCH_Q if q is None else q
    st = first_model(ref_pdb)
    if st is None:
        return {}, "기준 구조를 못 읽었다"
    from Bio.PDB.SASA import ShrakeRupley
    ShrakeRupley().compute(st, level="R")
    out = {}
    for ch in st:
        rs = [r for r in ch if "CA" in r]
        if len(rs) < 40:
            continue
        sasa = np.array([getattr(r, "sasa", np.nan) for r in rs], float)
        aa = [_3to1_g.get(r.get_resname(), "X") for r in rs]
        hydro = np.array([a in _KD_HYDRO for a in aa])
        expo = sasa > np.nanmedian(sasa)
        cand = np.where(hydro & expo)[0]
        if len(cand) == 0:
            out[ch.id] = np.array([], int); continue
        k = max(1, int(round(q * len(rs))))
        out[ch.id] = cand[np.argsort(-sasa[cand])][:k]
    return out, ""

_3to1_g = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
           "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
           "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"}

if RUN_CALVADOS:
    PATCH = {}
    for ab in CONS.항체.unique():
        _m = sorted(glob.glob(f"{REFD(ab)}/ref_m[0-9].pdb")) or [f"{REFD(ab)}/ref.pdb"]
        _m = [x for x in _m if os.path.isfile(x)]
        if not _m:
            print(f"  {ab:<22} 기준 구조 없음 — 건너뜀"); continue
        pt, why = sticky_patch(_m[0])
        if why:
            print(f"  {ab:<22} {why}"); continue
        PATCH[ab] = pt
        print(f"  {ab:<22} 끈끈한 패치 " +
              " · ".join(f"{c}:{len(v)}" for c, v in pt.items()))
    if PATCH:
        _n = [sum(len(v) for v in p.values()) for p in PATCH.values()]
        print(f"\n  패치 크기 {min(_n)}~{max(_n)} 잔기 · 항체마다 다르다.")
        print("  ★ 이 차이가 '같은 링커인데 도메인마다 다르다' 의 한 경로다 —")
        print("    링커가 가릴 면의 크기와 위치가 애초에 다르다.")
else:
    PATCH = {}
''')

code(r'''
# ── 10절-C · 관측량 — 궤적이 있으면 재고, 없으면 무엇이 필요한지 말한다 ────
# 분석 핵심은 tools/fvcalv.py 와 **같은 코드**다 (자체 시험 20/20 통과).
KB_CALV = 0.008314462618      # kJ/mol/K

def b22_from_pmf(r, w, T=None, r_core=None):
    """PMF → 2차 비리얼 계수.  B22 = −2π∫[exp(−W/kT)−1] r² dr   (nm³)

    ★ 꼬리에서 0 으로 맞춘다 — 안 하면 상수 치우침이 적분을 통째로 끌고 간다.
    ★ r[0]~core 구간은 격자 위에서 이미 f=−1 로 적분된다. 거기에 해석항을 또
      더하면 **두 배가 된다** (자체 시험이 딱딱한 구에서 정확히 그걸 잡았다).
    """
    T = CALV_TEMP if T is None else T
    r = np.asarray(r, float); w = np.asarray(w, float)
    if r.shape != w.shape or r.ndim != 1 or len(r) < 3:
        raise ValueError("r 과 w 는 길이 3 이상의 같은 1차원 배열이어야 한다")
    o = np.argsort(r); r, w = r[o], w[o]
    if np.any(r <= 0): raise ValueError("r 은 양수여야 한다")
    w = w - float(np.median(w[-max(1, len(r)//20):]))
    core = float(r[0]) if r_core is None else float(r_core)
    f = np.where(r < core, -1.0, np.exp(-w/(KB_CALV*T)) - 1.0)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(-2.0*np.pi*(tz(f*r**2, r) - (r[0]**3)/3.0))

def b22_reduced(b22_nm3, sigma):
    """딱딱한 구 대비. B2_HS = (2π/3)σ³ (σ = 접촉 지름).
    링커가 길면 σ 가 커져 B22 도 커지므로 **크기 효과를 빼고** 볼 때 쓴다."""
    hs = (2.0*np.pi/3.0)*float(sigma)**3
    return float(b22_nm3/hs) if hs > 0 else np.nan

def sticky_exposure(pos, patch_idx, linker_idx, cut=0.8):
    """패치 비드 중 링커에 **안 가려진** 비율 (프레임별)."""
    pos = np.asarray(pos, float)
    if pos.ndim != 3 or pos.shape[2] != 3: raise ValueError("pos 는 (프레임,비드,3)")
    patch_idx = np.asarray(patch_idx, int); linker_idx = np.asarray(linker_idx, int)
    if len(patch_idx) == 0: raise ValueError("패치가 비었다")
    if len(linker_idx) == 0: return np.ones(len(pos))
    out = np.empty(len(pos))
    for t, p in enumerate(pos):
        d = np.linalg.norm(p[patch_idx][:, None, :] - p[linker_idx][None, :, :], axis=-1)
        out[t] = float((d.min(axis=1) > cut).mean())
    return out

def linker_sweep(pos, linker_idx):
    """링커가 훑는 부피 대리값 — 위치 공분산 행렬식^(1/2) 의 프레임 평균.
    '자유분방함' 을 크기가 아니라 **퍼짐**으로 잰다 (신장도는 길이의 변장이었다)."""
    pos = np.asarray(pos, float); linker_idx = np.asarray(linker_idx, int)
    if len(linker_idx) < 3: return np.nan
    return float(np.mean([np.sqrt(max(float(np.linalg.det(
        np.cov((p[linker_idx]-p[linker_idx].mean(0)).T))), 0.0)) for p in pos]))

def b22_mixture(f_open, b_cc, b_oo, b_co=None):
    """닫힌 종과 열린 종이 섞여 있을 때의 **겉보기** B22.

        B22_app = (1−f)²·B_cc + 2f(1−f)·B_co + f²·B_oo

    ★ 선형 혼합이 아니다. B22 는 **쌍** 상호작용이라 조성에 2차로 들어간다.
      (1−f)·B_cc + f·B_oo 로 쓰면 교차항이 통째로 빠져 틀린다.
    b_co 를 모르면 **구간**을 준다 — 지어낸 한 값보다 정직하다.
    """
    f = float(f_open)
    if not (0.0 <= f <= 1.0): raise ValueError(f"f_open 은 0~1: {f}")
    b_cc, b_oo = float(b_cc), float(b_oo)
    if b_co is None:
        lo, hi = min(b_cc, b_oo), max(b_cc, b_oo)
        return ((1-f)**2*b_cc + 2*f*(1-f)*lo + f**2*b_oo,
                (1-f)**2*b_cc + 2*f*(1-f)*hi + f**2*b_oo)
    return float((1-f)**2*b_cc + 2*f*(1-f)*float(b_co) + f**2*b_oo)

def open_excess(f_open, b_cc, b_oo):
    """열림이 더한 몫만 — f²·(B_oo − B_cc). 교차항을 B_cc 로 두는 **보수적** 읽기다.
    '열린 것끼리 만나야만 추가 인력이 생긴다' 는 가정이고, 부호와 크기를 보는 데 쓴다."""
    f = float(f_open)
    if not (0.0 <= f <= 1.0): raise ValueError(f"f_open 은 0~1: {f}")
    return float(f**2*(float(b_oo) - float(b_cc)))

def conformer_noise(v):
    """같은 구성체에서 **다른 프레임**을 넣었을 때의 흔들림 (SD).
    구성체 간 차이가 이것보다 작으면 우리가 재는 것은 **프레임 선택 잡음**이다.
    씨앗 반복과 같은 역할 — 분모가 없으면 분자를 못 읽는다."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan

def within_share(values, groups):
    """항체내몫 — **라벨 검정보다 먼저** 본다. 0.3 미만이면 링커가 아니라 항체를 잰다."""
    v = np.asarray(values, float); g = np.asarray(groups)
    m = np.isfinite(v); v, g = v[m], g[m]
    if len(v) < 4 or len(np.unique(g)) < 2: return np.nan
    mu = [v[g == k].mean() for k in np.unique(g)]
    wi = np.concatenate([v[g == k] - v[g == k].mean() for k in np.unique(g)])
    sb, sw = float(np.std(mu, ddof=1)), float(np.std(wi, ddof=1))
    return float(sw**2/(sb**2 + sw**2)) if (sb**2 + sw**2) > 1e-30 else np.nan

print("10절 관측량 함수 준비 완료 (fvcalv.py 와 같은 코드 · 자체 시험 31/31)")
''')

code(r'''
# ── 10절-A3 · ★ 실제로 돌린다 ─────────────────────────────────────────────
# CALVADOS 저장소(KULL-Centre/CALVADOS)의 two_IDR_MDP 예제와 **같은 API** 다.
# 핵심 하나: `domains.yaml` 의 항목을 **중첩 리스트**로 주면 그 잔기들이 하나의
# 강체로 묶인다 (build.py:get_ssdomains → check_ssdomain(req_both=True) 이
#   i, j 가 **같은** ssdom 안에 있을 때만 구속을 건다).
#     scFv: [[[1,b1],[b2+1,N]]]   ← VH+VL 을 한 덩어리로 (기본. 배향이 고정된다)
#     scFv: [[1,b1],[b2+1,N]]     ← 따로 (짝지음을 비특이적 λ 가 정한다 — 못 믿는다)
CALV_RIGID_FV = True     # False 로 두면 VH·VL 을 따로 구속한다 (권장 안 함)
CALV_BOX_NM   = 30.0     # 상자 한 변. 두 사슬이 대부분 떨어져 있어야 g(r) 꼬리가 1 이 된다
CALV_NSAVE    = 1000     # 저장 간격(스텝)
CALV_NFRAMES  = 1000     # 저장 프레임 수 → steps = NSAVE × NFRAMES

def _write_calv_inputs(d, name, seq, b1, b2, src_pdb):
    """FASTA · PDB · domains.yaml 을 CALVADOS 형식으로 쓴다. 잔기 번호는 1-based."""
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{name}.fasta", "w") as f:
        f.write(f">{name}\n{seq}\n")
    # CALVADOS 는 CA 만 본다. BioEmu 프레임(백본만)도 그대로 쓸 수 있다.
    st = first_model(src_pdb)
    if st is None:
        raise ValueError(f"구조를 못 읽었다: {src_pdb}")
    cas = [r["CA"] for c in st for r in c if "CA" in r]
    if len(cas) != len(seq):
        raise ValueError(f"CA {len(cas)}개 ≠ 서열 {len(seq)}잔기 — 경계가 안 맞는다")
    with open(f"{d}/{name}.pdb", "w") as f:
        for i, a in enumerate(cas, 1):
            x, y, z = a.coord
            f.write(f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    dom = ([[[1, int(b1)], [int(b2) + 1, len(seq)]]] if CALV_RIGID_FV
           else [[1, int(b1)], [int(b2) + 1, len(seq)]])
    with open(f"{d}/domains.yaml", "w") as f:
        f.write(json.dumps({name: dom}))       # JSON 은 YAML 의 부분집합이다
    return dom

def run_calvados_pair(name, seq, b1, b2, src_pdb, work, res_csv, platform="CPU"):
    """같은 분자 **두 사슬**을 한 상자에 넣고 돌린 뒤 COM 궤적을 남긴다."""
    from calvados.cfg import Config, Components
    inp = f"{work}/input"; _write_calv_inputs(inp, name, seq, b1, b2, src_pdb)
    cfg = Config(sysname=name, box=[CALV_BOX_NM]*3, temp=CALV_TEMP,
                 ionic=CALV_IONIC, pH=CALV_PH, topol="random",
                 wfreq=CALV_NSAVE, steps=CALV_NSAVE*CALV_NFRAMES, runtime=0,
                 platform=platform, restart="checkpoint", frestart="restart.chk",
                 verbose=False)
    ana = (f"from calvados.analysis import calc_com_traj\n"
           f"calc_com_traj(path='{work}', sysname='{name}', output_path='{work}/data',"
           f" residues_file='{res_csv}', chainid_dict=dict({name}=(0,1)), start=100)\n")
    cfg.write(work, name="config.yaml", analyses=ana)
    comp = Components(molecule_type="protein", nmol=2, restraint=True,
                      charge_termini="both", fresidues=res_csv,
                      ffasta=f"{inp}/{name}.fasta", fdomains=f"{inp}/domains.yaml",
                      pdb_folder=inp, restraint_type="harmonic", use_com=True,
                      colabfold=1, k_harmonic=700.)
    comp.add(name=name)
    comp.write(work, name="components.yaml")
    # ★ sh() 는 리스트를 받고 cwd 를 못 바꾼다. run.py 가 상대경로를 쓰므로
    #   subprocess 로 직접 돌리면서 cwd 를 준다.
    env = dict(os.environ, MPLBACKEND="Agg"); env.pop("PYTHONPATH", None)
    pr = subprocess.run([sys.executable, f"{work}/run.py", "--path", work],
                        capture_output=True, text=True, cwd=work, env=env)
    if pr.returncode != 0:
        raise RuntimeError((pr.stderr or pr.stdout or "")[-600:])
    return pr

def b22_from_rdf(r, g, r_core_from_grid=True):
    """g(r) → B22.  B22 = −2π∫[g(r)−1]r²dr   (nm³)

    g = exp(−W/kT) 이므로 (g−1) 이 곧 마이어 f 함수다. PMF 를 거치지 않아서
    배제 영역(g=0)에서 log 가 발산하지 않는다 — 궤적에서 낼 때는 이 쪽이 안전하다.
    상류 예제와 같은 식이되, 격자 **밖**(0~r[0]) 완전배제분 +2π·r[0]³/3 을 더한다."""
    r = np.asarray(r, float); g = np.asarray(g, float)
    if r.shape != g.shape or r.ndim != 1 or len(r) < 3:
        raise ValueError("r 과 g 는 길이 3 이상의 같은 1차원 배열이어야 한다")
    o = np.argsort(r); r, g = r[o], g[o]
    if np.any(r <= 0): raise ValueError("r 은 양수여야 한다")
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out = -2.0*np.pi*tz((g - 1.0)*r**2, r)
    return float(out + (2.0*np.pi*(r[0]**3)/3.0 if r_core_from_grid else 0.0))

if RUN_CALVADOS and HAVE_CALV and CONF:
    import mdtraj as md
    try:
        import torch as _t; _gpu = bool(_t.cuda.is_available())
    except Exception:
        _gpu = bool(shutil.which("nvidia-smi"))
    _plat = "CUDA" if _gpu else "CPU"
    print(f"플랫폼 {_plat} · 상자 {CALV_BOX_NM} nm · "
          f"{CALV_NSAVE*CALV_NFRAMES:,} 스텝 · Fv 강체 {CALV_RIGID_FV}")
    if _plat == "CPU":
        print("  ★ GPU 가 없다. 구성체 하나에 몇 시간 걸린다 — 먼저 한 건만 돌려 보라.")
    rows, t0, _nrun = [], time.time(), 0
    _items = [(k, v) for k, v in CONF.items()
              if CALV_ONLY is None or k == tuple(CALV_ONLY)]
    _tot = sum(len(v) for _, s_ in _items for v in s_.values())
    print(f"  이번에 돌릴 것 {min(_tot, CALV_MAX_RUNS or _tot)}/{_tot}건"
          + (f" (CALV_MAX_RUNS={CALV_MAX_RUNS})" if CALV_MAX_RUNS else "")
          + (f" · CALV_ONLY={CALV_ONLY}" if CALV_ONLY else ""))
    print("  ※ 이어달리기 된다 — 끝난 구조는 json 으로 남아 다음 실행에서 건너뛴다.")
    for (ab, lk), sel in _items:
        r0 = CONS[(CONS.항체 == ab) & (CONS.링커 == lk)].iloc[0]
        for kind, lst in sel.items():
            b22s, expos, sweeps = [], [], []
            for c in lst:
                tag = f"{safe(ab)}__{safe(lk)}__{kind}__{c['태그']}"
                w = f"/content/_calv/{tag}"
                if os.path.isfile(f"{OUT}/calvados/{tag}.json"):
                    j = json.load(open(f"{OUT}/calvados/{tag}.json"))
                    b22s.append(j["B22"]); expos.append(j.get("노출", np.nan))
                    sweeps.append(j.get("훑는부피", np.nan)); continue
                if CALV_MAX_RUNS is not None and _nrun >= CALV_MAX_RUNS:
                    break
                if not budget(3600, f"CALVADOS {tag}"): break   # 남은 예산 확인
                _nrun += 1
                try:
                    os.makedirs(w, exist_ok=True)
                    run_calvados_pair(safe(ab)+"_"+safe(lk), r0.seq,
                                      int(r0.b1), int(r0.b2), c["경로"], w, RES_CSV,
                                      platform=_plat)
                    t = md.load(f"{w}/data/{safe(ab)}_{safe(lk)}_com_traj.dcd",
                                top=f"{w}/data/{safe(ab)}_{safe(lk)}_com_top.pdb")
                    rr, gg = md.compute_rdf(t, pairs=[[0, 1]],
                                            r_range=(0.5, CALV_BOX_NM/2),
                                            bin_width=0.1)
                    b = b22_from_rdf(rr, gg)
                    b22s.append(b)
                    os.makedirs(f"{OUT}/calvados", exist_ok=True)
                    json.dump(dict(B22=b, 태그=c["태그"], 종류=kind,
                                   도메인Rg=c["도메인Rg"]),
                              open(f"{OUT}/calvados/{tag}.json", "w"))
                    print(f"  {ab:<16} {lk:<12} {kind} {c['태그'][-8:]:>9} "
                          f"B22 {b:>9.1f} nm³  [{(time.time()-t0)/60:.0f}분]", flush=True)
                except Exception as e:
                    print(f"  ★ {tag} 실패: {type(e).__name__}: {str(e)[:120]}")
                finally:
                    shutil.rmtree(w, ignore_errors=True)
            if b22s:
                rows.append(dict(항체=ab, 링커=lk, 종류=kind,
                                 B22=round(float(np.mean(b22s)), 2),
                                 B22_SD=round(float(np.std(b22s, ddof=1)), 2)
                                 if len(b22s) > 1 else 0.0, n구조=len(b22s)))
    _done = len(glob.glob(f"{OUT}/calvados/*.json"))
    print(f"\n  누적 완료 구조 {_done}/{_tot}건 · 이번 실행 {_nrun}건 "
          f"· {(time.time()-t0)/60:.0f}분")
    if _nrun and CALV_MAX_RUNS is not None and _done < _tot:
        print(f"  ★ 한 건당 약 {(time.time()-t0)/60/max(_nrun,1):.1f}분 걸렸다. "
              f"남은 {_tot-_done}건이면 약 {(time.time()-t0)/60/max(_nrun,1)*(_tot-_done)/60:.1f}시간.")
        print("    시간을 보고 CALV_MAX_RUNS 를 올리거나 None 으로 풀어라.")
    if rows:
        W = pd.DataFrame(rows)
        P = W.pivot_table(index=["항체", "링커"], columns="종류",
                          values=["B22", "B22_SD"]).reset_index()
        P.columns = ["_".join([c for c in t if c]).strip("_") for t in P.columns]
        P = P.rename(columns={"B22_닫힘": "B22_닫힘", "B22_열림": "B22_열림",
                              "B22_SD_열림": "B22_열림_SD"})
        # 좌표 기반 열림분율을 붙인다 (ABangle 을 안 쓴다)
        if "열림_Rg" in FEAT.columns:
            P = P.merge(FEAT[["항체", "링커", "열림_Rg"]].rename(
                columns={"열림_Rg": "열림분율"}), on=["항체", "링커"], how="left")
        P.to_csv(f"{OUT}/calvados.csv", index=False, encoding="utf-8-sig")
        print(f"\n저장 {len(P)}행 → {OUT}/calvados.csv")
        display(P)
    else:
        print("  ★ 성공한 시뮬레이션이 없다. 위 실패 메시지를 보라.")
elif RUN_CALVADOS:
    print("10절-A3 건너뜀 —",
          "calvados 를 못 올렸다" if not HAVE_CALV else "고른 구조가 없다 (10절-A2 확인)")
''')

code(r'''
# ── 10절-D · 결과 수집과 **y 를 안 보는 판정** ────────────────────────────
CALV_CSV = f"{OUT}/calvados.csv"
CALV = pd.read_csv(CALV_CSV) if (REUSE_CSV and os.path.isfile(CALV_CSV)) else pd.DataFrame()
if not RUN_CALVADOS and not len(CALV):
    print("10절 결과 없음 — RUN_CALVADOS 를 켜고 돌려라.")
    print("\n무엇이 나오게 되나:")
    print("  구성체마다  B22(nm³) · B22_무차원 · 끈끈한면_노출 · 링커_훑는부피 · 분자간접촉")
    print("  전부 **라벨이 필요 없다** → 합성 패널 32개에 그대로 돌아간다.")
elif len(CALV):
    display(CALV)
    _need = [c for c in ("항체", "링커") if c not in CALV.columns]
    assert not _need, f"calvados.csv 에 {_need} 열이 없다"
    OBSV = [c for c in ("B22", "B22_무차원", "끈끈한면_노출", "링커_훑는부피",
                        "분자간접촉") if c in CALV.columns]
    print("="*76)
    print("★ 10절 판정 ① — 라벨을 보기 **전에**: 이게 링커를 재나 항체를 재나")
    print("="*76)
    _v = []
    for c in OBSV:
        sh_ = within_share(CALV[c].values, CALV.항체.values)
        _v.append(dict(관측량=c, 항체내몫=round(sh_, 2),
                       판정=("★ 항체를 재고 있다 — 여기서 접어라" if sh_ < 0.3 else
                             "치우쳐 있다" if sh_ < 0.5 else "쓸 만하다")))
    VC = pd.DataFrame(_v); display(VC)
    print("  각도 특징들이 0.20~0.29 여서 실패했다. 그 판정에는 라벨이 필요 없었고,")
    print("  지금도 필요 없다. 0.3 미만이면 **라벨 검정까지 가지 마라.**")

    # ── 형태를 섞는다 — 기전을 수로 쓴다 ──────────────────────────────────
    if {"B22_닫힘", "B22_열림", "열림분율"} <= set(CALV.columns):
        print("\n" + "="*76)
        print("★ 형태 혼합 — B22_app = (1−f)²B_cc + 2f(1−f)B_co + f²B_oo")
        print("="*76)
        print("  선형 혼합이 아니다. B22 는 쌍 상호작용이라 조성에 **2차**로 들어간다.")
        print("  기전이 '열린 것끼리 붙는다' 면 지배항은 f²·B_oo 다.\n")
        _m = []
        for r in CALV.itertuples():
            f = float(getattr(r, "열림분율", np.nan))
            cc, oo = float(r.B22_닫힘), float(r.B22_열림)
            if not all(np.isfinite([f, cc, oo])): continue
            co = float(getattr(r, "B22_교차", np.nan))
            if np.isfinite(co):
                app, band = b22_mixture(f, cc, oo, co), ""
            else:
                lo, hi = b22_mixture(f, cc, oo)
                app, band = (lo + hi)/2, f"[{lo:.0f}, {hi:.0f}]"
            _m.append(dict(항체=r.항체, 링커=r.링커, 열림분율=round(f, 3),
                           B22_닫힘=round(cc, 1), B22_열림=round(oo, 1),
                           B22_겉보기=round(app, 1), 교차항구간=band,
                           열림초과=round(open_excess(f, cc, oo), 1)))
        MIX = pd.DataFrame(_m)
        if len(MIX):
            display(MIX)
            if (MIX.교차항구간 != "").any():
                print("  ※ 교차항 B_co 를 안 쟀다 — 구간으로 보고한다. 정확한 값이 필요하면")
                print("    닫힌 것 하나 + 열린 것 하나를 **한 상자**에 넣고 따로 재라.")
            _key = pd.MultiIndex.from_arrays([CALV.항체, CALV.링커])
            for c in ("B22_겉보기", "열림초과"):
                CALV[c] = MIX.set_index(["항체", "링커"])[c].reindex(_key).values
            OBSV = OBSV + ["B22_겉보기", "열림초과"]
            display(pd.DataFrame([
                dict(관측량=c, 항체내몫=round(within_share(CALV[c].values,
                                                     CALV.항체.values), 2))
                for c in ("B22_겉보기", "열림초과")]))

    # ── 프레임 선택 잡음 — 구성체 간 차이가 이것보다 커야 읽을 수 있다 ──────
    if "B22_열림_SD" in CALV.columns and "B22_열림" in CALV.columns:
        print("\n" + "="*76)
        print("★ 프레임 선택 잡음 — 어느 구조를 넣었나가 답을 정한다")
        print("="*76)
        _fn = float(np.nanmedian(CALV.B22_열림_SD.values))
        _w = CALV.B22_열림 - CALV.groupby("항체").B22_열림.transform("mean")
        _bn = float(np.nanstd(_w.values, ddof=1))
        print(f"  프레임 간 SD 중앙 {_fn:.1f} nm³  ·  항체 안 구성체 간 SD {_bn:.1f} nm³")
        if _fn > 1e-12:
            print(f"  비 = {_bn/_fn:.2f}")
        if not (_fn > 1e-12 and _bn > 2*_fn):
            print("  ★★ 구성체 간 차이가 프레임 선택 잡음의 2배를 못 넘는다.")
            print("     지금 재고 있는 것은 **어느 프레임을 골랐나** 다. 프레임을 늘려라.")
        else:
            print("  구성체 간 차이가 프레임 잡음보다 충분히 크다 — 읽어도 된다.")

    _live = [r.관측량 for r in VC.itertuples() if r.항체내몫 >= 0.3]
    if not _live:
        print("\n  ★★ 살아남은 관측량이 없다. CALVADOS 도 항체를 재고 있다.")
        print("     링커 축이 아니라는 뜻이고, 표본을 늘려도 안 바뀐다.")
    elif "ML_RESULT" not in dir() or not HC:
        print(f"\n  살아남은 관측량: {_live} — 라벨 검정은 7절을 먼저 돌려라.")
    else:
        print("="*76)
        print(f"★ 10절 판정 ② — 라벨 검정 (탐색, Holm 보정). 살아남은 것만: {_live}")
        print("="*76)
        _M = USE.merge(CALV, on=["항체", "링커"], how="left", validate="one_to_one")
        _Y = make_target(_M, HC, block=GROUP, higher_is_worse=HIGHER_IS_WORSE).values
        _D = Design(_M, block=GROUP)
        _rs = []
        for c in _live:
            v = pd.to_numeric(_M[c], errors="coerce").values
            if np.isfinite(v).sum() < len(_M)-1 or np.nanstd(v) < 1e-12: continue
            r = perm_test_signal(v, _Y, _D, seed=21+len(_rs))
            _rs.append(dict(관측량=c, 일치=r["일치"], 일치도=round(r["일치도"], 3),
                            순열p=round(r["순열p"], 4)))
        if _rs:
            T10 = pd.DataFrame(_rs)
            T10["Holm_p"] = holm(T10.순열p.values).round(4)
            display(T10)
            print("  ※ 10절은 **탐색**이다. 사전 등록된 확증 검정은 7절 ① 하나뿐이다.")
            print("     여기서 유의해도 '발견' 이지 '확증' 이 아니다 — 다시 재서 확인해야 한다.")
        print("\n  ※ 그리고 이것으로 **도메인 교환(diabody) 가설은 검정되지 않는다.**")
        print("    CALVADOS 는 비특이적 회합만 본다. 특이적 계면은 못 만든다.")
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 나오는 것

| 파일 | 내용 |
|---|---|
| `{OUT}/{항체}/ref_m0..3.pdb` · `ref.pdb` | ABodyBuilder2 기준점 (Δ 의 영점) |
| `{OUT}/{항체}/flow/{링커}__BioEmu_s*.pdb` | 앙상블 프레임 |
| `{OUT}/{항체}/flow/{링커}__BioEmu.done` | 완료 표식 (이어달리기 판정) |
| **`{OUT}/frames.csv`** | 구조 한 줄씩 — 원시 ABangle 6값 · Δ 6축 · OCD5 · OCD6 |
| `{OUT}/ref_angles.csv` | 항체별 기준점 각도. Δ 를 다시 정의할 때 ABangle 없이 된다 |
| `{OUT}/antibody_keys.csv` | **(D1,D2,블록,배향) → 항체 키.** v09 폴더 이름을 고정한다 |
| **`{OUT}/geom.csv`** | 구조 한 줄씩 — 링커 접촉 · **나선도** · Rg · 신장도 |
| **`{OUT}/interface.csv`** | **항체 한 줄씩** — VH-VL 계면 BSA · 접촉밀도 · 소수성 · 흔들림 · 경쟁 설명 |
| `{OUT}/iptm.csv` | 항체 한 줄씩 — AF2-Multimer ipTM · pTM · 계면 PAE |
| **`{OUT}/features.csv`** | **구성체 한 줄씩** — ML 이 먹는 표 |
| **`{OUT}/insilico_screen.csv`** | 안 돌려본 링커까지의 예측 (후보 정렬용) |
| `{OUT}/fig_landscape.png` · `fig_channel.png` · `fig_ml.png` | 그림 셋 |
| `{OUT}/{항체}/{링커}__BioEmu_x20.pdb` | MODEL 20개짜리 (PyMOL/ChimeraX) |

---

## 결과를 읽는 순서 — 결과 보기 전에 박는다

**0단계 · 잡음 바닥이 잡혔는가** (4절 씨앗 반복 → 6절-B·6절-C 음성대조)
- 같은 서열 두 실행의 차이가 구성체 간 흩어짐의 절반을 넘는다 → **여기서 멈춘다.**
  아래 어떤 차이도 링커 때문인지 실행마다 달라서인지 구분할 수 없다.

**1단계 · 조성이 모델에 닿는가** (6절-D, 국소)
- `(EAAAK)ₙ` 이 `(G4S)ₙ` 보다 링커 나선도를 못 올린다 → **조성 채널이 죽었다.**
  교과서적 강직 나선조차 코일과 구분이 안 되는 것이다. 조성 해석을 전부 접는다.

**2단계 · 그 조성이 배향까지 가는가** (6절-C, 전역)
- η²(조성) 이 작고 η²(길이) 만 크다 → **배향은 길이만 본다.**
  3·4단계에서 무엇이 나오든 그것은 "긴 링커가 나쁘다" 의 다른 말이다. 그렇게 보고한다.
- **1단계 양성 · 2단계 음성은 실패가 아니라 결과다** — "조성은 링커 국소 구조를
  바꾸지만 도메인 배향까지는 전달되지 않는다". 이 연구가 낼 수 있는 가장 구체적인
  기전 진술이고, 그대로 쓰면 된다.
- 둘 다 작다 → 앙상블이 링커에 아예 안 반응한다. **여기서 접는다.**

**3단계 · 신호가 있는가** (7절 ①, 확증 검정)
- 순열 p < 0.05 → 신호가 있다. 3단계로.
- p ≥ 0.05 → ④ 검정력 곡선을 **반드시 같이 읽는다.**
  MDE 보다 작은 효과는 이 설계로 애초에 못 본다.
  "효과가 없다" 가 아니라 **"MDE 보다 작은 효과는 못 봤다"** 가 맞다.

**4단계 · 같은 링커인데 왜 블록마다 다른가** (7절-C, 계면 상호작용)
- 항체별 기울기를 계면강도에 회귀한다. 정밀도가중, Freedman–Lane, 정확 순열, **양측**.
- ★ **주효과로는 물을 수 없다.** 계면강도가 항체마다 상수라 항체 안 중심화가 지운다.
  상호작용만 살아남고, 다행히 그게 물어야 할 형태이기도 하다.
- ★ **관문은 ① 의 y-무관 점검뿐이다.** 기울기 이질성 F 는 관문이 아니다 — 전방위라
  방향을 아는 본검정보다 둔하고, 게이트로 쓰면 검정력을 절반 가까이 잃는다.
- ★ **②-C 반증 원장을 반드시 같이 읽는다.** 특히 **평균열림** 축 —
  AF2 도 BSA 도 필요 없는 공짜 경쟁자이고, 그게 유의하면 계면강도가 설명하는 것이
  "계면" 인지 "항체마다 작동점이 다르다" 인지 가를 수 없다.
- ★ **④ 의 λ_S·λ_O 를 본다.** 곱이 0.5 미만이면 어떤 효과크기로도 80% 검정력에
  못 닿는다. 귀무가 나와도 "없다" 가 아니라 "이만큼 시끄러우면 애초에 못 본다" 다.
- 부호가 음수면 조건부 예측 방향, 양수면 조건화가 새어 주변부 관계가 보인 것이다.

**5단계 · 길이 말고 다른 게 있는가** (7절 ③, 증분 검정)
- 증분 p < 0.05 → **BioEmu 를 돌린 값어치가 여기 있다.**
- p ≥ 0.05 → 앙상블은 길이 위에 아무것도 더하지 않았다. 그것도 결과다.
- ★ 단, 6절-B 의 교란 원장이 그 특징을 **"길이의 결정적 함수"** 로 판정했다면
  이 p 는 '효과 없음' 이 아니라 **'이 설계로는 답할 수 없음'** 이다. 셀이 그렇게 찍는다.

---

## 이 파이프라인이 증명할 수 없는 것 — 미리 적어 둔다

1. **응집은 모델 밖이다.** BioEmu 는 *single protein chains* 를 emulate 하고
   올리고머 상태는 암묵적이다 (Lewis et al. 2025, Discussion). 우리가 재는 것은
   **단량체 앙상블의 열린 정도**이고, HMW 는 **이량체 형성의 결과**다.
   그 사이의 연결(열린 단량체 → 사슬간 VH-VL 결합 → 응집)은 모델이 아니라 **가설**이다.
   상관이 나와도 그 가설을 증명하지는 않는다.

2. **인공 링커는 논문이 검증한 범위 밖이다 — 다만 양면이다.**
   성능이 서열유사도 30% 에서 평탄해지는데 그건 **천연 단백질** 기준이고
   `(G4S)ₙ` 에는 진화 정보가 없다. **반대 증거도 있다**: PPFT 는 점돌연변이
   50만 건으로 학습했고 ΔΔG Spearman > 0.6, Fig 4F 는 단일 치환 Ile7→Pro 가
   *그 잔기가 있는 나선의* 헬릭스 함량을 떨어뜨리는 것을 보인다. 한 잔기가
   모델에 닿는다. 그래서 이건 **양면 가설**이고, 6절-D(국소)·6절-C(전역)가
   그것을 직접 재는 유일한 장치다. 어느 쪽으로 나오든 답이 된다.

3. **150 프레임은 논문 기준의 1.5% 다.** 본문 결과는 전부 10,000 샘플로 냈다.
   분포의 **꼬리**(= 열림분율이 사는 곳)는 그만큼 덜 수렴했다.
   6절-B 의 특징 잡음 원장이 그 영향을 감쇠배율로 정량한다.

4. **독립 단위는 26이 아니라 5다.** 항체가 5개뿐이므로 자유도도 5개분이다.
   합성 데이터를 아무리 만들어도 이 숫자는 안 늘어난다 — 라벨이 없기 때문이다.
   늘리려면 실제로 만들어 재야 하고, **7절 ⑥ 이 어느 쪽이 값싼지 측정해서 말해 준다.**
   (검정 통계량이 항체 안 쌍이라 쌍 수 = 항체 × C(링커,2) 다. 링커에는 제곱으로,
   항체에는 선형으로 는다 — 그래서 **기존 항체에 링커를 더하는 쪽이 구성체당 더 싸다.**
   대신 항체를 늘리면 결론의 적용 범위가 넓어진다. 어느 쪽이 필요한지는 목적이 정한다.)

5. **A∘B 스크리닝은 검증이 아니다.** 후보를 정렬해 줄 뿐이고, 상위 후보는
   **실제로 만들어 재봐야** 한다. 8절이 `StageB몫` 을 찍는다 — 그 값이 0.7 을 넘으면
   불확실성의 대부분이 라벨 26개에서 오는 것이고, 셀이 스스로 그렇게 인쇄한다.

6. **평형만 놓고 보면 계면 친화도는 이량체 분율에서 정확히 상쇄된다.**
   단량체는 계면이 하나, 이량체는 둘, 열리는 데 하나가 드니 지수가 상쇄된다
   (`tools/thermo_check.py` 로 확인: Kd 3.5 자릿수에 이량체 분율 0.001988 → 0.001992).
   우리가 **조건부**로 묻기 때문에만 살아남는다 — 열린 분율을 BioEmu 로 측정해
   되돌려 넣지 않기 때문이다. 그 조건화가 새면 부호가 뒤집힌다. 그래서 양측이다.

7. **계면 강도의 대리지표는 친화도가 아니다.** ipTM 은 "AF2 가 이 배치를 얼마나
   확신하나" 이고 BSA 는 면적이다. 둘 다 ΔG 가 아니다. 그리고 VH-VL 계면은 PDB 에서
   가장 흔한 단백질-단백질 계면이라 ipTM 이 전부 0.9 근처에 몰려 변별력이 없을 수 있다 —
   3절-C 가 변동계수를 재서 그 경우 **사전 등록된 규칙대로 BSA 로 자동 전환**한다.

8. **`physical_steering` 은 논문에 없는 저장소 수정판이다.** SMC 재표집이 배치 안
   프레임에 공통 조상을 만들 수 있고, 그러면 "통계적으로 독립" 이라는 논문의 보장이
   따라오지 않는다. 6절-E 가 ICC 로 유효 표본수를 직접 잰다.
''')

# ═════════════════════════════════════════════════════════════════════════════
def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "FvFlow_v12.ipynb")
    bad = 0
    for i, (t, s) in enumerate(CELLS):
        if t != "code":
            continue
        # Colab 매직(!pip …)은 파이썬 문법이 아니다 — 주석으로 바꿔 놓고 파싱해 본다
        chk = "\n".join(("#MAGIC " + l) if l.lstrip().startswith("!") else l
                        for l in s.split("\n"))
        try:
            ast.parse(chk)
        except SyntaxError as e:
            bad += 1
            print(f"★ 셀 {i} 문법 오류 {e.lineno}행: {e.msg}")
            for k, l in enumerate(chk.split("\n")[max(0, e.lineno-3):e.lineno+2],
                                  max(1, e.lineno-2)):
                print(f"   {k:4d} | {l}")
    if bad:
        print(f"\n★ {bad}개 셀이 파싱되지 않는다 — 노트북을 쓰지 않는다")
        return 1
    if "--check" in sys.argv:
        print(f"OK — 코드 셀 {sum(1 for t, _ in CELLS if t=='code')}개 전부 파싱됨")
        return 0
    nb = dict(
        cells=[dict(cell_type=t, metadata={},
                    **({"source": (s + "\n").splitlines(keepends=True)}
                       if t == "markdown" else
                       {"source": (s + "\n").splitlines(keepends=True),
                        "execution_count": None, "outputs": []}))
               for t, s in CELLS],
        metadata=dict(
            colab=dict(provenance=[], toc_visible=True),
            kernelspec=dict(name="python3", display_name="Python 3"),
            language_info=dict(name="python"),
            accelerator="GPU"),
        nbformat=4, nbformat_minor=0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    nc = sum(1 for t, _ in CELLS if t == "code")
    nm = len(CELLS) - nc
    ch = sum(len(s) for _, s in CELLS)
    print(f"썼다 {out}")
    print(f"  셀 {len(CELLS)}개 (코드 {nc} · 마크다운 {nm}) · 원본 {ch:,}자")
    return 0


if __name__ == "__main__":
    sys.exit(main())
