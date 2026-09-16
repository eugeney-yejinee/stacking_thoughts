"""전체 시험을 **한 줄로** 돌린다.

순서가 곧 의존이다 — 위가 깨지면 아래는 읽을 필요가 없다.

  1. build --check   코드 셀이 파싱되나
  2. check_order     위에서부터 돌릴 때 이름이 다 준비되나 (정적)
  3. run_in_order    위에서부터 **실제로** 돌려본다 (동적)
     · 기본       — v09 출력이 다 있는 상태
     · --empty    — 빈 Drive. 5절 assert 에서 멈추는 것은 **설계**이고,
                    그 뒤 NameError 는 연쇄다. 러너가 둘 다 실패로 안 센다.
  4. 모듈 자체 시험  fvml · fvfeat · fvinterx
  5. test_notebook   분석 셀을 가짜 라벨 세 세계로 실행

돌리는 법:  python3 tools/test_all.py [--quick]
  --quick 이면 오래 걸리는 4·5 를 건너뛴다 (순서 문제만 볼 때).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

#: (이름, 인자, 0이 아닌 종료를 실패로 볼까)
FAST = [
    ("노트북 생성 + 파싱", ["build_v12.py", "--check"], True),
    ("셀 순서 (정적)", ["check_order.py"], True),
    ("셀 순서 (실행)", ["run_in_order.py"], True),
    # 빈 Drive 에서는 5절 assert 에서 멈추는 것이 설계다. 뿌리가 그 한 건인지만 본다.
    ("빈 Drive 에서 멈추는 자리", ["run_in_order.py", "--empty"], True),
]
SLOW = [
    ("추론 핵심 (fvml)", ["test_fvml.py"], True),
    ("특징 추출 (fvfeat)", ["test_fvfeat.py"], True),
    ("계면 상호작용 (fvinterx)", ["test_fvinterx.py"], True),
    ("분석 셀 실행 (세 세계)", ["test_notebook.py"], True),
]


def run(name, args, strict):
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, *args[:1]), *args[1:]],
                       capture_output=True, text=True)
    dt = time.time() - t0
    ok = (r.returncode == 0) or not strict
    tail = (r.stdout or r.stderr).rstrip().split("\n")
    print(f"{'✔' if ok else '✘'} {name:<26} {dt:5.1f}s")
    for L in tail[-4:]:
        print(f"      {L[:110]}")
    return ok, r


def main():
    quick = "--quick" in sys.argv
    plan = FAST + ([] if quick else SLOW)
    bad = []
    print("=" * 78)
    for name, args, strict in plan:
        ok, r = run(name, args, strict)
        if not ok:
            bad.append(name)
    print("=" * 78)
    if bad:
        print(f"실패 {len(bad)}건: {', '.join(bad)}")
        return 1
    print(f"전부 통과 ({len(plan)}건)" + ("  · --quick 이라 느린 시험은 건너뜀" if quick else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
