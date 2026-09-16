"""노트북을 **셀 0번부터 차례로** 읽으며 순서 의존을 찾는다.

test_notebook.py 는 분석 셀만 골라 돌리면서 앞 셀이 만들었을 이름을 가짜로 넣어 준다.
그래서 "위에서부터 차례로 돌리면 죽는" 문제를 원리적으로 못 잡는다. 이 파일이 그걸 잡는다.

잡는 것
  1. **앞으로 참조** — 아직 정의 안 된 이름을 쓰는 셀
  2. **조건부 정의** — if 안에서만 정의되는 이름을 뒤 셀이 무조건 쓰는 경우
     (스위치를 끄면 NameError 가 난다 — 노트북에서 제일 흔한 사고다)
  3. 셀이 정의하는 이름과 쓰는 이름 목록

돌리는 법:  python3 tools/check_order.py [노트북경로]
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import sys

NB_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "notebooks", "FvFlow_v12.ipynb")

#: Colab 이 기본으로 넣어 주는 이름들
COLAB = {"display", "get_ipython", "In", "Out", "exit", "quit"}


def strip_magic(src: str) -> str:
    """`!pip …` 같은 Colab 매직을 주석으로 바꾼다 (파이썬 문법이 아니다)."""
    return "\n".join(("#MAGIC " + l) if l.lstrip().startswith(("!", "%")) else l
                     for l in src.split("\n"))


class Scan(ast.NodeVisitor):
    """한 셀이 **최상위에서** 무엇을 정의하고 무엇을 읽는지 모은다.

    함수 안의 지역 이름은 세지 않는다 — 함수 본문에서 읽는 전역만 '사용' 으로 센다.
    """

    def __init__(self):
        self.defined = set()          # 무조건 정의됨
        self.cond = set()             # if/try 안에서만 정의됨
        self.used = []                # (이름, 줄번호)
        self._depth = 0               # 조건부 깊이
        self._scopes = []             # 함수/컴프리헨션 지역 이름

    # ── 정의 ────────────────────────────────────────────────────────────
    def _bind(self, name):
        (self.cond if self._depth else self.defined).add(name)

    def _local(self, name):
        if self._scopes:
            self._scopes[-1].add(name)
        else:
            self._bind(name)

    def visit_Assign(self, node):
        for t in node.targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    self._local(n.id)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self._local(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        # ★ `x += 1` 은 x 를 읽고 쓴다. 다만 함수 안에서 이미 지역으로 묶였으면
        #   전역 사용이 아니다 — 안 그러면 `rows, rss, dof = [], 0, 0` 뒤의
        #   `rss += ...` 가 전부 오탐이 된다.
        if isinstance(node.target, ast.Name):
            if not any(node.target.id in sc for sc in self._scopes):
                self.used.append((node.target.id, node.lineno))
            self._local(node.target.id)
        self.visit(node.value)

    def visit_For(self, node):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self._local(n.id)
        self.visit(node.iter)
        self._depth += 1                      # 루프 몸통은 0회일 수 있다
        for s in node.body + node.orelse:
            self.visit(s)
        self._depth -= 1

    def visit_With(self, node):
        for it in node.items:
            self.visit(it.context_expr)
            if it.optional_vars:
                for n in ast.walk(it.optional_vars):
                    if isinstance(n, ast.Name):
                        self._local(n.id)
        for s in node.body:
            self.visit(s)

    def visit_Import(self, node):
        for a in node.names:
            self._bind((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self._bind(a.asname or a.name)

    def visit_FunctionDef(self, node):
        self._bind(node.name)
        self._scopes.append({a.arg for a in node.args.args}
                            | {a.arg for a in node.args.kwonlyargs}
                            | ({node.args.vararg.arg} if node.args.vararg else set())
                            | ({node.args.kwarg.arg} if node.args.kwarg else set()))
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        for s in node.body:
            self.visit(s)
        self._scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._scopes.append({a.arg for a in node.args.args}
                            | {a.arg for a in node.args.kwonlyargs})
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        self.visit(node.body)
        self._scopes.pop()

    def visit_ClassDef(self, node):
        self._bind(node.name)
        self._scopes.append(set())
        for s in node.body:
            self.visit(s)
        self._scopes.pop()

    def visit_ExceptHandler(self, node):
        if node.name:
            self._local(node.name)
        for s in node.body:
            self.visit(s)

    def _comp(self, node):
        self._scopes.append(set())
        for g in node.generators:
            self.visit(g.iter)
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name):
                    self._scopes[-1].add(n.id)
            for f in g.ifs:
                self.visit(f)
        for f in ("elt", "key", "value"):
            if hasattr(node, f) and getattr(node, f) is not None:
                self.visit(getattr(node, f))
        self._scopes.pop()

    visit_ListComp = visit_SetComp = visit_GeneratorExp = _comp

    def visit_DictComp(self, node):
        self._comp(node)

    # ── 조건부 ──────────────────────────────────────────────────────────
    def visit_If(self, node):
        self.visit(node.test)
        self._depth += 1
        for s in node.body + node.orelse:
            self.visit(s)
        self._depth -= 1

    def visit_Try(self, node):
        self._depth += 1
        for s in node.body + node.handlers + node.orelse:
            self.visit(s)
        self._depth -= 1
        for s in node.finalbody:
            self.visit(s)

    def visit_While(self, node):
        self.visit(node.test)
        self._depth += 1
        for s in node.body + node.orelse:
            self.visit(s)
        self._depth -= 1

    # ── 사용 ────────────────────────────────────────────────────────────
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not any(node.id in sc for sc in self._scopes):
                self.used.append((node.id, node.lineno))


def analyse(path=NB_DEFAULT):
    nb = json.load(open(path, encoding="utf-8"))
    known = set(dir(builtins)) | COLAB
    cells, problems = [], []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = strip_magic("".join(c["source"]))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(dict(셀=i, 종류="문법", 이름="", 줄=e.lineno, 메시지=e.msg))
            continue
        sc = Scan()
        for s in tree.body:
            sc.visit(s)
        # 이 셀이 쓰는데 **아직 아무도 정의 안 한** 이름
        for nm, ln in sc.used:
            if nm in known or nm in sc.defined or nm in sc.cond:
                continue
            if nm.startswith("__"):
                continue
            problems.append(dict(셀=i, 종류="앞으로참조", 이름=nm, 줄=ln,
                                 메시지="이 셀 이전에 정의된 적이 없다"))
        # 이 셀이 쓰는데 **조건부로만** 정의된 이름
        for nm, ln in sc.used:
            if nm in known or nm in sc.defined:
                continue
            if nm in COND_ONLY and nm not in sc.cond:
                problems.append(dict(셀=i, 종류="조건부정의", 이름=nm, 줄=ln,
                                     메시지=f"셀 {COND_ONLY[nm]} 의 if/try 안에서만 정의된다"))
        for nm in sc.defined:
            COND_ONLY.pop(nm, None)
        for nm in sc.cond:
            if nm not in known:
                COND_ONLY.setdefault(nm, i)
        known |= sc.defined | sc.cond
        cells.append(dict(셀=i, 정의=len(sc.defined), 조건부=len(sc.cond),
                          첫줄=src.strip().split("\n")[0][:60]))
    return cells, problems


COND_ONLY: dict[str, int] = {}

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else NB_DEFAULT
    cells, probs = analyse(path)
    print(f"{os.path.basename(path)} — 코드 셀 {len(cells)}개")
    print("=" * 78)
    if not probs:
        print("순서 문제 없음 — 위에서부터 차례로 돌려도 이름이 다 준비된다.")
        sys.exit(0)
    from collections import Counter
    cnt = Counter(p["종류"] for p in probs)
    print("찾은 것:", dict(cnt))
    print()
    cur = None
    for p in probs:
        if p["셀"] != cur:
            cur = p["셀"]
            first = next((c["첫줄"] for c in cells if c["셀"] == cur), "")
            print(f"\n▶ 셀 {cur}  {first}")
        print(f"   [{p['종류']}] {p['이름']:<22} {p['줄']:>4}행  {p['메시지']}")
    print()
    print("=" * 78)
    print(f"총 {len(probs)}건")
    sys.exit(1)
