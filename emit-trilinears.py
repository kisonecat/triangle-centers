#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emit_trilinears.py — supports `where f/g/...` definitions, outputs `original`,
adds commas between objects, and aggressively expands implicit multiplication
(e.g., bc -> b*c, abc -> a*b*c).
"""

import argparse, json, re, sys
from typing import List, Tuple, Optional, Dict

# ----------------------------
# Config & regexes
# ----------------------------

ALLOWED_NAMES = {
    'a','b','c','A','B','C','s','S','R','r','ra','rb','rc','omega','pi'
}
FUNC_MATH = {'sin','cos','tan','sqrt'}
FUNC_HELPERS = {'sec','csc','cot'}

TAIL_CUT_RE = re.compile(r'(?i)\bwhere\b.*$')
EQ_SPLIT_RE = re.compile(r'\s=\s')

# Definitions: any name(p1,p2,p3) = RHS, from expr or context
WHERE_DEF_RE = re.compile(
    r'(?i)\bwhere\b\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)\s*=\s*(.+)$'
)
BARE_DEF_RE = re.compile(
    r'(?i)^\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)\s*=\s*(.+)$'
)

CALL_ABC_RE = re.compile(
    r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([abc])\s*,\s*([abc])\s*,\s*([abc])\s*\)'
)

# Function words stuck to symbols
FIX_FUNC_STUCK_RE = re.compile(r'\b(sin|cos|tan|sec|csc|cot)\s*([ABC])\b')
FIX_FUNC_SIMPLE_FRAC_RE = re.compile(r'\b(sin|cos|tan|sec|csc|cot)\s*([ABC])\s*/\s*([0-9]+)')
FIX_FUNC_SIMPLE_SCALE_RE = re.compile(r'\b(sin|cos|tan|sec|csc|cot)\s*(\d+)\s*([ABC])')

# Implicit multiplication helpers
RUN_VARS_RE = re.compile(r'(?<![A-Za-z0-9_])([abcABC]{2,})(?![A-Za-z0-9_])')  # abc -> a*b*c
IMPL_MULT_VARS_RE = re.compile(r'(?<![A-Za-z0-9_])([abcABC])\s*([abcABC])')    # a b -> a*b
IMPL_MULT_NUM_VAR_RE = re.compile(r'(\d)\s*([A-Za-z(])')                        # 2a or 2( -> 2*a / 2*(
IMPL_MULT_CLOSE_VAR_RE = re.compile(r'(\))\s*([A-Za-z(])')                      # )a or )( -> )*a / )*( 
VAR_LP_RE = re.compile(r'([abcABC])\s*\(')                                      # a( -> a*(

# Powers / braces
BRACE_SUP_RE = re.compile(r'\^\s*\{([^{}]+)\}')
BRACES_RE = re.compile(r'[\{\}]')
TRIG_POW_PAREN_RE = re.compile(r'\b(sin|cos|tan|sec|csc|cot)\s*\^\s*([0-9]+)\s*\(')
TRIG_POW_BARE_RE = re.compile(r'\b(sin|cos|tan|sec|csc|cot)\s*\^\s*([0-9]+)\s*([ABC])(?:\s*/\s*([0-9]+))?')

SQ2PAR_RE = re.compile(r'[\[\]]')
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")

# ----------------------------
# Utility & normalization
# ----------------------------

def normalize_greek(s: str) -> str:
    s = s.replace('π', 'pi').replace('Π', 'pi')
    s = s.replace('ω', 'omega').replace('Ω', 'omega')
    s = s.replace('−', '-').replace('–', '-').replace('—', '-')
    s = s.replace('⋅', '*').replace('’', "'").replace('′', "'")
    return s

def extract_definitions(expr: str, context_lines: List[str]) -> Dict[str, Tuple[Tuple[str,str,str], str]]:
    defs: Dict[str, Tuple[Tuple[str,str,str], str]] = {}
    def _harvest(line: str):
        line = line.strip()
        m = WHERE_DEF_RE.search(line)
        if m:
            fname, p1, p2, p3, rhs = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            defs[fname] = ((p1, p2, p3), rhs.strip()); return
        m2 = BARE_DEF_RE.search(line)
        if m2:
            fname, p1, p2, p3, rhs = m2.group(1), m2.group(2), m2.group(3), m2.group(4), m2.group(5)
            defs[fname] = ((p1, p2, p3), rhs.strip())
    if expr: _harvest(expr)
    for ln in context_lines or []:
        if isinstance(ln, str): _harvest(ln)
    return defs

def substitute_call_with_def(first_coord: str, defs: Dict[str, Tuple[Tuple[str,str,str], str]]) -> str:
    if not defs: return first_coord
    def subst_once(m):
        fname, x1, x2, x3 = m.groups()
        if fname not in defs: return m.group(0)
        (p1, p2, p3), rhs = defs[fname]
        mapping = {p1: x1, p2: x2, p3: x3}
        out = rhs
        for formal, actual in mapping.items():
            out = re.sub(rf'\b{re.escape(formal)}\b', actual, out)
        return f"({out})"
    return CALL_ABC_RE.sub(subst_once, first_coord)

def cut_junk(expr: str) -> str:
    expr = expr.strip()
    expr = TAIL_CUT_RE.sub('', expr)
    expr = EQ_SPLIT_RE.split(expr)[0]
    return expr.strip()

def first_coordinate(expr: str) -> str:
    i = expr.find(':')
    return expr[:i].strip() if i >= 0 else expr.strip()

def normalize_functions(expr: str) -> str:
    expr = FIX_FUNC_SIMPLE_FRAC_RE.sub(r'\1(\2/\3)', expr)
    expr = FIX_FUNC_SIMPLE_SCALE_RE.sub(r'\1(\2*\3)', expr)
    expr = FIX_FUNC_STUCK_RE.sub(r'\1(\2)', expr)
    return expr

def normalize_trig_powers(expr: str) -> str:
    expr = TRIG_POW_PAREN_RE.sub(lambda m: f"POWF({m.group(1)}(", expr)
    def repl_bare(m):
        fn, power, var, denom = m.group(1), m.group(2), m.group(3), m.group(4)
        arg = var if not denom else f"{var}/{denom}"
        return f"Math.pow({fn}({arg}),{power})"
    return TRIG_POW_BARE_RE.sub(repl_bare, expr)

def close_powf(expr: str) -> str:
    if "POWF(" not in expr: return expr
    out, i, n = [], 0, len(expr)
    while i < n:
        if expr.startswith("POWF(", i):
            out.append("Math.pow("); i += 5
            depth = 1
            while i < n and depth > 0:
                ch = expr[i]; out.append(ch)
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                i += 1
            out.append(",2)")
        else:
            out.append(expr[i]); i += 1
    return "".join(out)

# Catch runs of only triangle vars a|b|c (upper/lower), even after digits,
# but NOT when they're part of longer identifiers (letters/underscore around).
RUN_VARS_RE = re.compile(r'(?<![A-Za-z_])([abcABC]{2,})(?![A-Za-z0-9_])')

def expand_implicit_multiplication(s: str) -> str:
    """
    Iteratively expand implicit multiplication:
      bc   -> b*c
      abc  -> a*b*c
      2a   -> 2*a
      a(b) -> a*(b)
      )a   -> )*a
      2abc -> 2*a*b*c
      a(bc)-> a*(b*c)
    Runs multiple passes until no more changes are needed.
    """
    def pass_once(text: str) -> str:
        # 1) Runs like abc -> a*b*c (handles …2abc, (abc), etc.)
        def _run_repl(m):
            run = m.group(1)
            return '*'.join(list(run))
        text = RUN_VARS_RE.sub(_run_repl, text)

        # 2) Number before name or '(' : 2a -> 2*a ; 2( -> 2*(
        text = re.sub(r'(\d)\s*([A-Za-z(])', r'\1*\2', text)

        # 3) ')' before name or '(' : )a -> )*a ; )( -> )*(
        text = re.sub(r'(\))\s*([A-Za-z(])', r'\1*\2', text)

        # 4) var before '(' : a( -> a*(
        text = re.sub(r'([abcABC])\s*\(', r'\1*(', text)

        # 5) Adjacent single vars separated by spaces (safety): a b -> a*b
        text = re.sub(r'(?<![A-Za-z0-9_])([abcABC])\s+([abcABC])(?![A-Za-z0-9_])', r'\1*\2', text)

        return text

    prev = None
    cur = s
    # Iterate until stable (cap passes to avoid infinite loops)
    for _ in range(10):
        prev = cur
        cur = pass_once(cur)
        if cur == prev:
            break
    return cur

def pretokenize(expr: str) -> str:
    s = normalize_greek(expr)
    s = SQ2PAR_RE.sub(lambda m: '(' if m.group(0) == '[' else ')', s)
    s = BRACE_SUP_RE.sub(lambda m: f"^{m.group(1)}", s)
    s = BRACES_RE.sub('', s)
    s = normalize_functions(s)
    s = normalize_trig_powers(s)
    s = close_powf(s)
    s = expand_implicit_multiplication(s)
    return re.sub(r'\s+', ' ', s).strip()

# ----------------------------
# Tiny expression parser -> JS
# ----------------------------

NUM='NUM'; NAME='NAME'; OP='OP'; LP='('; RP=')'; ABS='|'

def tokenize(s: str):
    toks=[]; i=0; n=len(s)
    while i<n:
        ch=s[i]
        if ch.isspace(): i+=1; continue
        if ch.isdigit() or (ch=='.' and i+1<n and s[i+1].isdigit()):
            j=i+1
            while j<n and (s[j].isdigit() or s[j]=='.'): j+=1
            toks.append((NUM,s[i:j])); i=j; continue
        if ch.isalpha() or ch=='_' or ch in 'πΩω':
            j=i+1
            while j<n and (s[j].isalnum() or s[j]=='_' or s[j]=="'"): j+=1
            name=s[i:j].replace('π','pi').replace('Ω','omega').replace('ω','omega')
            toks.append((NAME,name)); i=j; continue
        if ch in '+-*/^,:': toks.append((OP,ch)); i+=1; continue
        if ch=='(': toks.append((LP,ch)); i+=1; continue
        if ch==')': toks.append((RP,ch)); i+=1; continue
        if ch=='|': toks.append((ABS,ch)); i+=1; continue
        i+=1
    return toks

class Node: 
    def js(self)->str: raise NotImplementedError
class Num(Node):
    def __init__(self,v): self.v=v
    def js(self): return self.v
class Var(Node):
    def __init__(self,n): self.n=n
    def js(self):
        nm=self.n
        if nm in ('r_a','r_b','r_c'): nm=nm.replace('_','')
        if nm=='pi': return 'Math.PI'
        return nm
class Bin(Node):
    def __init__(self,op,l,r): self.op,self.l,self.r=op,l,r
    def js(self):
        if self.op=='^': return f"Math.pow({self.l.js()},{self.r.js()})"
        return f"({self.l.js()} {self.op} {self.r.js()})"
class Unary(Node):
    def __init__(self,op,x): self.op,self.x=op,x
    def js(self): return f"({self.op}{self.x.js()})"
class Call(Node):
    def __init__(self,n,arg): self.n,self.arg=n,arg
    def js(self):
        n=self.n
        if n in FUNC_MATH: return f"Math.{n}({self.arg.js()})"
        if n in FUNC_HELPERS: return f"_{n.upper()}({self.arg.js()})"
        return f"(({n})*({self.arg.js()}))"
class AbsN(Node):
    def __init__(self,x): self.x=x
    def js(self): return f"Math.abs({self.x.js()})"

class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def peek(self,k=0): j=self.i+k; return self.toks[j] if j<len(self.toks) else None
    def eat(self,k=None,v=None):
        t=self.peek()
        if t and (k is None or t[0]==k) and (v is None or t[1]==v):
            self.i+=1; return t
        return None
    def parse(self): return self.expr()
    def expr(self):
        n=self.term()
        while True:
            t=self.peek()
            if t and t[0]==OP and t[1] in '+-':
                op=self.eat(OP)[1]; n=Bin(op,n,self.term())
            else: break
        return n
    def term(self):
        n=self.power()
        while True:
            t=self.peek()
            if t and t[0]==OP and t[1] in '*/':
                op=self.eat(OP)[1]; n=Bin(op,n,self.power()); continue
            if t and (t[0] in (NUM,NAME) or t[0] in (LP,ABS)):
                n=Bin('*',n,self.power()); continue
            break
        return n
    def power(self):
        n=self.unary()
        t=self.peek()
        if t and t[0]==OP and t[1]=='^':
            self.eat(OP,'^'); n=Bin('^',n,self.power())
        return n
    def unary(self):
        t=self.peek()
        if t and t[0]==OP and t[1] in '+-':
            op=self.eat(OP)[1]; return Unary(op,self.unary())
        return self.factor()
    def factor(self):
        t=self.peek()
        if not t: return Num('0')
        if t[0]==NUM: self.eat(NUM); return Num(t[1])
        if t[0]==NAME:
            name=self.eat(NAME)[1]
            if self.eat(LP):
                arg=self.expr(); self.eat(RP)
                return Call(name,arg)
            if name=='abs' and self.eat(ABS):
                inner=self.expr(); self.eat(ABS); return AbsN(inner)
            return Var(name)
        if t[0]==LP:
            self.eat(LP); n=self.expr(); self.eat(RP); return n
        if t[0]==ABS:
            self.eat(ABS); inner=self.expr(); self.eat(ABS); return AbsN(inner)
        self.eat(); return Num('0')

# ----------------------------
# Scoring / selection
# ----------------------------

def unknown_names_in(expr: str) -> List[str]:
    names = set(NAME_RE.findall(expr))
    keep=set()
    for nm in names:
        nm2 = nm.replace('π','pi').replace('ω','omega')
        if nm2 in FUNC_MATH|FUNC_HELPERS|{'Math','POWF'}: continue
        if nm2 in ('r_a','r_b','r_c'): nm2 = nm2.replace('_','')
        if nm2 in ALLOWED_NAMES: continue
        keep.add(nm)
    return sorted(keep)

def score_expr(raw: str, parsed_ok: bool) -> float:
    unk_pen = 100*len(unknown_names_in(raw))
    length_pen = 0.1*len(raw)
    bad = 0 if parsed_ok else 500
    if '|' in raw: bad += 2
    return unk_pen + length_pen + bad

# ----------------------------
# JS emission
# ----------------------------

JS_HEADER = "const CENTERS = [\n"
JS_FOOTER = "];"
JS_HELPERS_BODY = """      // common triangle helpers
      const s = (a + b + c) / 2;
      const S = Math.sqrt(Math.max(0, s*(s-a)*(s-b)*(s-c))); // area by Heron
      const r = S / s;
      const R = (Math.abs(Math.sin(A))>1e-12) ? (a/(2*Math.sin(A))) :
                (Math.abs(Math.sin(B))>1e-12) ? (b/(2*Math.sin(B))) :
                (Math.abs(Math.sin(C))>1e-12) ? (c/(2*Math.sin(C))) : NaN;
      const ra = S/(s-a), rb = S/(s-b), rc = S/(s-c);
      const cot_omega = (a*a + b*b + c*c) / (4*S);
      const omega = Math.atan(1 / cot_omega); // Brocard angle
      const _SEC = (x)=>1/Math.cos(x);
      const _CSC = (x)=>1/Math.sin(x);
      const _COT = (x)=>1/Math.tan(x);
"""

def emit_js_object(center: str, name: str, original_expr: str, expr_js: str) -> str:
    return f"""  {{
    center: {json.dumps(center)},
    name: {json.dumps(name)},
    original: {json.dumps(original_expr)},
    trilinear: function({{a,b,c,A,B,C}}) {{
{JS_HELPERS_BODY}      return ({expr_js.strip()});
    }}
  }}"""

# ----------------------------
# Core selection / compile
# ----------------------------

def pick_and_compile(expr_items: List[Tuple[str, List[str]]]) -> Tuple[Optional[str], Optional[str]]:
    candidates = []
    for raw, ctx in expr_items:
        if not raw or not raw.strip(): continue

        defs = extract_definitions(raw, ctx)
        original_expr = raw

        cut = cut_junk(raw)
        if not cut: continue

        first = first_coordinate(cut)
        if not first: continue

        first_inlined = substitute_call_with_def(first, defs)

        norm = pretokenize(first_inlined)
        try:
            toks = tokenize(norm)
            node = Parser(toks).parse()
            parsed_ok = node is not None
            js = node.js() if node else None
        except Exception:
            parsed_ok = False
            js = None

        sc = score_expr(norm, parsed_ok)
        candidates.append((sc, js, parsed_ok, original_expr, norm))

    if not candidates:
        return None, None

    candidates.sort(key=lambda t: t[0])
    for sc, js, ok, orig, _norm in candidates:
        if ok and js: return js, orig
    _sc, _js, _ok, orig, norm = candidates[0]
    return norm, orig

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_file')
    args = ap.parse_args()

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(args.json_file, 'r', encoding='latin-1', errors='ignore') as f:
            data = json.load(f)

    out = [JS_HEADER]
    for entry in data:
        center = (entry.get('center') or '').strip()
        name = (entry.get('name') or '').strip()

        expr_items: List[Tuple[str, List[str]]] = []
        for t in entry.get('trilinears', []):
            if isinstance(t, dict):
                expr = t.get('expr', '')
                ctx = [c for c in (t.get('context') or []) if isinstance(c, str)]
                expr_items.append((expr, ctx))

        js_expr, original_expr = pick_and_compile(expr_items)
        if not js_expr:
            js_expr = "NaN"
            original_expr = original_expr or ""

        # NOTE: append a comma after each object, per your request.
        out.append(emit_js_object(center, name, original_expr, js_expr) + ',')

    out.append(JS_FOOTER)
    sys.stdout.write("\n".join(out))

if __name__ == '__main__':
    main()

