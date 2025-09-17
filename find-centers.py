#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract 'Trilinears' lines (with math markup preserved) from ETC.html,
including immediately following 'where …' / 'Let …' explanatory clauses.

Usage:
  python find-centers.py ETC.html > trilinears.json
  python find-centers.py ETC.html --emit html > trilinears.html.json
  python find-centers.py ETC.html --csv > trilinears.csv
"""
import argparse, csv, json, re, sys
from typing import List, Dict, Any
from bs4 import BeautifulSoup, UnicodeDammit, NavigableString, Tag
from html import unescape
from bs4 import Tag, NavigableString
from html import unescape
import re

LABEL_HEADS = (
    r"Trilinears", r"Barycentrics", r"Tripolars", r"Coordinates",
    r"X\(\d+\)", r"Also,", r"As a point", r"As an example"
)
STOP_LABEL_RE = re.compile(rf"^\s*(?:{'|'.join(LABEL_HEADS)})\b", re.IGNORECASE)
TRILINEAR_HEAD_RE = re.compile(r"^\s*Trilinears\b", re.IGNORECASE)
CONTEXT_START_RE = re.compile(r"^\s*(where|let)\b", re.IGNORECASE)

CENTER_ID_RE = re.compile(r"X\((\d+)\)")
NAME_AFTER_EQUALS_RE = re.compile(r"=\s*([A-Z][A-Z0-9 \-\(\)\/]+)")

def load_soup(path: str) -> BeautifulSoup:
    with open(path, "rb") as fh:
        raw = fh.read()
    dammit = UnicodeDammit(raw, is_html=True)
    text = dammit.unicode_markup or raw.decode("latin-1", errors="replace")
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(text, parser)
        except Exception:
            continue
    return BeautifulSoup(text, "html.parser")

def normalize_spaces(s: str) -> str:
    # collapse NBSP and whitespace
    import re
    from html import unescape
    return re.sub(r'\s+', ' ', unescape(s.replace('\xa0', ' '))).strip()

def extract_center_header(h3: Tag) -> dict:
    """
    Robustly parse headers like:
      <h3 id="X12">X(12)&nbsp;=&nbsp;{X(1),X(5)}-HARMONIC CONJUGATE OF X(11)</h3>
    Returns {"center": "X(12)", "name": "{X(1),X(5)}-HARMONIC CONJUGATE OF X(11)"}.
    """
    # Prefer id if present (e.g., id="X12")
    center = ""
    h3_id = h3.get("id")
    if h3_id and h3_id.startswith("X") and h3_id[1:].isdigit():
        center = f"X({int(h3_id[1:])})"

    # Fallback: parse center number from text
    txt_raw = " ".join(h3.stripped_strings)  # preserves "=" and content
    txt = normalize_spaces(txt_raw)

    if not center:
        m = re.search(r'X\((\d+)\)', txt)
        if m:
            center = f"X({m.group(1)})"

    # Name = everything after the first '=' (if any)
    name = ""
    eq_pos = txt.find('=')
    if eq_pos != -1:
        name = txt[eq_pos + 1 :].strip()
        # trim leading punctuation/spaces that sometimes sneak in
        name = name.lstrip(':;.- ').strip()

    return {"center": center, "name": name}

def html_to_texish(node) -> str:
    """
    Convert an inline HTML fragment to TeX-ish plain text:
      <sup>k</sup> -> ^{k}, <sub>k</sub> -> _{k}.
    Traverses children (no self-recursion).
    """
    def render(n) -> str:
        if isinstance(n, NavigableString):
            return str(n)
        if not isinstance(n, Tag):
            return ""
        if n.name == "sup":
            inner = "".join(render(c) for c in n.children)
            return f"^{{{inner}}}"
        if n.name == "sub":
            inner = "".join(render(c) for c in n.children)
            return f"_{{{inner}}}"
        if n.name == "br":
            return " "
        # Generic tag: ignore the wrapper, render children
        return "".join(render(c) for c in n.children)

    out = render(node)
    out = unescape(out)
    out = re.sub(r"[ \t\u00a0]+", " ", out).strip()
    return out

def strip_leading_label(s: str) -> str:
    return re.sub(r'^\s*Trilinears\b[:\s\u00a0-]*', '', s, flags=re.IGNORECASE).strip()

def block_to_html_lines(block: List[Tag]) -> List[str]:
    """
    Turn a list of sibling nodes (from an h3 to next h3) into visual 'lines' of HTML,
    splitting on <br> and block boundaries (<p>, <li>, etc.).
    """
    out_lines: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            html = "".join(buf).strip()
            if html:
                out_lines.append(html)
            buf.clear()

    def recurse(n: Any):
        if isinstance(n, NavigableString):
            buf.append(str(n))
            return
        if not isinstance(n, Tag):
            return
        # Block elements start a new visual line
        if n.name in ("p", "li", "div", "h4", "h5", "table", "tr", "td"):
            flush()
            for c in n.children:
                recurse(c)
            flush()
            return
        if n.name in ("br",):
            flush()
            return
        # inline: keep tag markup so we don't lose sup/sub
        # but minimize: we only need sup/sub tags preserved; others can be inlined
        if n.name in ("sup", "sub"):
            # keep exact tag to process later
            buf.append(str(n))
            return
        # generic inline container: descend
        for c in n.children:
            recurse(c)

    for node in block:
        recurse(node)
    flush()
    # normalize whitespace inside each line
    cleaned = []
    for h in out_lines:
        # collapse multiple spaces but keep tags untouched
        # remove redundant <span> etc. wrappers around whitespace
        h = re.sub(r'\s+', ' ', h)
        h = h.strip()
        if h:
            cleaned.append(h)
    return cleaned

def gather_section_nodes(h3: Tag) -> List[Tag]:
    nodes: List[Tag] = []
    for sib in h3.next_siblings:
        if isinstance(sib, Tag) and sib.name == "h3":
            break
        nodes.append(sib)
    return nodes

def extract_trilinears_from_section(h3: Tag, emit: str) -> Dict[str, Any]:
    header = extract_center_header(h3)
    nodes = gather_section_nodes(h3)
    # produce HTML "lines"
    lines_html = block_to_html_lines(nodes)

    entries = []
    i = 0
    while i < len(lines_html):
        html_line = lines_html[i]
        # Identify trilinear header lines
        if TRILINEAR_HEAD_RE.match(BeautifulSoup(html_line, "html.parser").get_text(strip=True)):
            # expression on this line (remove "Trilinears" label)
            expr_html = strip_leading_label(html_line)
            # Gather context lines that start with 'where' or 'Let'
            ctx_html: List[str] = []
            j = i + 1
            while j < len(lines_html):
                txt_j = BeautifulSoup(lines_html[j], "html.parser").get_text(strip=True)
                if CONTEXT_START_RE.match(txt_j):
                    ctx_html.append(lines_html[j])
                    j += 1
                    continue
                # stop if we hit another label or an empty line
                if not txt_j or STOP_LABEL_RE.match(txt_j):
                    break
                # be conservative: only accept short follow-ups that look like continuations
                # (e.g., “such that …”, “and …” right after a where/let)
                if ctx_html and re.match(r'^(such that|and|also|then|hence)\b', txt_j, re.IGNORECASE):
                    ctx_html.append(lines_html[j])
                    j += 1
                    continue
                break
            i = j - 1  # outer loop will i+=1

            if emit == "html":
                entry = {
                    "expr_html": expr_html,
                    "context_html": ctx_html
                }
            else:
                # convert HTML fragments to desired text flavor
                expr_soup = BeautifulSoup(expr_html, "html.parser")
                if emit == "tex":
                    expr = html_to_texish(expr_soup)
                    ctx = [html_to_texish(BeautifulSoup(h, "html.parser")) for h in ctx_html]
                else:
                    # plain text but keep ^ and _ markers for sup/sub
                    expr = html_to_texish(expr_soup)
                    ctx = [html_to_texish(BeautifulSoup(h, "html.parser")) for h in ctx_html]
                entry = {"expr": expr, "context": ctx}
            entries.append(entry)
        i += 1

    return {"center": header["center"], "name": header["name"], "trilinears": entries}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html_path")
    ap.add_argument("--emit", choices=["tex", "html", "text"], default="tex",
                    help="How to serialize math: 'tex' (^{},_{}), raw 'html', or plain 'text' (same as tex here).")
    ap.add_argument("--csv", action="store_true", help="Emit CSV instead of JSON (one row per trilinear).")
    args = ap.parse_args()

    soup = load_soup(args.html_path)

    centers = []
    for h3 in soup.find_all("h3"):
        # only real center headers: have id like X1 or text containing X(n)
        if h3.get("id", "").startswith("X") and h3.get("id", "")[1:].isdigit():
            centers.append(extract_trilinears_from_section(h3, args.emit))
        else:
            txt = " ".join(h3.stripped_strings)
            if CENTER_ID_RE.search(txt):
                centers.append(extract_trilinears_from_section(h3, args.emit))

    # keep only centers with at least one trilinear entry
    centers = [c for c in centers if c["trilinears"]]

    if args.csv:
        w = csv.writer(sys.stdout)
        if args.emit == "html":
            w.writerow(["center", "name", "expr_html", "context_html"])
            for c in centers:
                for e in c["trilinears"]:
                    w.writerow([c["center"], c["name"], e.get("expr_html", ""), " | ".join(e.get("context_html", []))])
        else:
            w.writerow(["center", "name", "expr", "context"])
            for c in centers:
                for e in c["trilinears"]:
                    w.writerow([c["center"], c["name"], e.get("expr", ""), " | ".join(e.get("context", []))])
    else:
        json.dump(centers, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

