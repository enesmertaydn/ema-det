import re
import os

PAPER_DIR = os.path.dirname(os.path.abspath(__file__)) + r"\..\paper"

tex = open(os.path.join(PAPER_DIR, "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(PAPER_DIR, "refs.bib"), encoding="utf-8").read()

cite_groups = re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", tex)
all_cites = set()
for c in cite_groups:
    all_cites.update(k.strip() for k in c.split(","))

bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))

missing_bib = all_cites - bib_keys
unused_bib = bib_keys - all_cites
print("Cite keys used but missing from refs.bib:", missing_bib or "none")
print("Bib entries defined but never cited:", unused_bib or "none")

labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
missing_labels = refs - labels
print("\n\\ref{} targets missing a \\label{}:", missing_labels or "none")

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
print("\nFigure files referenced:")
for f in figs:
    p = os.path.normpath(os.path.join(PAPER_DIR, f))
    print(" ", f, "->", "OK" if os.path.exists(p) else "MISSING")

depth = 0
bad = False
for i, ch in enumerate(tex):
    if ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
    if depth < 0:
        print("\nBrace imbalance: unmatched } at char", i, repr(tex[max(0, i - 40):i + 10]))
        bad = True
        break
if not bad:
    print("\nBrace balance: final depth {} (0 = balanced)".format(depth))
