# -*- coding: utf-8 -*-
"""
merge_controlled_comparison.py — one-off utility.

The "final" (spatial-bbox) N=5 result was produced by a run before the
architecture flip (tag "final_spatialbbox"), and the "cbam" (spatial-bbox)
N=5 result by a separate --only cbam run afterward (tag "cbam", since
spatial bbox is now the default with no suffix). Each run's own
controlled_comparison.json only contains its own config, so this merges
both into the single combined file that the paper/README reference.
"""
import json

FINAL_SRC = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\controlled_comparison_spatialbbox.json"
CBAM_SRC = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\controlled_comparison.json"
OUT = r"C:\Users\enesm\Desktop\emadet\emadetforspringer\results\controlled_comparison.json"

final_doc = json.load(open(FINAL_SRC, encoding="utf-8"))
cbam_doc = json.load(open(CBAM_SRC, encoding="utf-8"))

assert list(final_doc["results"].keys()) == ["final_spatialbbox"], final_doc["results"].keys()
assert list(cbam_doc["results"].keys()) == ["cbam"], cbam_doc["results"].keys()
assert cbam_doc["config"]["seeds"] == final_doc["config"]["seeds"], "seed lists must match"
assert cbam_doc["config"]["epochs"] == 150 and len(cbam_doc["config"]["seeds"]) == 5, \
    "cbam_doc looks like a smoke test, not the real N=5 run -- refusing to merge"

merged = {
    "note": ("Both configs trained under identical N={}-seed / {}-epoch / "
             "early-stopping(patience={}) protocol, spatial-bbox-head architecture "
             "(see paper Section on the bbox-head ablation) -- supersedes both the "
             "conference draft's numbers and this project's earlier legacy-bbox-head "
             "N=5 comparison (see controlled_comparison_legacygapbbox.json)."
             ).format(len(cbam_doc["config"]["seeds"]), cbam_doc["config"]["epochs"],
                       cbam_doc["config"]["patience"]),
    "config": cbam_doc["config"],
    "results": {
        "final": final_doc["results"]["final_spatialbbox"],
        "cbam": cbam_doc["results"]["cbam"],
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged -> {OUT}")
for tag, r in merged["results"].items():
    print(f"\n{tag}:")
    for k, v in r["summary"].items():
        print(f"  {k:18s}: {v['mean']:.2f} +/- {v['std']:.2f}")
