import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/financebench/financebench_open_source.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

terms = [
    ("AMD", "quick ratio"),
    ("PepsiCo", "capital expenditure"),
    ("Amazon", "revenue"),
    ("Amazon", "DPO"),
    ("Best Buy", "inventories"),
    ("Boeing", "net property"),
    ("Johnson & Johnson", "inventory"),
    ("Adobe", "operating income"),
    ("Microsoft", "debt"),
    ("Pfizer", "PPNE"),
    ("General Mills", "cash conversion cycle"),
]

for comp, term in terms:
    for r in records:
        if comp.lower() in r.get("company", "").lower() and term.lower() in r.get("question", "").lower():
            ev = r.get("evidence", [{}])[0]
            print(f"ID: {r.get('financebench_id')} | {r.get('company')} | {r.get('doc_name')} (p.{ev.get('evidence_page_num')})")
            print(f"  Q: {r.get('question')}")
            print(f"  A: {r.get('answer')}")
            print()
            break
