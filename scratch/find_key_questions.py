import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/financebench/financebench_open_source.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(x) for x in f.readlines()]

for c in ['AMD', 'PepsiCo', 'Microsoft', 'Apple', 'Amazon', 'Walmart', 'Best Buy', 'Boeing', 'Adobe', 'Johnson & Johnson']:
    matching = [l for l in lines if c.lower() in l['company'].lower()]
    print(f"=== {c}: {len(matching)} questions ===")
    for m in matching[:3]:
        ev = m.get('evidence', [{}])[0]
        print(f"  [{m['doc_name']} p.{ev.get('evidence_page_num')}] Q: {m['question']}")
        print(f"     Ans: {m['answer']}")
