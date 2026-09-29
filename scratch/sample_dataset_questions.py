import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/financebench/financebench_open_source.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

selected_companies = [
    "3M", "PepsiCo", "AMD", "Best Buy", "Boeing", 
    "Johnson & Johnson", "Adobe", "Pfizer", "American Express", "General Mills"
]

results = []
for comp in selected_companies:
    matches = [r for r in records if r["company"] == comp]
    for m in matches[:2]:  # take up to 2 per company
        ev = m.get("evidence", [{}])[0]
        results.append({
            "company": comp,
            "doc_name": m.get("doc_name"),
            "page": ev.get("evidence_page_num"),
            "question": m.get("question"),
            "answer": m.get("answer"),
            "justification": m.get("justification"),
            "question_type": m.get("question_type"),
            "question_reasoning": m.get("question_reasoning")
        })

print(f"Total curated: {len(results)}")
for idx, r in enumerate(results, 1):
    print("=" * 80)
    print(f"[{idx}] {r['company']} ({r['doc_name']}, Page {r['page']})")
    print(f"    Category: {r['question_reasoning']} | Type: {r['question_type']}")
    print(f"    Question: {r['question']}")
    print(f"    Ground Truth Answer: {r['answer']}")
    if r['justification']:
        print(f"    Justification: {r['justification'][:200]}...")
