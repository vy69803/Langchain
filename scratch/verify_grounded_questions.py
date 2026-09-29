import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/financebench/financebench_open_source.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# Let's pick 12 pristine questions across different archetypes
chosen_ids = [
    # 1. 3M CapEx 2018
    "financebench_id_00001",
    # 2. 3M Net PP&E 2018
    "financebench_id_00002",
    # 3. 3M Capital Intensive FY2022
    "financebench_id_00003",
    # 4. PepsiCo CapEx FY2021
    "financebench_id_00026",
    # 5. AMD Quick Ratio FY2022
    "financebench_id_00019",
    # 6. Microsoft COGS FY2016
    "financebench_id_00085",
    # 7. Microsoft Debt Balance FY2023 vs FY2022
    "financebench_id_00086",
    # 8. Amazon YoY Revenue Growth FY2016-FY2017
    "financebench_id_00010",
    # 9. Amazon FY2017 DPO
    "financebench_id_00009",
    # 10. Walmart FY2018-FY2020 3-yr EBITDA margin
    "financebench_id_00146",
    # 11. Boeing FY2022 Revenue Concentration >20%
    "financebench_id_00045",
    # 12. Adobe FY2015-FY2016 Operating Income Growth
    "financebench_id_00008",
    # 13. General Mills FY2020 Working Capital Ratio
    "financebench_id_00072",
    # 14. Johnson & Johnson FY2022 Inventory Turnover
    "financebench_id_00078",
]

# Let's inspect by matching financebench_id or finding by company/keywords
found = []
for r in records:
    fid = r.get("financebench_id")
    if fid in chosen_ids or any(k in r.get("question", "") for k in ["FY2018 capital expenditure", "net PPNE for 3M", "quick ratio for AMD", "FY2016 COGS for Microsoft", "unadjusted EBITDA % margin for Walmart", "operating cash flow ratio for Adobe", "working capital ratio? Define working capital ratio", "more than 20% of Boeing's revenue"]):
        found.append(r)

print(f"Found {len(found)} candidate grounded questions.")
for q in found[:15]:
    ev = q.get("evidence", [{}])[0]
    print(f"ID: {q.get('financebench_id')} | Comp: {q.get('company')} | Doc: {q.get('doc_name')} (p.{ev.get('evidence_page_num')})")
    print(f"  Q: {q.get('question')[:100]}...")
    print(f"  Ans: {q.get('answer')}")
    print()
