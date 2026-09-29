import asyncio
import sys

sys.path.insert(0, "src")
from production_api.agent import production_agent
from production_api.models import ChatRequest

async def test():
    # 1. Without filter (what user did)
    req_no_filter = ChatRequest(
        message="What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
    )
    res1 = await production_agent.process_request(req_no_filter)
    print("--- 1. Without filter ---")
    print("Sources retrieved:", [s.metadata.get("company") for s in res1.sources])
    print("Response:\n", res1.response[:300])

    # 2. With filter: company == 3M
    req_filtered = ChatRequest(
        message="What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
        metadata={"filter": {"company": "3M"}, "candidate_k": 25, "top_k": 5}
    )
    res2 = await production_agent.process_request(req_filtered)
    print("\n--- 2. With filter: company == 3M ---")
    print("Sources retrieved:", [s.metadata.get("company") for s in res2.sources])
    print("Response:\n", res2.response[:300])

if __name__ == "__main__":
    asyncio.run(test())
