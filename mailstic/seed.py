import json
import httpx

api = httpx.Client(base_url="http://localhost:8000")

with open("lib/processing/data/qa_base.json", "r") as f:
    data = json.load(f)

    for pof, qas in data.items():
        created_pof = api.post("/pof", json={"name": pof}).json()

        for qa in qas:
            api.post(
                f"/pof/{created_pof["slug"]}/qa", json={"question": "", "answer": qa}
            )
