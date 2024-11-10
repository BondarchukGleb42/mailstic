import os

import httpx

api = httpx.Client(base_url=os.getenv("API_URL"))
