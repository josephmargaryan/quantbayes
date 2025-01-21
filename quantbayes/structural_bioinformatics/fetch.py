import requests
import json
import pandas as pd

# Download files
# !curl -O https://files.rcsb.org/download/1MBN.pdb

# Define the API endpoint
search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

with open("query_example.json") as query_payload:
    query_payload = json.load(query_payload)


response = requests.post(search_url, json=query_payload)
data = response.json()
result_set = data.get("result_set", [])
df = pd.DataFrame(result_set)
print(df)

"""data_api_url = "https://data.rcsb.org/rest/v1/core/entry/"
pdb_ids = ["2PGH", "3PEL", "3GOU", "1NGK", "6IHX"]

for pdb_id in pdb_ids:
    response = requests.get(f"{data_api_url}{pdb_id}")
    metadata = response.json()
    print(f"Details for {pdb_id}:")
    print(metadata.get('struct_keywords', {}).get('pdbx_keywords', "No keywords"))"""
