import requests
from azure.identity import AzureCliCredential, DefaultAzureCredential

papyrus_endpoint = "https://WestUS2Large.papyrus.binginternal.com/chat/completions"
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

#cur_credential = AzureCliCredential()
cur_credential = DefaultAzureCredential()
access_token = cur_credential.get_token(verify_scope).token

# Replace "PapyrusCustomer" if you have an existing quota string
papyrus_quota_id = "PapyrusCustomer"

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json",
    "papyrus-model-name": "deepseekr1-eval",
    "papyrus-quota-id": papyrus_quota_id,
    "papyrus-timeout-ms": "100000",
    }

#json_dict = {"prompt": "how to cook fish?", "max_tokens": 32}
json_dict = {"model": "/mnt/blob/deepseek-bf16/", "messages": [{"role": "user", "content": "Explain the concept of photosynthesis in a way that a 10-year-old can understand."}]}

response = requests.post(papyrus_endpoint, headers=headers, json=json_dict)
print(response.status_code)
print(response.headers)
print(response.text)
