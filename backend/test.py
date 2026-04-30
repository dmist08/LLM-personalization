# test_single.py
import requests

response = requests.post(
    "https://myibgao--my-custom-llm-api-api.modal.run/generate",
    json={
        "prompt": "Scientists discovered a new battery.",
        "method": "no_personalization"
    },
    timeout=300
)

print("Status:", response.status_code)
print("Response:", response.json())