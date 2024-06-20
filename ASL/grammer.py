import requests
def correct(gra):
    url = "https://v1.genr.ai/api/circuit-element/correct-grammar"

    payload = {
        "text": gra,
        "temperature": 0
    }
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text