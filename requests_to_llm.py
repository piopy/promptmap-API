import requests
import json

url = "https://ENDPOINT"


def call_llm( 
    message: str,
):
    payload = json.dumps(
        {
            # edit the payload according to your LLM API, for example:
            "messages": [{"role": "user", "content": message}],
        }
    )
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        # other headers
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    # Try to make your function output a dictionary like this:
    # {
    #     "message": "success",
    #     "query": "MESSAGE",
    #     "response": "LLM RESPONSE",
    # }
    return response.json()
