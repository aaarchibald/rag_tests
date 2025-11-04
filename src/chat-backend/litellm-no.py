# import requests

# class LiteLLMClient:
#     def __init__(self, api_key, base_url="http://localhost:4000"):
#         self.api_key = api_key
#         self.base_url = base_url

#     def chat(self, model: str, message: str) -> str:
#         url = f"{self.base_url}/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": model,
#             "messages": [{"role": "user", "content": message}]
#         }

#         try:
#             response = requests.post(url, headers=headers, json=data, timeout=60)
#             response.raise_for_status()
#             content = response.json()["choices"][0]["message"]["content"]
#             return content
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed: {e}")
#             return "Error: Failed to connect to LiteLLM Proxy"
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             return "Error: Unexpected issue"

# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     client = LiteLLMClient(api_key="sk-test-master-key")

#     # Test local-llama3-1b
#     print("Testing local-llama3-1b:")
#     reply = client.chat(model="local-llama3-1b", message="What is the capital of France?")
#     print("Response:", reply)

#     # Test local-llama3-latest
#     print("\nTesting local-llama3-latest:")
#     reply = client.chat(model="local-llama3-latest", message="Explain the benefits of open source software.")
#     print("Response:", reply)
