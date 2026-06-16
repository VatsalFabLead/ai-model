"""Example client — OpenAI-compatible chat API."""

import httpx

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
API_KEY = "change-me-to-a-strong-key"


def main() -> None:
  payload = {
    "model": "custom-nexus-v1",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Who are you?"},
    ],
    "max_tokens": 128,
    "temperature": 0.7,
  }
  headers = {"Authorization": f"Bearer {API_KEY}"}

  with httpx.Client(timeout=120) as client:
    response = client.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
  main()
