import urllib.request
import urllib.error

endpoints = ["/", "/health", "/docs", "/passenger_wsgi.py"]

for ep in endpoints:
    url = f"https://fabai.fableadtech.in{ep}"
    print(f"\n--- Testing URL: {url} ---")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as res:
            print(f"Status Code: {res.status}")
            txt = res.read().decode("utf-8", errors="replace")[:300]
            print(txt.encode("ascii", errors="replace").decode("ascii"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        txt = e.read().decode("utf-8", errors="replace")[:300]
        print(txt.encode("ascii", errors="replace").decode("ascii"))
    except Exception as e:
        print(f"Error: {e}")
