import requests

key = "TD.7nY1VBeQdUxxMyMK.BDpk0GcNRMraF7E.Xnov8iHJGsDqKgF.ELjyVH474rltzA7.kuy6pcsxCVRI2N-.X7yr"
headers = {"Authorization": f"Bearer {key}"}
url = "https://api.tardis.dev/v1/exchanges"
try:
    print(f"Testing {key[:15]}...")
    resp = requests.get(url, headers=headers, timeout=10)
    print(resp.status_code)
except Exception as e:
    print(e)
