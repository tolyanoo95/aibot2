import subprocess
import json

keys = [
    "TD.Lgnnny4S9XJdgge-.2saAQyvGTV2T1hv.KOAgHbkOVXFZqvP.yrbCY1PgDD8vT9j.TZNoxTNYhh5hTTl.kb33",
    "TD.crlhp8EFSOGonUOJ.6mMxeWIFYQtvGJj.2PFO-VHWKbk-cm2.nDpVPcpljk4fkLW.Oxd2kRjOMzi0Yts.hTVZ",
    "TD.DVnuaoaxFNRT1xnb.-c5pe6luxohV8Qa.RxOZfHDkhJrG955.c8KctNpdMxM8OAK.4wvtFAQD745RNQc.puci",
    "TD.RqK4Y7OSvpsvAleb.Xx7iuf-XX9V86DO.8Z0PfckIAyaRTL1.4-qwXpUEwMwzGrZ.zy3SGfRocFeXVgT.Nfgv",
    "TD.BDDr5Mkdv9P7jcBv.HGl8KnCzXgfUQSx.GiAyuItybr9u2tg.5DBEOQ-S7tzj-sN.p9Z42Wlj-bksdHl.rpdN",
    "TD.P1RQy3kV6rkCX-Js.8wpD7WjlIBu6l5O.wujparLjM0uHh5P.haoaBrqe-I8WegV.9mLshEQh6GT1INZ.8qOm",
    "TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4",
    "TD.DPgWHdw6g8MVojvQ.WVt-I5iaYeOUeHV.ya9VDZXlhKlPDbI.CpnH4VLpAN1kdtJ.lg6yktNRehqCPw2.2ZV9",
    "TD.7nY1VBeQdUxxMyMK.BDpk0GcNRMraF7E.Xnov8iHJGsDqKgF.ELjyVH474rltzA7.kuy6pcsxCVRI2N-.X7yr",
    "TD.Is8O6MvHHQAxTNeb.nHBwFeFnSwZKuBO.ytzIYxI-wciClZo.ocwIPaWIuXv-1Tx.qHUZVDXLgYjEtqA.a4DC",
    "TD.KDeubOKBOaOItHn6.rq2Tkeqseo2dt9x.rz-lob9od0fD--Y.2LWH9ZScU7GUvW3.dWSpVV5T-vKB6Mf.TGq-",
    "TD.34Q-uNaLfiYDgGrL.ASWs359NTrG3xtC.hJ6BeUBTavJthip.bv88e75OfyB8uWY.Fj9Bir3R4yU8vBo.V8aa",
    "TD.VuEId8q-wEOzCEnA.sUJBubEjkdYY1F7.NG-ZpKVhfjGuV6T.OjW7fGZmo7P4pUP.KhDisysGEM4SgW7.4NXI",
    "TD.sSGNjEpohBl9i0FV.zz0-s-b6DZ9c2lt.uoNpgBPGaTITrA9.YlW26JKS0PSfoOG.2pEnnkHAnTpOloD.3wMo",
]

url = "https://api.tardis.dev/v1/exchanges"

for i, key in enumerate(keys, 1):
    short = key[:20] + "..."
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "20", "-w", "\n%{http_code}", url,
             "-H", f"Authorization: Bearer {key}"],
            capture_output=True, text=True, timeout=25
        )
        lines = result.stdout.strip().split("\n")
        http_code = lines[-1].strip()
        if http_code == "200":
            print(f"  {i:2d}. ✅ WORKING  | {short}")
        elif http_code == "401":
            body = "\n".join(lines[:-1])
            try:
                msg = json.loads(body).get("message", "Unauthorized")
            except:
                msg = "Unauthorized"
            print(f"  {i:2d}. ❌ INVALID  | {short} | {msg}")
        elif http_code == "429":
            print(f"  {i:2d}. ⚠️  RATE LIM | {short} | rate limited, try later")
        else:
            print(f"  {i:2d}. ❓ HTTP {http_code} | {short}")
    except Exception as e:
        print(f"  {i:2d}. 💀 ERROR    | {short} | {e}")
