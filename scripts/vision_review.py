import sys, json, base64, urllib.request
key=open('${DEEPSEEK_KEY_FILE}').read().strip()
prompt=sys.argv[1]; files=sys.argv[2:]
parts=[{"type":"text","text":prompt}]
for f in files:
    data=base64.b64encode(open(f,'rb').read()).decode()
    parts.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+data}})
body={"model":"deepseek-v4-flash-vision-exp","messages":[{"role":"user","content":parts}],"max_tokens":1500}
req=urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
try:
    r=json.load(urllib.request.urlopen(req,timeout=180))
    print(r["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print("HTTP",e.code,e.read().decode()[:1000])
