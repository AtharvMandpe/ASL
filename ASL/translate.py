import requests
def changelang(txte):
        url = "https://v1.genr.ai/api/circuit-element/translate-text"

        payload = {
            "text": txte,
            "temperature": 0.3,
            "target_language": "English"
        }
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, json=payload, headers=headers)

        print(response.text)
        l=[]
        a = 0;
        for i in response.text:
         if(i=='"'):
            a =0
         if(i=='h'):
            a = 1
         if(a==1):
            l.append(i) 
        tttt=''.join(l)
        print(tttt)
        return tttt;