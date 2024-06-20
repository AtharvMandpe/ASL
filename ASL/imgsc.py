import requests
import webbrowser
def img(words):
    url = "https://v1.genr.ai/api/circuit-element/generate-image"

    payload = {
        "prompt": words,
        "height": 512,
        "width": 512,
        "model": "stable-diffusion-2",
        "n_images": 1
    }
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)
    str= response.text
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
    webbrowser.open(tttt)