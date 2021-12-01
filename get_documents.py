import requests

url = "https://karararama.danistay.gov.tr/getDokuman?id="

with open("urls.txt") as file:
    lines = file.readlines()
    urls = [line.rstrip() for line in lines]

for url in urls:
    response = requests.get(url)
    name = url.split("?id=")[1]
    file = open("documents/" + name + ".html","w")
    file .write(response.text)
    file.close()
    print(url)
