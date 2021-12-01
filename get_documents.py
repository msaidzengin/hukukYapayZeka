import requests

url = "https://karararama.danistay.gov.tr/getDokuman?id="

with open("documents.txt") as file:
    lines = file.readlines()
    document_ids = [line.rstrip() for line in lines]

for id in document_ids:
    request_url = url + id
    response = requests.get(url)
    file = open("documents/" + id + ".html","w")
    file .write(response.text)
    file.close()
    print(request_url)
    break