import glob
import html
import html2text
import json

all_files = glob.glob("./documents/*.html")

kararlar = {}

for file in all_files:

    doc_id = file.split("/")[-1].split(".")[0]

    with open(file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    text = html.unescape(lines[69])
    text = html2text.html2text(text)

    print(doc_id)
    kararlar[doc_id] = text

with open('kararlar.json', 'w', encoding='utf-8') as f:
    json.dump(kararlar, f, ensure_ascii=False, indent=4)
