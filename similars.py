import string, json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sbert_model = SentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens")

def preprocess(text):
    text = lower(text.translate(str.maketrans("", "", string.punctuation)))
    text = " ".join(text.split())
    return text

def lower(word):
    return (
        word.replace("İ", "i")
            .replace("Ş", "ş")
            .replace("Ç", "ç")
            .replace("Ğ", "ğ")
            .replace("Ö", "ö")
            .replace("Ü", "ü")
            .replace("I", "ı")
            .lower()
    )

def create_vectors(texts):
    corpus = [preprocess(i) for i in texts]
    document_embeddings = sbert_model.encode(corpus)
    return document_embeddings, corpus

def find_similars(document_embeddings, corpus, text, top_k=3, threshold=0.5):
    text = preprocess(text)
    query_embedding = sbert_model.encode(text)
    embeddings = np.insert(document_embeddings, 0, query_embedding, axis=0)
    pairwise_similarities = cosine_similarity(embeddings)
    indices = np.argsort(pairwise_similarities[0])[::-1][1:top_k+1] - 1  # -1 because we put query to the first index
    result = [{"index": i, "score": float(pairwise_similarities[0][i + 1]), "text": corpus[i]} for i in indices]
    return result

with open('kararlar.json', encoding='utf-8') as f:
    kararlar = json.load(f)

kararlar = [i for i in kararlar.values()]
document_embeddings, corpus = create_vectors(kararlar)

text = "Danıştay 12. Daire Başkanlığı         2021/5027 E.  ,  2021/5116 K. İçtihat Metni T.C. D A N I Ş T A Y ONİKİNCİ DAİRE Esas No : 2021/5027 Karar No : 2021/5116 TEMYİZ EDEN (DAVACI) : ... VEKİLİ : Av. … KARŞI TARAF (DAVALI) : … Bakanlığı VEKİLİ : Av. … İSTEMİN KONUSU : ... Bölge İdare Mahkemesi ... İdari Dava Dairesinin ... tarih ve E:…, K:… sayılı kararının temyizen incelenerek bozulması istenilmektedir. YARGILAMA SÜRECİ : Dava konusu istem: Kütahya İli, ... Adliyesi'nde hizmetli olarak görev yapan davacının, ''Silahla Tehdit'' suçundan 1 yıl 8 ay hapis cezasıyla cezalandırılmasına ilişkin ... Asliye Ceza Mahkemesinin … tarihli ve E:…, K:… sayılı kararının 03/10/2019 tarihinde kesinleşmesi üzerine, … Adli Yargı İlk Derece Mahkemesi Adalet Komisyonu Başkanlığının … tarihli ve … sayılı kararı ile getirilen teklif doğrultusunda, 657 sayılı Devlet Memurları Kanunu'nun 48/A-5 ve 98/b maddeleri uyarınca Devlet memurluğu görevine son verilmesine ilişkin … onay tarihli işlemin iptali istenilmiştir. İlk Derece Mahkemesi kararının özeti: … İdare Mahkemesince verilen … tarih ve E:…, K:… sayılı kararla; davacının kasten işlenmiş bir suçtan dolayı 1 yıl 8 ay süre ile kesinleşmiş mahkumiyetinin bulunduğu, bu haliyle 657 sayılı Devlet Memurları Kanunu'nun 48/A-5 maddesinde yer alan memuriyete girişte öngörülen şartlardan birini kaybetmiş olduğu görüldüğünden, aynı Kanun'un 98/b maddesinde yer verilen amir hüküm uyarınca davacının Devlet memurluğu görevine son verilmesine ilişkin dava konusu işlemde hukuka aykırılık bulunmadığı sonucuna ulaşıldığı gerekçesiyle davanın reddine karar verilmiştir. Bölge İdare Mahkemesi kararının özeti: ... Bölge İdare Mahkemesi ... İdari Dava Dairesince; istinaf başvurusuna konu İdare Mahkemesi kararının hukuka ve usule uygun olduğu ve davacı tarafından ileri sürülen iddiaların söz konusu kararın kaldırılmasını gerektirecek nitelikte görülmediği gerekçesiyle, 2577 sayılı İdari Yargılama Usulü Kanunu'nun 45. maddesinin üçüncü fıkrası uyarınca istinaf başvurusunun reddine karar verilmiştir. TEMYİZ EDENİN İDDİALARI : Söz konusu mahkeme kararının kesinleşmediği, Anayasa Mahkemesine ve Avrupa İnsan Hakları Mahkemesine başvuracağı, eksik inceleme sonucunda, hakkında açılan diğer davalarda yaptığı ön ödemeler dikkate alınmadan, hakkında daha önce verilen hükmün açıklanmasının geri bırakılması kararı kaldırılarak hapis cezasına hükmedildiği, mağdur olduğu, aynı olaydan iki kez ceza aldığı, ön ödeme yaptığına ilişkin evrakın ... Asliye Ceza Mahkemesince ilgili yerlere sunulmadığı ileri sürülmektedir. KARŞI TARAFIN SAVUNMASI: Dava konusu işlemin mevzuata uygun olduğu savunulmaktadır. DANIŞTAY TETKİK HÂKİMİ : … DÜŞÜNCESİ : Temyiz isteminin reddi ile usul ve yasaya uygun olan Bölge İdare Mahkemesi kararının onanması gerektiği düşünülmektedir. TÜRK MİLLETİ ADINA Karar veren Danıştay Onikinci Dairesince; Tetkik Hâkiminin açıklamaları dinlendikten ve dosyadaki belgeler incelendikten sonra, 2577 sayılı İdari Yargılama Usulü Kanunu'nun 17. maddesinin ikinci fıkrası uyarınca davacının duruşma istemi yerinde görülmeyerek gereği görüşüldü: HUKUKİ DEĞERLENDİRME: Bölge idare mahkemelerinin nihai kararlarının temyizen bozulması, 2577 sayılı İdari Yargılama Usulü Kanunu'nun 49. maddesinde yer alan sebeplerden birinin varlığı hâlinde mümkündür. Temyizen incelenen karar usul ve hukuka uygun olup, dilekçede ileri sürülen temyiz nedenleri kararın bozulmasını gerektirecek nitelikte görülmemiştir. KARAR SONUCU: Açıklanan nedenlerle; 1. Davacının temyiz isteminin reddine, 2. Davanın yukarıda özetlenen gerekçeyle reddine ilişkin İdare Mahkemesi kararına karşı yapılan istinaf başvurusunun reddi yolundaki temyize konu ... Bölge İdare Mahkemesi ... İdari Dava Dairesinin ... tarih ve E:…, K:… sayılı kararının ONANMASINA, 3. Temyiz giderlerinin istemde bulunan üzerinde bırakılmasına, 4. 2577 sayılı İdari Yargılama Usulü Kanunu'nun 50. maddesi uyarınca, bu onama kararının taraflara tebliğini ve bir örneğinin de ... Bölge İdare Mahkemesi ... İdari Dava Dairesine gönderilmesini teminen dosyanın … İdare Mahkemesine gönderilmesine, 19/10/2021 tarihinde kesin olarak oybirliğiyle karar verildi."

result = find_similars(document_embeddings, corpus, text)
print(text)
print("-----------")

for i in result:
    print(i)
    print("------------------")
