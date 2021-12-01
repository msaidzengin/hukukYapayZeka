import string, json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    explanations = [preprocess(i) for i in texts]
    documents = pd.DataFrame(explanations, columns=["explanations"])
    sbert_model = SentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens")
    document_embeddings = sbert_model.encode(documents["explanations"])
    pairwise_similarities = cosine_similarity(document_embeddings)
    return pairwise_similarities

def find_similars(pairwise_similarities, text, n=5, threshold=0.5):
    similarities = pairwise_similarities[text]
    similarities = similarities.reshape(-1)
    similarities = similarities[similarities > threshold]
    similarities = similarities[:n]
    return similarities

with open('kararlar.json', encoding='utf-8') as f:
    kararlar = json.load(f)

kararlar = [y for x,y in kararlar.items()][:10]
pairwise_similarities = create_vectors(kararlar)
print(pairwise_similarities)
find_similars(pairwise_similarities, "karar")

#text1 = 'Machine learning is the study of computer algorithms that improve automatically through experience. Machine learning algorithms build a mathematical model based on sample data, known as training data. The discipline of machine learning employs various approaches to teach computers to accomplish tasks where no fully satisfactory algorithm is available.'
#text2 = 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.'
#print(find_similarity(text1, text2))
