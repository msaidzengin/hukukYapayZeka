import string
import pandas as pd
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

def find_similarity(text1, text2):
    
    explanations = [text1, text2]
    explanations = [preprocess(i) for i in explanations]
    documents = pd.DataFrame(explanations, columns=["explanations"])

    sbert_model = SentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens")
    document_embeddings = sbert_model.encode(documents["explanations"])
    pairwise_similarities = cosine_similarity(document_embeddings)
    result = pairwise_similarities[0][1] * 100
    return round(result, 2)


text1 = 'Machine learning is the study of computer algorithms that improve automatically through experience. Machine learning algorithms build a mathematical model based on sample data, known as training data. The discipline of machine learning employs various approaches to teach computers to accomplish tasks where no fully satisfactory algorithm is available.'
text2 = 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.'
print(find_similarity(text1, text2))
