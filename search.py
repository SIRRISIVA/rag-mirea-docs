from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from ingest import build_chunks
from eval_set import QUESTIONS

chunks = build_chunks()
texts = [ch["text"] for ch in chunks]

# --- лексический поиск: TF-IDF ---------------------------------------------
vec = TfidfVectorizer()
X = vec.fit_transform(texts)

vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
X_char = vec_char.fit_transform(texts)


def search(query, vec, X, k=5):
    q = vec.transform([query.lower()])
    scores = (X @ q.T).toarray().ravel()
    top = np.argsort(-scores)[:k]
    return [(chunks[i], scores[i]) for i in top]


# --- семантический поиск: эмбеддинги ---------------------------------------
# multilingual-e5-small понимает русский (all-MiniLM из NLP-проекта - английская).
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Кодирование 493 чанков занимает минуту-две, поэтому результат кладём на диск.
# При следующем запуске он просто загрузится. Если число чанков изменилось
# (поменяли размер чанка), кэш устарел - считаем заново.
CACHE = Path("emb_cache.npy")
if CACHE.exists() and np.load(CACHE).shape[0] == len(texts):
    E = np.load(CACHE)
else:
    # e5 обучена с префиксами: база - "passage: ", запрос - "query: ".
    # normalize_embeddings=True делает длину векторов равной 1,
    # поэтому косинус = скалярное произведение (как у TF-IDF).
    E = model.encode(["passage: " + t for t in texts],
                     normalize_embeddings=True, show_progress_bar=True)
    np.save(CACHE, E)


def search_emb(query, k=5):
    q = model.encode(["query: " + query], normalize_embeddings=True)
    scores = (E @ q.T).ravel()
    top = np.argsort(-scores)[:k]
    return [(chunks[i], scores[i]) for i in top]


# --- оценка ----------------------------------------------------------------
def recall_at_k(search_fn, k=5):
    """Доля вопросов, у которых нужный документ попал в топ-k выдачи."""
    hits = 0
    for item in QUESTIONS:
        results = search_fn(item["q"], k=k)
        sources = [ch["source"] for ch, score in results]
        if item["doc"] in sources:
            hits += 1
    return hits / len(QUESTIONS)


METHODS = {
    "TF-IDF слова":   lambda q, k: search(q, vec, X, k),
    "TF-IDF символы": lambda q, k: search(q, vec_char, X_char, k),
    "Эмбеддинги e5":  search_emb,
}

if __name__ == "__main__":
    print(f"{'метод':<16}{'recall@1':>10}{'recall@3':>10}{'recall@5':>10}")
    for name, fn in METHODS.items():
        row = "".join(f"{recall_at_k(fn, k):>10.3f}" for k in (1, 3, 5))
        print(f"{name:<16}{row}")

    print("\n--- пример выдачи эмбеддингов ---")
    for ch, score in search_emb("сколько раз можно пересдавать экзамен", k=3):
        print(round(float(score), 3), ch["source"], ch["chunk_id"])
        print(ch["text"][:200])
        print("---")
