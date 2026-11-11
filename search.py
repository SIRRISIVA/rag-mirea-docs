import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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


def scores_tfidf(query, vectorizer=vec, matrix=X):
    q = vectorizer.transform([query.lower()])
    return (matrix @ q.T).toarray().ravel()


def scores_tfidf_char(query):
    return scores_tfidf(query, vec_char, X_char)


def search(query, vectorizer=vec, matrix=X, k=5):
    scores = scores_tfidf(query, vectorizer, matrix)
    top = np.argsort(-scores)[:k]
    return [(chunks[i], scores[i]) for i in top]


# --- семантический поиск: эмбеддинги ---------------------------------------
model = SentenceTransformer("intfloat/multilingual-e5-small")

CACHE = Path("emb_cache.npy")
if CACHE.exists() and np.load(CACHE).shape[0] == len(texts):
    E = np.load(CACHE)
else:
    E = model.encode(["passage: " + t for t in texts],
                     normalize_embeddings=True, show_progress_bar=True)
    np.save(CACHE, E)


def scores_emb(query):
    q = model.encode(["query: " + query], normalize_embeddings=True)
    return (E @ q.T).ravel()


def search_emb(query, k=5):
    scores = scores_emb(query)
    top = np.argsort(-scores)[:k]
    return [(chunks[i], scores[i]) for i in top]


# --- гибридный поиск: min-max нормализация ----------------------------------
def minmax(s):
    mi = s.min()
    ma = s.max()
    if ma - mi < 1e-9:
        return np.zeros_like(s)
    return (s - mi) / (ma - mi)


def search_hybrid(query, k=5, alpha=0.3):
    s_lex = minmax(scores_tfidf_char(query))
    s_sem = minmax(scores_emb(query))
    hybrid_scores = alpha * s_lex + (1.0 - alpha) * s_sem
    top = np.argsort(-hybrid_scores)[:k]
    return [(chunks[i], hybrid_scores[i]) for i in top]


# --- гибридный поиск: Reciprocal Rank Fusion (RRF) --------------------------
def search_rrf(query, k=5, c=60):
    s_lex = scores_tfidf_char(query)
    s_sem = scores_emb(query)

    order_lex = np.argsort(-s_lex)
    order_sem = np.argsort(-s_sem)

    ranks_lex = np.empty(len(chunks), dtype=int)
    ranks_lex[order_lex] = np.arange(1, len(chunks) + 1)

    ranks_sem = np.empty(len(chunks), dtype=int)
    ranks_sem[order_sem] = np.arange(1, len(chunks) + 1)

    rrf_scores = (1.0 / (c + ranks_lex)) + (1.0 / (c + ranks_sem))
    top = np.argsort(-rrf_scores)[:k]
    return [(chunks[i], rrf_scores[i]) for i in top]


# --- оценка качества -------------------------------------------------------
def recall_at_k(search_fn, k=5):
    """Доля вопросов, у которых нужный документ попал в топ-k выдачи."""
    hits = 0
    for item in QUESTIONS:
        results = search_fn(item["q"], k=k)
        sources = [ch["source"] for ch, score in results]
        if item["doc"] in sources:
            hits += 1
    return hits / len(QUESTIONS)


def mrr(search_fn, k=10):
    """Mean Reciprocal Rank по документам."""
    total = 0.0
    for item in QUESTIONS:
        results = search_fn(item["q"], k=k)
        found = False
        for pos, (ch, score) in enumerate(results):
            if ch["source"] == item["doc"]:
                total += 1.0 / (pos + 1)
                found = True
                break
        if not found:
            total += 0.0
    return total / len(QUESTIONS)


METHODS = {
    "TF-IDF слова":   lambda q, k: search(q, vec, X, k),
    "TF-IDF символы": lambda q, k: search(q, vec_char, X_char, k),
    "Эмбеддинги e5":  search_emb,
    "Гибрид Min-Max": lambda q, k: search_hybrid(q, k, alpha=0.3),
    "Гибрид RRF":     search_rrf,
}

if __name__ == "__main__":
    print(f"{'Метод':<20}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR@10':>10}")
    print("-" * 60)
    for name, fn in METHODS.items():
        r1 = recall_at_k(fn, 1)
        r3 = recall_at_k(fn, 3)
        r5 = recall_at_k(fn, 5)
        m = mrr(fn, 10)
        print(f"{name:<20}{r1:>10.3f}{r3:>10.3f}{r5:>10.3f}{m:>10.3f}")

    print("\n--- Подбор alpha для Min-Max (шаг 0.1) ---")
    best_alpha, best_mrr = 0.0, 0.0
    for alpha in np.arange(0, 1.05, 0.1):
        fn = lambda q, k: search_hybrid(q, k, alpha=alpha)
        r1 = recall_at_k(fn, 1)
        m = mrr(fn, 10)
        if m > best_mrr:
            best_mrr = m
            best_alpha = alpha
        print(f"alpha = {alpha:.1f}: recall@1 = {r1:.3f}, MRR = {m:.3f}")
    print(f"Лучший alpha = {best_alpha:.1f} (MRR = {best_mrr:.3f})")
