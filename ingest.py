from pathlib import Path
from pypdf import PdfReader
from collections import Counter
import re

def load_docs(folder="data"):
    files = list(Path(folder).glob("*.pdf"))

    docs = []
    for f in files:
        reader = PdfReader(f)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts)
        docs.append({"source": f.name, "text": text})

    return docs


def clean(text=""):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, size=800, overlap=150):
    chunks = []
    step = size - overlap                     # на сколько сдвигается окно
    for start in range(0, len(text), step):
        chunks.append(text[start:start+size])    # срез от start длиной size
    return chunks


def split_sentences(text):
    return re.split(r'(?<=[.;!?])\s+', text)

def chunk_sentences(text, size=800, overlap_sents=1):
    sents = split_sentences(text)
    chunks = []
    current = []

    for s in sents:
        candidate = " ".join(current + [s])
        if current and len(candidate) > size:
            chunks.append(" ".join(current))
            current = current[-overlap_sents:]
        current.append(s)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(size=800, overlap_sents=1):
    chunks = []
    docs = load_docs()
    for doc in docs:
        text = clean(doc["text"])
        for i, text_chunk in enumerate(chunk_sentences(text, size=size, overlap_sents=overlap_sents)):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": text_chunk})
    return chunks

if __name__ == "__main__":
    chunks = build_chunks()
    print(len(chunks))
    counts = Counter(ch["source"] for ch in chunks)
    print(counts)
    print([len(ch["text"]) for ch in chunks[:5]])
    print(chunks[len(chunks) // 2])