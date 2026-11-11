import os
import time
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

import search

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup / check index
    assert len(search.chunks) > 0, "Chunks index is empty"
    yield

app = FastAPI(
    title="MIREA Documents RAG Retrieval API",
    description="Высокоточный сервис гибридного поиска (Dense E5 + Sparse Char TF-IDF / RRF) по локальным актам РТУ МИРЭА",
    version="1.0.0",
    lifespan=lifespan
)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Поисковый запрос пользователя")
    top_k: int = Field(5, ge=1, le=20, description="Количество возвращаемых фрагментов")
    method: Literal["hybrid", "rrf", "dense", "sparse"] = Field("hybrid", description="Алгоритм поиска")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Вес sparse компонента в min-max гибриде")

class ChunkHit(BaseModel):
    chunk_id: int
    source: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    method: str
    execution_time_ms: float
    total_results: int
    results: List[ChunkHit]

class HealthResponse(BaseModel):
    status: str
    chunks_count: int
    model_name: str
    methods_supported: List[str]

class BenchmarkMetrics(BaseModel):
    method: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_10: float

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        chunks_count=len(search.chunks),
        model_name="intfloat/multilingual-e5-small",
        methods_supported=["hybrid", "rrf", "dense", "sparse"]
    )

@app.post("/api/v1/search", response_model=SearchResponse)
def search_documents(req: SearchRequest):
    t0 = time.perf_counter()
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")

    if req.method == "hybrid":
        raw_results = search.search_hybrid(q, k=req.top_k, alpha=req.alpha)
    elif req.method == "rrf":
        raw_results = search.search_rrf(q, k=req.top_k)
    elif req.method == "dense":
        raw_results = search.search_emb(q, k=req.top_k)
    elif req.method == "sparse":
        raw_results = search.search(q, search.vec_char, search.X_char, k=req.top_k)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported search method")

    latency_ms = (time.perf_counter() - t0) * 1000.0

    hits = [
        ChunkHit(
            chunk_id=ch["chunk_id"],
            source=ch["source"],
            text=ch["text"],
            score=round(float(score), 4)
        )
        for ch, score in raw_results
    ]

    return SearchResponse(
        query=q,
        method=req.method,
        execution_time_ms=round(latency_ms, 2),
        total_results=len(hits),
        results=hits
    )

@app.get("/api/v1/benchmarks", response_model=List[BenchmarkMetrics])
def get_benchmarks():
    methods = {
        "TF-IDF слова":   lambda q, k: search.search(q, search.vec, search.X, k),
        "TF-IDF символы": lambda q, k: search.search(q, search.vec_char, search.X_char, k),
        "Эмбеддинги e5":  search.search_emb,
        "Гибрид Min-Max (alpha=0.5)": lambda q, k: search.search_hybrid(q, k, alpha=0.5),
        "Гибрид RRF":     search.search_rrf,
    }

    report = []
    for name, fn in methods.items():
        report.append(
            BenchmarkMetrics(
                method=name,
                recall_at_1=round(search.recall_at_k(fn, 1), 3),
                recall_at_3=round(search.recall_at_k(fn, 3), 3),
                recall_at_5=round(search.recall_at_k(fn, 5), 3),
                mrr_at_10=round(search.mrr(fn, 10), 3),
            )
        )
    return report
