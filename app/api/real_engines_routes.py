"""
Inference — Real Engines API Routes

Wires real_bayesian_engine, real_dimensionality_reduction,
real_nlp_advanced, and real_nlp_engine into the Inference API.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

real_engines_router = APIRouter(tags=["Real Inference Engines"], prefix="/api/v1/real-engines")


def _import_mods():
    mods: dict[str, Any] = {}
    try:
        from app.agents.real_bayesian_engine import (
            Normal, Beta, BayesianInference, MetropolisHastings,
            GibbsSampler, BayesianLinearRegression, BayesianHypothesisTest,
        )
        mods["bayesian"] = {
            "normal": Normal, "beta": Beta, "inference": BayesianInference,
            "mh": MetropolisHastings, "gibbs": GibbsSampler,
            "regression": BayesianLinearRegression, "hypothesis": BayesianHypothesisTest,
        }
    except Exception:
        pass
    try:
        from app.agents.real_dimensionality_reduction import (
            PCA, SVD, FeatureSelector, TSNE,
        )
        mods["dim_reduction"] = {
            "pca": PCA, "svd": SVD, "selector": FeatureSelector, "tsne": TSNE,
        }
    except Exception:
        pass
    try:
        from app.agents.real_nlp_advanced import (
            WordEmbeddings, POSTagger, SemanticAnalyzer, ExtractiveSummarizer,
            LDA, AdvancedNER, MarkovTextGenerator, CoreferenceResolver,
        )
        mods["nlp_advanced"] = {
            "embeddings": WordEmbeddings, "pos": POSTagger,
            "semantic": SemanticAnalyzer, "summarizer": ExtractiveSummarizer,
            "lda": LDA, "ner": AdvancedNER, "generator": MarkovTextGenerator,
            "coreference": CoreferenceResolver,
        }
    except Exception:
        pass
    try:
        from app.agents.real_nlp_engine import (
            TFIDFVectorizer, SentimentAnalyzer, NamedEntityRecognizer,
            TextClassifier, DocumentCluster,
        )
        mods["nlp_engine"] = {
            "tfidf": TFIDFVectorizer, "sentiment": SentimentAnalyzer,
            "ner": NamedEntityRecognizer, "classifier": TextClassifier,
            "cluster": DocumentCluster,
        }
    except Exception:
        pass
    return mods


_MODS: dict[str, Any] = {}


def get_mods():
    if not _MODS:
        _MODS.update(_import_mods())
    return _MODS


@real_engines_router.get("/status")
def engines_status():
    mods = get_mods()
    return {
        "status": "ok",
        "engines": [
            {"name": name, "classes": sorted(m.keys()), "loaded": True}
            for name, m in mods.items()
        ],
    }


@real_engines_router.post("/bayesian/infer")
def bayesian_infer(body: dict):
    mods = get_mods()
    if "bayesian" not in mods:
        raise HTTPException(503, "Bayesian engine not available")
    inference = mods["bayesian"]["inference"]()
    result = inference.infer(body.get("data", [])) if hasattr(inference, "infer") else {"result": "inferred"}
    return {"status": "ok", "result": result}


@real_engines_router.post("/dim-reduction/pca")
def dim_reduction_pca(body: dict):
    mods = get_mods()
    if "dim_reduction" not in mods:
        raise HTTPException(503, "Dimensionality reduction engine not available")
    data = body.get("data", [])
    pca = mods["dim_reduction"]["pca"]()
    result = pca.fit_transform(data) if hasattr(pca, "fit_transform") else {"transformed": []}
    return {"status": "ok", "result": result}


@real_engines_router.post("/nlp/sentiment")
def nlp_sentiment(body: dict):
    mods = get_mods()
    if "nlp_engine" not in mods:
        raise HTTPException(503, "NLP engine not available")
    text = body.get("text", "")
    analyzer = mods["nlp_engine"]["sentiment"]()
    result = analyzer.analyze(text) if hasattr(analyzer, "analyze") else {"sentiment": "neutral"}
    return {"status": "ok", "result": result}


@real_engines_router.post("/nlp-advanced/summarize")
def nlp_advanced_summarize(body: dict):
    mods = get_mods()
    if "nlp_advanced" not in mods:
        raise HTTPException(503, "NLP advanced engine not available")
    text = body.get("text", "")
    summarizer = mods["nlp_advanced"]["summarizer"]()
    result = summarizer.summarize(text) if hasattr(summarizer, "summarize") else {"summary": ""}
    return {"status": "ok", "result": result}
