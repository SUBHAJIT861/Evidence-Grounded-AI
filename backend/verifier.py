import re

from sentence_transformers import CrossEncoder

from backend.embeddings import (
    load_embedding_model,
    create_embeddings
)


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_model = None
_embedding_model = None


def load_verification_model():
    global _model

    if _model is None:
        print("Loading NLI verification model...")
        _model = CrossEncoder(MODEL_NAME)

    return _model


def load_semantic_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = load_embedding_model()

    return _embedding_model


def split_evidence_into_sentences(evidence):

    sentence_evidence = []

    for item in evidence:

        text = item.get("text", "").strip()

        if not text:
            continue

        text = re.sub(
            r"(?i)table of contents|ask anything|skip to content",
            " ",
            text
        )

        text = re.sub(
            r"(?im)^image\s*\d+\s*:.*$",
            " ",
            text
        )

        sentences = re.split(
            r'(?<=[.!?])\s+|\n+',
            text
        )

        for sentence in sentences:

            sentence = re.sub(
                r"\s+",
                " ",
                sentence
            ).strip()

            if len(sentence.split()) < 6:
                continue

            if re.match(
                r"(?i)^(image|figure|table of contents|references)\b",
                sentence
            ):
                continue

            if sentence.count("|") >= 2:
                continue

            result = item.copy()
            result["text"] = sentence

            sentence_evidence.append(result)

    return sentence_evidence


def retrieve_relevant_evidence(
    claim,
    evidence,
    top_k=5,
    min_similarity=0.35
):

    sentence_evidence = split_evidence_into_sentences(
        evidence
    )

    if not sentence_evidence:
        return []

    embedding_model = load_semantic_model()

    claim_embedding = create_embeddings(
        [claim],
        embedding_model
    )

    evidence_embeddings = create_embeddings(
        [
            item["text"]
            for item in sentence_evidence
        ],
        embedding_model
    )

    similarities = (
        evidence_embeddings @ claim_embedding[0]
    )

    ranked = []

    for item, similarity in zip(
        sentence_evidence,
        similarities
    ):

        similarity = float(similarity)

        if similarity >= min_similarity:

            result = item.copy()
            result["semantic_similarity"] = similarity

            ranked.append(result)

    ranked.sort(
        key=lambda item: item["semantic_similarity"],
        reverse=True
    )

    return ranked[:top_k]


def verify_claim(
    claim,
    evidence,
    threshold=0.70,
    min_similarity=0.35
):

    if not claim or not claim.strip() or not evidence:

        return {
            "claim": claim,
            "status": "Unsupported",
            "reason": "Claim or evidence is missing.",
            "confidence": 0.0,
            "evidence": "",
            "page": "",
            "scores": {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 0.0
            }
        }

    relevant_evidence = retrieve_relevant_evidence(
        claim,
        evidence,
        top_k=5,
        min_similarity=min_similarity
    )

    if not relevant_evidence:

        return {
            "claim": claim,
            "status": "Unsupported",
            "reason": "No sufficiently relevant evidence found.",
            "confidence": 0.0,
            "evidence": "",
            "page": "",
            "scores": {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 0.0
            }
        }

    model = load_verification_model()

    pairs = [
        [item["text"], claim]
        for item in relevant_evidence
    ]

    predictions = model.predict(
        pairs,
        apply_softmax=True
    )

    labels = model.model.config.id2label

    results = []

    for item, prediction in zip(
        relevant_evidence,
        predictions
    ):

        scores = {
            str(labels[i]).lower(): float(score)
            for i, score in enumerate(prediction)
        }

        results.append({
            "evidence": item,
            "entailment": scores.get("entailment", 0.0),
            "contradiction": scores.get("contradiction", 0.0),
            "neutral": scores.get("neutral", 0.0)
        })

    # Evaluate each evidence sentence independently.
    supported_items = [
        r for r in results
        if r["entailment"] >= threshold
        and r["entailment"] > r["contradiction"]
    ]

    contradicted_items = [
        r for r in results
        if r["contradiction"] >= threshold
        and r["contradiction"] > r["entailment"]
    ]

    max_entailment = max(
        r["entailment"] for r in results
    )

    max_contradiction = max(
        r["contradiction"] for r in results
    )

    max_neutral = max(
        r["neutral"] for r in results
    )

    # Select the strongest evidence.
    best = max(
        results,
        key=lambda r: max(
            r["entailment"],
            r["contradiction"],
            r["neutral"]
        )
    )

    # Determine final status.
    if supported_items and contradicted_items:

        status = "Partially Supported"
        reason = (
            "Different relevant evidence sentences "
            "provide supporting and contradicting signals."
        )

        best = max(
            supported_items + contradicted_items,
            key=lambda r: max(
                r["entailment"],
                r["contradiction"]
            )
        )

    elif supported_items:

        status = "Supported"
        reason = "Relevant evidence supports the claim."

        best = max(
            supported_items,
            key=lambda r: r["entailment"]
        )

    elif contradicted_items:

        status = "Contradicted"
        reason = "Relevant evidence contradicts the claim."

        best = max(
            contradicted_items,
            key=lambda r: r["contradiction"]
        )

    else:

        status = "Unsupported"
        reason = (
            "Available evidence does not sufficiently "
            "establish or contradict the claim."
        )

    selected = best["evidence"]

    confidence = max(
        best["entailment"],
        best["contradiction"],
        best["neutral"]
    )

    return {
        "claim": claim,
        "status": status,
        "reason": reason,
        "confidence": round(confidence, 4),
        "evidence": selected.get("text", ""),
        "page": selected.get("page", ""),
        "title": selected.get("title", ""),
        "url": selected.get("url", ""),
        "semantic_similarity": round(
            selected.get("semantic_similarity", 0.0),
            4
        ),
        "scores": {
            "entailment": round(max_entailment, 4),
            "contradiction": round(max_contradiction, 4),
            "neutral": round(max_neutral, 4)
        }
    }


if __name__ == "__main__":

    sample_evidence = [
        {
            "page": "Web",
            "title": "Mountain Formation",
            "text": (
                "Mountains can form when tectonic plates collide. "
                "This pressure causes the Earth's crust to fold. "
                "Fault-block mountains include the Sierra Nevada."
            )
        }
    ]

    test_claims = [
        "Mountains can form when tectonic plates collide.",
        "The Earth's crust can fold under pressure.",
        "The Sierra Nevada is an example of fault-block mountains."
    ]

    for claim in test_claims:

        result = verify_claim(
            claim,
            sample_evidence
        )

        print("\n------------------------------")
        print("Claim:", result["claim"])
        print("Status:", result["status"])
        print("Confidence:", result["confidence"])
        print("Reason:", result["reason"])
        print("Evidence:", result.get("evidence", ""))