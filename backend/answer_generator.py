import re


STOP_WORDS = {
    "who", "what", "when", "where", "why", "how",
    "is", "are", "was", "were", "the", "a", "an",
    "of", "in", "on", "at", "to", "for", "and",
    "or", "do", "does", "did", "tell", "me", "about",
    "it", "this", "that", "these", "those", "with",
    "from", "by", "as", "be", "has", "have", "had",
    "can", "could", "would", "should", "also"
}

BOILERPLATE_TERMS = [
    "disambiguation",
    "table of contents",
    "skip to content",
    "jump to navigation",
    "main menu",
    "privacy policy",
    "terms of use",
    "for other uses",
    "part of a series",
    "references",
    "external links",
    "see also"
]


def clean_text(text):

    if not text:
        return ""

    # Remove Markdown headings
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text
    )

    # Remove image and figure captions
    text = re.sub(
        r"(?im)^\s*(image|figure)\s*\d*\s*:.*$",
        "",
        text
    )

    # Remove bracketed missing-content markers
    text = re.sub(
        r"\[\s*\.{2,}\s*\]",
        " ",
        text
    )

    text = re.sub(
        r"\[\s*\.\.\.",
        " ",
        text
    )

    text = re.sub(
        r"\.{3,}",
        " ",
        text
    )

    # Remove remaining Markdown formatting
    text = re.sub(r"[*_`#]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def get_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z]{2,}\b",
        text.lower()
    )

    return {
        word for word in words
        if word not in STOP_WORDS
    }


def calculate_relevance(question, text):

    question_words = get_keywords(question)
    sentence_words = get_keywords(text)

    if not question_words or not sentence_words:
        return 0.0

    common = question_words.intersection(sentence_words)

    return len(common) / len(question_words)


def is_boilerplate(sentence):

    lower_sentence = sentence.lower()

    if any(
        term in lower_sentence
        for term in BOILERPLATE_TERMS
    ):
        return True

    if sentence.count("|") >= 2:
        return True

    if sentence.count("›") >= 2:
        return True

    if len(sentence.split()) > 45:
        return True

    return False


def is_duplicate(sentence, existing_sentences):

    words = get_keywords(sentence)

    if not words:
        return True

    for existing in existing_sentences:

        existing_words = get_keywords(existing)

        if sentence.lower() == existing.lower():
            return True

        union = words.union(existing_words)

        if not union:
            continue

        similarity = (
            len(words.intersection(existing_words))
            / len(union)
        )

        if similarity >= 0.75:
            return True

    return False


def extract_relevant_sentences(
    question,
    text,
    existing_sentences,
    max_sentences=2
):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    candidates = []

    for sentence in sentences:

        sentence = normalize_text(sentence)

        if len(sentence.split()) < 8:
            continue

        if sentence.endswith("?"):
            continue

        if is_boilerplate(sentence):
            continue

        score = calculate_relevance(
            question,
            sentence
        )

        if score < 0.34:
            continue

        if is_duplicate(
            sentence,
            existing_sentences
        ):
            continue

        candidates.append((score, sentence))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    for score, sentence in candidates:

        if len(selected) >= max_sentences:
            break

        if is_duplicate(
            sentence,
            existing_sentences + selected
        ):
            continue

        selected.append(sentence)

    return selected


def generate_answer(question, evidence):

    if not evidence:

        return {
            "answer": (
                "I don't have enough evidence "
                "to answer this question."
            ),
            "sources": []
        }

    ranked_evidence = []

    for item in evidence:

        text = clean_text(
            item.get("text", "")
        )

        if not text:
            continue

        score = calculate_relevance(
            question,
            text
        )

        if score >= 0.34:

            ranked_evidence.append({
                **item,
                "text": text,
                "relevance_score": score
            })

    ranked_evidence.sort(
        key=lambda item: item["relevance_score"],
        reverse=True
    )

    answer_parts = []
    sources = []
    used_sentences = []
    used_sources = set()

    for item in ranked_evidence:

        if len(answer_parts) >= 5:
            break

        sentences = extract_relevant_sentences(
            question,
            item["text"],
            used_sentences,
            max_sentences=2
        )

        if not sentences:
            continue

        for sentence in sentences:

            answer_parts.append(sentence)
            used_sentences.append(sentence)

            if len(answer_parts) >= 5:
                break

        source_key = (
            item.get("url", ""),
            item.get("title", ""),
            item.get("page", "")
        )

        if source_key not in used_sources:

            used_sources.add(source_key)

            sources.append({
                "title": item.get(
                    "title",
                    "Unknown Source"
                ),
                "url": item.get("url", ""),
                "page": item.get("page", ""),
                "source_type": item.get(
                    "source_type",
                    ""
                ),
                "relevance_score": item[
                    "relevance_score"
                ]
            })

    if not answer_parts:

        return {
            "answer": (
                "I don't have enough relevant evidence "
                "to answer this question."
            ),
            "sources": []
        }

    answer = (
        "Relevant information found in the retrieved evidence:\n\n"
        + "\n\n".join(answer_parts)
        + "\n\nNote: This answer is based on retrieved evidence. "
        "Its claims are checked separately by the verification module."
    )

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    sample_question = "How are mountains formed?"

    sample_evidence = [
        {
            "title": "Sample Mountain Source",
            "url": "https://example.com/mountains",
            "page": "Web",
            "source_type": "web",
            "text": """
            Image 10: subducting tectonic plate [...]
            Mountains form through the movement of tectonic plates.
            ## Volcanic activity
            Volcanic mountains form when magma erupts
            onto the Earth's surface.
            What are the different types of mountains?
            """
        }
    ]

    result = generate_answer(
        sample_question,
        sample_evidence
    )

    print("\nGenerated Answer:\n")
    print(result["answer"])

    print("\nSources:\n")
    print(result["sources"])