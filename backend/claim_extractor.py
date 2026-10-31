import re


NON_CLAIM_STARTERS = (
    "image ",
    "figure ",
    "table ",
    "what are ",
    "how do ",
    "why do ",
    "where do ",
    "when do ",
    "note:",
    "references",
    "contents"
)


def extract_claims(answer, max_claims=10):

    if not answer or not answer.strip():
        return []

    # Remove common answer headings
    answer = re.sub(
        r"(?im)^relevant information found in the retrieved evidence:\s*",
        "",
        answer
    )

    # Remove markdown headings
    answer = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        answer
    )

    # Remove note lines
    answer = re.sub(
        r"(?im)^note:.*$",
        "",
        answer
    )

    # Remove image captions
    answer = re.sub(
        r"(?im)^image\s*\d+\s*:.*$",
        "",
        answer
    )

    # Remove incomplete image-caption fragments
    answer = re.sub(
        r"(?i)image\s*\d+\s*:\s*[^.\n]*",
        "",
        answer
    )

    # Remove citation artifacts
    answer = re.sub(
        r"\[[^\]]*\]",
        "",
        answer
    )

    # Split into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+|\n+',
        answer
    )

    claims = []
    seen = set()

    for sentence in sentences:

        sentence = sentence.strip()

        sentence = sentence.strip(
            ' "\'“”‘’'
        )

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        if not sentence:
            continue

        if len(sentence.split()) < 6:
            continue

        # Remove question-like sentences
        if sentence.endswith("?"):
            continue

        # Remove non-claim headings and fragments
        if sentence.lower().startswith(
            NON_CLAIM_STARTERS
        ):
            continue

        # Remove sentences containing image-caption artifacts
        if re.search(
            r"(?i)image\s*\d+\s*:",
            sentence
        ):
            continue

        # Remove incomplete fragments
        if sentence.startswith(
            ("...", "[...", "##")
        ):
            continue

        # Remove duplicate sentences
        normalized = sentence.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        claims.append(sentence)

        if len(claims) >= max_claims:
            break

    return claims


if __name__ == "__main__":

    sample_answer = """
    ## Volcanic activity

    Image 10: subducting tectonic plate

    Volcanic mountains form when magma and volcanic debris
    erupt onto the Earth's surface.

    Mountains form through tectonic plate movements.

    What are the different types of mountains?

    Block faulting can create steep mountain ranges.
    """

    results = extract_claims(sample_answer)

    print("\nExtracted Claims:")

    for number, claim in enumerate(results, start=1):
        print(f"{number}. {claim}")