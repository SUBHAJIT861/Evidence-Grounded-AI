import json
from pathlib import Path


def create_source(title, url, content):
    return {
        "title": title,
        "url": url,
        "content": content
    }


def save_sources(sources):
    file_path = Path("data/sources.json")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(sources, file, indent=4, ensure_ascii=False)


def load_sources():
    file_path = Path("data/sources.json")

    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":

    source = create_source(
        "Sample Science Source",
        "https://example.com/science",
        "Evidence-based information about science."
    )

    save_sources([source])

    saved_data = load_sources()

    print("Saved Sources:")
    print(saved_data)