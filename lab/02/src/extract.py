from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ai_corpus.json"
OUT_DIR = ROOT / "data" / "processed"


ENTITY_TERMS = {
    "Person": [
        "Geoffrey Hinton",
        "Yann LeCun",
        "Yoshua Bengio",
        "Ashish Vaswani",
        "Noam Shazeer",
        "Niki Parmar",
        "Jakob Uszkoreit",
        "Llion Jones",
        "Aidan Gomez",
        "Lukasz Kaiser",
        "Illia Polosukhin",
        "Jacob Devlin",
        "Ming-Wei Chang",
        "Kenton Lee",
        "Kristina Toutanova",
        "Alex Krizhevsky",
        "Ilya Sutskever",
    ],
    "Paper": [
        "Attention Is All You Need",
        "BERT",
        "AlexNet",
    ],
    "Institution": [
        "University of Toronto",
        "New York University",
        "Meta AI",
        "University of Montreal",
        "Mila",
        "Google Brain",
        "Google",
        "OpenAI",
    ],
    "Concept": [
        "deep learning",
        "neural networks",
        "backpropagation",
        "representation learning",
        "convolutional neural networks",
        "Transformer",
        "self-attention",
        "natural language processing",
        "bidirectional encoder representations from transformers",
        "computer vision",
        "large language models",
        "ImageNet classification",
    ],
    "Year": ["2012", "2017", "2018"],
}


RELATION_PATTERNS = [
    (r"(?P<head>.+?) is a (?P<year>20\d{2}) paper authored by (?P<tail>.+)", "AUTHORED", "Paper", "Person"),
    (r"(?P<head>.+?) is a computer scientist affiliated with (?P<tail>.+)", "AFFILIATED_WITH", "Person", "Institution"),
    (r"(?P<head>.+?) was a research team affiliated with (?P<tail>.+)", "AFFILIATED_WITH", "Institution", "Institution"),
    (r"(?P<head>.+?) proposed (?P<tail>.+)", "PROPOSED", "Person", "Concept"),
    (r"(?P<head>.+?) proposed the (?P<tail>.+)", "PROPOSED", "Paper", "Concept"),
    (r"(?P<head>.+?) develops (?P<tail>.+)", "PROPOSED", "Institution", "Concept"),
    (r"(?P<head>.+?) is related to (?P<tail>.+)", "RELATED_TO", "Concept", "Concept"),
    (r"Researchers at (?P<head>.+?) authored papers on (?P<tail>.+)", "RELATED_TO", "Institution", "Concept"),
]


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    label: str
    source_doc: str


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    rel_type: str
    evidence: str
    source_doc: str


def slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_text(value: str) -> str:
    value = value.strip().strip(".")
    value = value.replace(" and ", ", ")
    value = re.sub(r"\s+", " ", value)
    return value


def split_items(value: str) -> list[str]:
    value = normalize_text(value)
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def entity_type(name: str) -> str | None:
    clean = normalize_text(name)
    for label, terms in ENTITY_TERMS.items():
        if clean in terms:
            return label
    return None


def canonical_entity(name: str) -> str | None:
    clean = normalize_text(name)
    for terms in ENTITY_TERMS.values():
        for term in terms:
            if clean.lower() == term.lower():
                return term
    return None


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def collect_entities_with_spacy(text: str) -> set[str]:
    try:
        import spacy
        from spacy.pipeline import EntityRuler
    except Exception:
        return set()

    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    assert isinstance(ruler, EntityRuler)
    patterns = []
    for label, terms in ENTITY_TERMS.items():
        patterns.extend({"label": label, "pattern": term} for term in terms)
    ruler.add_patterns(patterns)
    return {ent.text for ent in nlp(text).ents}


def collect_entities_by_rules(text: str) -> set[str]:
    found: set[str] = set()
    for terms in ENTITY_TERMS.values():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                found.add(term)
    return found


def resolve_item(value: str, expected_type: str) -> str | None:
    value = normalize_text(value)
    candidates = [value]
    if expected_type == "Concept":
        candidates.extend(
            [
                value.removeprefix("the "),
                value.removesuffix(" architecture"),
                value.removesuffix(" for visual recognition"),
                value.removesuffix(" for ImageNet classification"),
            ]
        )
    for candidate in candidates:
        term = canonical_entity(candidate)
        if term and entity_type(term) == expected_type:
            return term
    return None


def extract_relations(doc_id: str, text: str) -> list[Relation]:
    relations: list[Relation] = []
    for sentence in sentence_split(text):
        clean_sentence = sentence.strip().strip(".")
        for pattern, rel_type, head_type, tail_type in RELATION_PATTERNS:
            match = re.match(pattern, clean_sentence)
            if not match:
                continue
            head = resolve_item(match.group("head"), head_type)
            if not head:
                continue
            tail_text = match.group("tail")
            if rel_type == "AUTHORED":
                paper = head
                year = match.group("year")
                if entity_type(year) == "Year":
                    relations.append(
                        Relation(
                            source=slug(paper),
                            target=slug(year),
                            rel_type="PUBLISHED_IN",
                            evidence=clean_sentence,
                            source_doc=doc_id,
                        )
                    )
                for author in split_items(tail_text):
                    tail = resolve_item(author, tail_type)
                    if tail:
                        relations.append(
                            Relation(slug(tail), slug(paper), rel_type, clean_sentence, doc_id)
                        )
                continue
            for tail_item in split_items(tail_text):
                tail = resolve_item(tail_item, tail_type)
                if tail:
                    relations.append(
                        Relation(slug(head), slug(tail), rel_type, clean_sentence, doc_id)
                    )
    return relations


def main() -> None:
    docs = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}

    for doc in docs:
        doc_id = doc["id"]
        text = doc["text"]
        found_names = collect_entities_with_spacy(text) | collect_entities_by_rules(text)
        for name in sorted(found_names):
            label = entity_type(name)
            if not label:
                continue
            entity_id = slug(name)
            entities.setdefault(entity_id, Entity(entity_id, name, label, doc_id))
        for relation in extract_relations(doc_id, text):
            relations.setdefault((relation.source, relation.target, relation.rel_type), relation)

    with (OUT_DIR / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "label", "source_doc"])
        writer.writeheader()
        for entity in sorted(entities.values(), key=lambda item: (item.label, item.name)):
            writer.writerow(entity.__dict__)

    with (OUT_DIR / "relationships.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "type", "evidence", "source_doc"])
        writer.writeheader()
        for relation in sorted(relations.values(), key=lambda item: (item.rel_type, item.source, item.target)):
            writer.writerow(
                {
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation.rel_type,
                    "evidence": relation.evidence,
                    "source_doc": relation.source_doc,
                }
            )

    with (OUT_DIR / "triples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "relation", "tail"])
        writer.writeheader()
        for relation in sorted(relations.values(), key=lambda item: (item.rel_type, item.source, item.target)):
            writer.writerow({"head": relation.source, "relation": relation.rel_type, "tail": relation.target})

    print(f"nodes={len(entities)} relationships={len(relations)}")


if __name__ == "__main__":
    main()
