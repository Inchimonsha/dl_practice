import os
import json
import numpy as np

from src.config import embedder


def chunk_text_with_overlap(
        text: str,
        max_tokens: int = 500,
        overlap_tokens: int = 100) -> list[str]:
    """Разбивает текст на чанки с перекрытием по токенам."""

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap_tokens  # сдвиг с перекрытием

    return chunks


def load_embeddings(corpus_path: str = "data/corpus.json"):
    """Загружает эмбеддинги."""

    embeddings_file = "data/embeddings.npy"
    metadata_file = "data/metadata.json"

    if os.path.isfile(embeddings_file) and os.path.isfile(metadata_file):
        # Загружаем существующие эмбеддинги
        embeddings = np.load(embeddings_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"Загружено эмбеддингов: {len(embeddings)}")
    else:
        # Генерируем новые эмбеддинги
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        embeddings = []
        metadata = []

        for data in corpus:
            text = data.get("full_text", "")
            message_number = data.get("message_number")
            chunks = chunk_text_with_overlap(text)

            for idx, chunk in enumerate(chunks):
                embedding_vector = embedder.embed_query(chunk)
                embeddings.append(embedding_vector)
                metadata.append({
                    "message_number": message_number,
                    "chunk_index": idx,
                    "text": chunk,
                })

        # Сохраняем эмбеддинги и метаданные
        np.save(embeddings_file, np.array(embeddings))
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return np.array(embeddings), metadata


def generate_embeddings(corpus_path: str = "data/corpus.json"):
    """Генерирует эмбеддинги."""

    embeddings_file = "data/embeddings.npy"
    metadata_file = "data/metadata.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    embeddings = []
    metadata = []

    for data in corpus:
        full_text = data.get("full_text", "")
        message_number = data.get("message_number")
        # category = data.get("category")
        # subcategory = data.get("subcategory")
        # metadata = data.get("metadata", {})
        # buttons = metadata.get("buttons", [])

        chunks = chunk_text_with_overlap(
            full_text,
            max_tokens=500,
            overlap_tokens=100
        )

        for idx, chunk in enumerate(chunks):
            embedding_vector = embedder.embed_query(chunk)
            embeddings.append(embedding_vector)
            metadata.append({
                "message_number": message_number,
                "chunk_index": idx,
                "text": chunk,
            })

    np.save(embeddings_file, np.array(embeddings))

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return embeddings, metadata
