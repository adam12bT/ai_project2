"""Persistent Chroma storage; embedding generation stays local and explicit."""
from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(self, path, collection='python_docs'):
        self.client = chromadb.PersistentClient(
            path=str(Path(path).resolve()), settings=Settings(anonymized_telemetry=False))
        self.name = collection
        self.collection = None

    def build(self, chunks, vectors, corpus_sha256, embedder):
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError('Each nonempty corpus chunk needs one embedding.')
        if self.name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(
            self.name, embedding_function=None,
            configuration={'hnsw': {'space': 'cosine', 'num_threads': 2}},
            metadata={'corpus_sha256': corpus_sha256, 'embedder': embedder})
        for start in range(0, len(chunks), 256):
            batch = chunks[start:start + 256]
            self.collection.add(
                ids=[c['id'] for c in batch],
                embeddings=vectors[start:start + len(batch)].tolist(),
                documents=[c['text'] for c in batch],
                metadatas=[{k: c[k] for k in ('page', 'title', 'section', 'url') if k in c}
                           for c in batch])

    def open(self, corpus_sha256, embedder, count):
        try:
            self.collection = self.client.get_collection(self.name, embedding_function=None)
        except chromadb.errors.NotFoundError:
            raise ValueError('Chroma index missing. Run python -m librarian.engine') from None
        metadata = self.collection.metadata or {}
        if (metadata.get('corpus_sha256') != corpus_sha256
                or metadata.get('embedder') != embedder or self.collection.count() != count):
            raise ValueError('Chroma index does not match corpus/model. Run python -m librarian.engine')

    def search(self, vector, candidates, page=None):
        options = {'where': {'page': page}} if page else {}
        result = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=candidates,
            include=['distances'], **options)
        return [(chunk_id, 1.0 - float(distance))
                for chunk_id, distance in zip(result['ids'][0], result['distances'][0])]
