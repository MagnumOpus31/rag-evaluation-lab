import math
import re
from collections import Counter


class LightweightRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

        self.documents = [
            self._tokenize(chunk)
            for chunk in chunks
        ]

        self.document_frequency = Counter()

        for tokens in self.documents:
            for token in set(tokens):
                self.document_frequency[token] += 1

        self.num_documents = len(self.documents)

        self.idf = {
            token: math.log(
                (self.num_documents + 1)
                / (frequency + 1)
            ) + 1
            for token, frequency in self.document_frequency.items()
        }

        self.vectors = [
            self._create_vector(tokens)
            for tokens in self.documents
        ]

    def _tokenize(self, text):
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    def _create_vector(self, tokens):
        counts = Counter(tokens)

        vector = {}

        for token, count in counts.items():
            if token in self.idf:
                vector[token] = (
                    (1 + math.log(count))
                    * self.idf[token]
                )

        norm = math.sqrt(
            sum(value * value for value in vector.values())
        )

        if norm > 0:
            vector = {
                token: value / norm
                for token, value in vector.items()
            }

        return vector

    def _similarity(self, query_vector, document_vector):
        if not query_vector or not document_vector:
            return 0.0

        if len(query_vector) > len(document_vector):
            query_vector, document_vector = (
                document_vector,
                query_vector
            )

        return sum(
            value * document_vector.get(token, 0.0)
            for token, value in query_vector.items()
        )

    def retrieve(self, query, top_k=3):
        query_tokens = self._tokenize(query)
        query_vector = self._create_vector(query_tokens)

        scored_results = []

        for index, document_vector in enumerate(self.vectors):
            score = self._similarity(
                query_vector,
                document_vector
            )

            scored_results.append(
                {
                    "text": self.chunks[index],
                    "score": float(score),
                    "index": index
                }
            )

        scored_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return scored_results[:top_k]