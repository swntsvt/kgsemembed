"""Dense embedding encoder with model-aware prefix handling.

Transforms verbalised entity text into unit-normalised dense vectors using an
already-loaded ``SentenceTransformer``.  Model-specific prompting behaviour is
driven entirely by the embedding model registry: source entities receive the
configured query prefix and candidate entities receive the configured document
prefix.  This module never loads models, performs no file I/O, and contains no
verbalisation logic.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from kgsemembed.embeddings.models import ModelConfig, get_model_config

_SUPPORTED_ROLES = ("source", "candidate")


class EmbeddingEncoder:
    """
    Encode verbalised entity text into unit-normalised dense vectors.

    The encoder applies model-specific prompting conventions sourced from the
    embedding model registry and delegates vectorisation to an injected
    ``SentenceTransformer``.  Source and candidate entities are encoded through
    separate operations to support asymmetric prompting models such as Stella.

    Parameters
    ----------
    model_key : str
        Supported model key, one of ``"M1"`` through ``"M5"``.
    model : SentenceTransformer
        An already-loaded model instance. The encoder never loads models itself.
    """

    def __init__(self, model_key: str, model: SentenceTransformer) -> None:
        self.config: ModelConfig = get_model_config(model_key)
        self.model = model

    def _apply_prefix(self, text: str, role: str) -> str:
        """
        Apply the configured prefix for ``role`` to ``text``.

        Parameters
        ----------
        text : str
            Verbalised entity text.
        role : str
            Either ``"source"`` (query prefix) or ``"candidate"`` (doc prefix).

        Returns
        -------
        str
            The prefixed text, or ``text`` unchanged when no prefix is configured.

        Raises
        ------
        ValueError
            If ``role`` is not one of the supported roles.
        """
        if role == "source":
            prefix = self.config.query_prefix
        elif role == "candidate":
            prefix = self.config.doc_prefix
        else:
            supported = ", ".join(repr(name) for name in _SUPPORTED_ROLES)
            raise ValueError(
                f"Invalid role: {role!r}. Supported roles are: {supported}."
            )
        if prefix is None:
            return text
        return prefix.format(text)

    def _encode_one(self, text: str, role: str) -> np.ndarray:
        """
        Encode a single ``text`` for ``role`` into a 1-D ``float32`` vector.

        Parameters
        ----------
        text : str
            Verbalised entity text.
        role : str
            Either ``"source"`` or ``"candidate"``.

        Returns
        -------
        np.ndarray
            One-dimensional unit-normalised embedding of dtype ``float32``.
        """
        embedding = self.model.encode(
            self._apply_prefix(text, role),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32).ravel()

    def encode_source(self, text: str) -> np.ndarray:
        """
        Encode a source entity into a unit-normalised ``float32`` vector.

        Parameters
        ----------
        text : str
            Verbalised source entity text.

        Returns
        -------
        np.ndarray
            One-dimensional embedding of dtype ``float32``.
        """
        return self._encode_one(text, "source")

    def encode_candidate(self, text: str) -> np.ndarray:
        """
        Encode a candidate entity into a unit-normalised ``float32`` vector.

        Parameters
        ----------
        text : str
            Verbalised candidate entity text.

        Returns
        -------
        np.ndarray
            One-dimensional embedding of dtype ``float32``.
        """
        return self._encode_one(text, "candidate")

    def encode_batch(
        self,
        texts: List[str],
        role: str = "candidate",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of entities into unit-normalised ``float32`` vectors.

        Parameters
        ----------
        texts : List[str]
            Verbalised entity texts to encode under a single ``role``.
        role : str, optional
            Either ``"source"`` or ``"candidate"`` (default ``"candidate"``).
        show_progress : bool, optional
            Whether to display the encoding progress bar (default ``True``).

        Returns
        -------
        np.ndarray
            Two-dimensional array of shape ``(len(texts), embedding_dim)`` and
            dtype ``float32``.
        """
        prefixed = [self._apply_prefix(text, role) for text in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def cosine_similarity_matrix(
        self,
        source_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between unit-normalised embeddings.

        The embeddings are assumed to be unit-normalised, so similarity reduces
        to the dot product and no re-normalisation is performed.

        Parameters
        ----------
        source_embeddings : np.ndarray
            Source embeddings of shape ``(n_source, embedding_dim)``.
        candidate_embeddings : np.ndarray
            Candidate embeddings of shape ``(n_candidate, embedding_dim)``.

        Returns
        -------
        np.ndarray
            Similarity matrix of shape ``(n_source, n_candidate)`` and dtype
            ``float32``.
        """
        source = np.asarray(source_embeddings, dtype=np.float32)
        candidate = np.asarray(candidate_embeddings, dtype=np.float32)
        return np.dot(source, candidate.T).astype(np.float32, copy=False)
