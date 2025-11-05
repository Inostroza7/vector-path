"""
Proveedores de embeddings para VectorPath.

Este módulo define la interfaz abstracta para proveedores de embeddings
y las implementaciones concretas para diferentes servicios.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """
    Clase base abstracta para proveedores de embeddings.

    Los proveedores de embeddings deben implementar esta interfaz
    para ser compatibles con VectorPath.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera un embedding para un texto individual.

        Args:
            text: Texto a embedar

        Returns:
            Vector numpy con el embedding
        """
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para múltiples textos en batch (más eficiente).

        Args:
            texts: Lista de textos a embedar

        Returns:
            Array numpy con los embeddings (shape: [n_texts, embedding_dim])
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Retorna la dimensión de los embeddings producidos por este proveedor.

        Returns:
            Dimensión del embedding
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Retorna el nombre del modelo usado por este proveedor.

        Returns:
            Nombre del modelo
        """
        pass


class AzureOpenAIProvider(EmbeddingProvider):
    """
    Proveedor de embeddings usando Azure OpenAI.

    Example:
        >>> provider = AzureOpenAIProvider(
        ...     api_key="your-key",
        ...     azure_endpoint="https://your-resource.openai.azure.com/",
        ...     model="text-embedding-3-small"
        ... )
        >>> embedding = provider.get_embedding("hello world")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        """
        Inicializa el proveedor de Azure OpenAI.

        Args:
            api_key: Azure OpenAI API key (opcional, usa variable de entorno)
            azure_endpoint: Azure endpoint (opcional, usa variable de entorno)
            api_version: API version (opcional, usa variable de entorno)
            model: Modelo de embeddings de Azure OpenAI
        """
        from openai import AzureOpenAI

        self._model = model
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

        # Dimensiones de embedding por modelo
        self._embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def get_embedding(self, text: str) -> np.ndarray:
        """Genera embedding para un texto usando Azure OpenAI."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self._model
        )
        return np.array(response.data[0].embedding)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para múltiples textos en batch."""
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            input=texts,
            model=self._model
        )
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Retorna la dimensión del embedding."""
        return self._embedding_dims.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Retorna el nombre del modelo."""
        return f"azure/{self._model}"


class OpenAIProvider(EmbeddingProvider):
    """
    Proveedor de embeddings usando OpenAI directamente (no Azure).

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
        >>> embedding = provider.get_embedding("hello world")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        organization: Optional[str] = None,
    ):
        """
        Inicializa el proveedor de OpenAI.

        Args:
            api_key: OpenAI API key (opcional, usa variable de entorno OPENAI_API_KEY)
            model: Modelo de embeddings de OpenAI
            organization: Organization ID (opcional)
        """
        from openai import OpenAI

        self._model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization or os.getenv("OPENAI_ORGANIZATION"),
        )

        # Dimensiones de embedding por modelo
        self._embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def get_embedding(self, text: str) -> np.ndarray:
        """Genera embedding para un texto usando OpenAI."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self._model
        )
        return np.array(response.data[0].embedding)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para múltiples textos en batch."""
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            input=texts,
            model=self._model
        )
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Retorna la dimensión del embedding."""
        return self._embedding_dims.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Retorna el nombre del modelo."""
        return f"openai/{self._model}"


class CustomEmbeddingProvider(EmbeddingProvider):
    """
    Proveedor personalizado de embeddings usando funciones callback.

    Permite usar cualquier servicio de embeddings custom implementando
    funciones simples.

    Example:
        >>> def my_embed_fn(text: str) -> np.ndarray:
        ...     # Tu implementación custom
        ...     return np.random.rand(384)
        >>>
        >>> def my_batch_fn(texts: List[str]) -> np.ndarray:
        ...     return np.array([my_embed_fn(t) for t in texts])
        >>>
        >>> provider = CustomEmbeddingProvider(
        ...     embed_fn=my_embed_fn,
        ...     batch_fn=my_batch_fn,
        ...     dimension=384,
        ...     model_name="my-custom-model"
        ... )
    """

    def __init__(
        self,
        embed_fn,
        batch_fn,
        dimension: int,
        model_name: str = "custom",
    ):
        """
        Inicializa un proveedor personalizado de embeddings.

        Args:
            embed_fn: Función que toma un texto y retorna un embedding (np.ndarray)
            batch_fn: Función que toma una lista de textos y retorna embeddings (np.ndarray)
            dimension: Dimensión de los embeddings
            model_name: Nombre del modelo para identificación
        """
        self._embed_fn = embed_fn
        self._batch_fn = batch_fn
        self._dimension = dimension
        self._model_name = model_name

    def get_embedding(self, text: str) -> np.ndarray:
        """Genera embedding usando la función custom."""
        return self._embed_fn(text)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings en batch usando la función custom."""
        return self._batch_fn(texts)

    @property
    def embedding_dimension(self) -> int:
        """Retorna la dimensión del embedding."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Retorna el nombre del modelo."""
        return f"custom/{self._model_name}"
