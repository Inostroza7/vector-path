"""
Vector Path - Path Semántico Agnóstico con Múltiples Proveedores

Una librería ligera de enrutamiento semántico que soporta múltiples proveedores de embeddings
(Azure OpenAI, OpenAI, y proveedores personalizados) para emparejar inteligentemente
consultas de usuarios con paths predefinidos basándose en similitud semántica.

Example con Azure OpenAI (por defecto):
    >>> from vector_path import VectorPath, Path
    >>>
    >>> paths = [
    ...     Path(name="soporte", utterances=["ayúdame", "necesito asistencia"]),
    ...     Path(name="facturacion", utterances=["factura", "pago", "costo"])
    ... ]
    >>>
    >>> router = VectorPath(paths=paths)
    >>> result = router("Necesito ayuda con mi factura")
    >>> print(result.name)  # 'soporte' o 'facturacion' según similitud

Example con OpenAI:
    >>> from vector_path import VectorPath, Path
    >>> from vector_path.providers import OpenAIProvider
    >>>
    >>> provider = OpenAIProvider(api_key="sk-...", model="text-embedding-3-small")
    >>> router = VectorPath(paths=paths, provider=provider)

Example con proveedor personalizado:
    >>> from vector_path.providers import CustomEmbeddingProvider
    >>> import numpy as np
    >>>
    >>> def my_embed(text: str) -> np.ndarray:
    ...     # Tu implementación
    ...     return np.random.rand(384)
    >>>
    >>> provider = CustomEmbeddingProvider(
    ...     embed_fn=my_embed,
    ...     batch_fn=lambda texts: np.array([my_embed(t) for t in texts]),
    ...     dimension=384,
    ...     model_name="my-model"
    ... )
    >>> router = VectorPath(paths=paths, provider=provider)
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__license__ = "MIT"

from vector_path.core import (
    VectorPath,
    Path,
    PathChoice,
    create_vector_path,
)

from vector_path.providers import (
    EmbeddingProvider,
    AzureOpenAIProvider,
    OpenAIProvider,
    CustomEmbeddingProvider,
)

__all__ = [
    # Core classes
    "VectorPath",
    "Path",
    "PathChoice",
    "create_vector_path",
    # Providers
    "EmbeddingProvider",
    "AzureOpenAIProvider",
    "OpenAIProvider",
    "CustomEmbeddingProvider",
    # Metadata
    "__version__",
]
