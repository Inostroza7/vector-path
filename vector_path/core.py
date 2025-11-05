"""
Vector Path - Path semántico agnóstico con soporte para múltiples proveedores

Enrutador semántico que utiliza embeddings para hacer enrutamiento basado en similitud
semántica. Soporta múltiples proveedores de embeddings (Azure OpenAI, OpenAI, custom).

Autor: Sistema de IA
Fecha: 2025
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

from vector_path.providers import (
    EmbeddingProvider,
    AzureOpenAIProvider,
    OpenAIProvider,
)

# Cargar variables de entorno
load_dotenv()


@dataclass
class Path:
    """
    Representa un path (ruta) semántico con ejemplos de expresiones (utterances).

    Attributes:
        name: Nombre único del path
        utterances: Lista de expresiones de ejemplo que activan este path
        description: Descripción opcional del path
        metadata: Metadata adicional (opcional)
    """
    name: str
    utterances: List[str]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validar que el path tenga al menos una utterance."""
        if not self.utterances:
            raise ValueError(f"Path '{self.name}' debe tener al menos una utterance")


@dataclass
class PathChoice:
    """
    Resultado de una consulta al router.

    Attributes:
        name: Nombre del path seleccionado (None si no hay match)
        score: Score de similitud (0-1)
        scores: Diccionario con scores de todos los paths evaluados
    """
    name: Optional[str] = None
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)


class VectorPath:
    """
    Path semántico agnóstico que soporta múltiples proveedores de embeddings.

    Este enrutador:
    1. Genera embeddings de las utterances de cada path usando el proveedor especificado
    2. Compara queries entrantes usando similitud de coseno
    3. Retorna el path más similar si supera el threshold

    Example con Azure OpenAI:
        >>> from vector_path import VectorPath, Path
        >>> from vector_path.providers import AzureOpenAIProvider
        >>>
        >>> provider = AzureOpenAIProvider(model="text-embedding-3-small")
        >>> router = VectorPath(paths=[politics, chitchat], provider=provider)
        >>> result = router("how's the weather?")
        >>> print(result.name)  # 'chitchat'

    Example con OpenAI:
        >>> from vector_path.providers import OpenAIProvider
        >>>
        >>> provider = OpenAIProvider(api_key="sk-...", model="text-embedding-3-small")
        >>> router = VectorPath(paths=[politics, chitchat], provider=provider)
    """

    def __init__(
        self,
        paths: List[Path],
        provider: Optional[EmbeddingProvider] = None,
        score_threshold: float = 0.7,
        top_k: int = 5,
        # Parámetros legacy para compatibilidad hacia atrás (deprecated)
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Inicializa el enrutador semántico.

        Args:
            paths: Lista de objetos Path
            provider: Proveedor de embeddings (EmbeddingProvider). Si no se proporciona,
                     se crea un AzureOpenAIProvider por defecto usando los parámetros legacy.
            score_threshold: Umbral mínimo de similitud (0-1)
            top_k: Número de utterances más similares a considerar

            Parámetros legacy (deprecated, usar provider en su lugar):
            model: Modelo de embeddings
            api_key: API key del proveedor
            azure_endpoint: Azure endpoint (solo para Azure)
            api_version: API version (solo para Azure)
        """
        self.paths = paths
        self.score_threshold = score_threshold
        self.top_k = top_k

        # Configurar proveedor
        if provider is not None:
            self.provider = provider
        else:
            # Compatibilidad hacia atrás: crear AzureOpenAIProvider por defecto
            self.provider = AzureOpenAIProvider(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                model=model or "text-embedding-3-small",
            )

        # Almacenar embeddings e índice
        self.embeddings: np.ndarray = np.array([])
        self.path_names: List[str] = []
        self.utterances: List[str] = []

        # Inicializar índice de embeddings
        self._build_index()

    def _build_index(self):
        """
        Construye el índice de embeddings para todas las utterances de todos los paths.
        """
        print(f"🔨 Construyendo índice de embeddings usando {self.provider.model_name}...")

        all_utterances = []
        all_path_names = []

        # Recopilar todas las utterances
        for path in self.paths:
            for utterance in path.utterances:
                all_utterances.append(utterance)
                all_path_names.append(path.name)

        # Generar embeddings en batch (más eficiente)
        if all_utterances:
            self.embeddings = self.provider.get_embeddings_batch(all_utterances)
            self.utterances = all_utterances
            self.path_names = all_path_names
            print(f"✅ Índice construido: {len(all_utterances)} utterances de {len(self.paths)} paths")
        else:
            print("⚠️ No se encontraron utterances para indexar")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula la similitud de coseno entre dos vectores.

        Args:
            vec1: Vector 1
            vec2: Vector 2

        Returns:
            Similitud de coseno (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calcula similitudes entre query y todos los embeddings del índice.

        Args:
            query_embedding: Embedding del query

        Returns:
            Array de similitudes
        """
        # Normalizar vectores para similitud de coseno eficiente
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        index_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Producto punto = similitud de coseno para vectores normalizados
        similarities = np.dot(index_norm, query_norm)

        return similarities

    def _aggregate_scores(self, similarities: np.ndarray) -> Dict[str, float]:
        """
        Agrega scores de similitud por path.

        Para cada path, toma el promedio de los top_k utterances más similares.

        Args:
            similarities: Array de similitudes

        Returns:
            Diccionario con scores agregados por path
        """
        path_scores: Dict[str, List[float]] = {}

        # Agrupar similitudes por path
        for path_name, similarity in zip(self.path_names, similarities):
            if path_name not in path_scores:
                path_scores[path_name] = []
            path_scores[path_name].append(float(similarity))

        # Calcular score agregado (promedio de top_k similitudes)
        aggregated = {}
        for path_name, scores in path_scores.items():
            # Ordenar y tomar top_k
            top_scores = sorted(scores, reverse=True)[:self.top_k]
            # Promedio de top scores
            aggregated[path_name] = np.mean(top_scores)

        return aggregated

    def __call__(self, query: str) -> PathChoice:
        """
        Clasifica una query y retorna el path más apropiado.

        Args:
            query: Texto de la consulta

        Returns:
            PathChoice con el path seleccionado o None si no hay match
        """
        if not self.embeddings.size:
            return PathChoice(name=None, score=0.0, scores={})

        # Generar embedding del query
        query_embedding = self.provider.get_embedding(query)

        # Calcular similitudes
        similarities = self._compute_similarities(query_embedding)

        # Agregar scores por path
        path_scores = self._aggregate_scores(similarities)

        # Encontrar el mejor path
        if not path_scores:
            return PathChoice(name=None, score=0.0, scores={})

        best_path = max(path_scores, key=path_scores.get)  # type: ignore
        best_score = path_scores[best_path]

        # Verificar threshold
        if best_score >= self.score_threshold:
            return PathChoice(
                name=best_path,
                score=best_score,
                scores=path_scores
            )
        else:
            return PathChoice(
                name=None,
                score=best_score,
                scores=path_scores
            )

    def add_path(self, path: Path):
        """
        Añade un nuevo path al router y actualiza el índice.

        Args:
            path: Objeto Path a añadir
        """
        self.paths.append(path)
        self._build_index()
        print(f"✅ Path '{path.name}' añadido al router")

    def get_path_info(self) -> str:
        """
        Retorna información sobre los paths configurados.

        Returns:
            String con información de los paths
        """
        info = f"📊 VectorPath con {len(self.paths)} paths:\n"
        for path in self.paths:
            info += f"  • {path.name}: {len(path.utterances)} utterances\n"
        info += f"⚙️ Configuración:\n"
        info += f"  • Proveedor: {self.provider.model_name}\n"
        info += f"  • Threshold: {self.score_threshold}\n"
        info += f"  • Top-K: {self.top_k}\n"
        return info

    def evaluate_query(self, query: str, verbose: bool = True) -> PathChoice:
        """
        Evalúa una query y muestra información detallada.

        Args:
            query: Texto de la consulta
            verbose: Si True, muestra información detallada

        Returns:
            PathChoice con el path seleccionado
        """
        result = self(query)

        if verbose:
            print(f"\n🔍 Query: '{query}'")
            print(f"📈 Scores por path:")
            for path_name, score in sorted(result.scores.items(), key=lambda x: x[1], reverse=True):
                emoji = "✅" if path_name == result.name else "  "
                print(f"  {emoji} {path_name}: {score:.4f}")

            if result.name:
                print(f"\n🎯 Path seleccionado: '{result.name}' (score: {result.score:.4f})")
            else:
                print(f"\n❌ Sin match (mejor score: {result.score:.4f} < threshold: {self.score_threshold})")

        return result


# Funciones helper para crear el path rápidamente
def create_vector_path(
    paths: List[Path],
    provider: Optional[EmbeddingProvider] = None,
    score_threshold: float = 0.7,
    # Legacy parameters para compatibilidad
    model: Optional[str] = None,
) -> VectorPath:
    """
    Función helper para crear un VectorPath de forma rápida.

    Args:
        paths: Lista de paths
        provider: Proveedor de embeddings (opcional)
        score_threshold: Umbral de similitud
        model: Modelo (legacy, solo si provider es None)

    Returns:
        VectorPath configurado
    """
    if provider is None and model is not None:
        # Compatibilidad hacia atrás
        provider = AzureOpenAIProvider(model=model)

    return VectorPath(
        paths=paths,
        provider=provider,
        score_threshold=score_threshold,
    )
