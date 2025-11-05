"""
Ejemplo de uso de múltiples proveedores de embeddings con vector-path.

Este ejemplo demuestra cómo usar diferentes proveedores:
- Azure OpenAI (por defecto)
- OpenAI directo
- Proveedor personalizado
"""

from vector_path import VectorPath, Path
from vector_path.providers import (
    AzureOpenAIProvider,
    OpenAIProvider,
    CustomEmbeddingProvider,
)
import numpy as np
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


def example_azure_openai():
    """Ejemplo usando Azure OpenAI (por defecto)."""
    print("\n" + "=" * 60)
    print("Ejemplo 1: Azure OpenAI Provider")
    print("=" * 60)

    paths = [
        Path(
            name="politica",
            utterances=[
                "¿qué opinas del presidente?",
                "pensamientos sobre el gobierno",
                "¿qué piensas sobre las elecciones?"
            ]
        ),
        Path(
            name="charla_casual",
            utterances=[
                "¿cómo está el clima hoy?",
                "¿qué tal?",
                "¿cómo estás?"
            ]
        )
    ]

    # Forma 1: Dejar que use Azure por defecto (compatibilidad hacia atrás)
    print("\n📌 Opción A: Usar Azure OpenAI implícitamente (legacy)")
    router = VectorPath(paths=paths, score_threshold=0.7)
    print(router.get_path_info())

    # Forma 2: Especificar explícitamente el provider de Azure
    print("\n📌 Opción B: Especificar Azure OpenAI explícitamente")
    azure_provider = AzureOpenAIProvider(
        model="text-embedding-3-small"
        # api_key, azure_endpoint, api_version se toman de .env
    )
    router2 = VectorPath(paths=paths, provider=azure_provider, score_threshold=0.7)

    result = router2.evaluate_query("¿cómo está el clima?", verbose=True)


def example_openai():
    """Ejemplo usando OpenAI directo (no Azure)."""
    print("\n" + "=" * 60)
    print("Ejemplo 2: OpenAI Provider (Directo)")
    print("=" * 60)

    paths = [
        Path(
            name="soporte",
            utterances=["necesito ayuda", "problema técnico", "error"]
        ),
        Path(
            name="ventas",
            utterances=["quiero comprar", "precio", "información de producto"]
        )
    ]

    # Usar OpenAI directo en lugar de Azure
    openai_provider = OpenAIProvider(
        # api_key se toma de variable de entorno OPENAI_API_KEY
        model="text-embedding-3-small"
    )

    router = VectorPath(paths=paths, provider=openai_provider, score_threshold=0.7)
    print(router.get_path_info())

    # Nota: Esto requiere tener configurado OPENAI_API_KEY en .env
    print("\n⚠️ Nota: Este ejemplo requiere OPENAI_API_KEY en .env")
    print("Si no está configurado, el ejemplo fallará.")

    # Descomentar para probar:
    # result = router.evaluate_query("¿cuánto cuesta?", verbose=True)


def example_custom_provider():
    """Ejemplo usando un proveedor personalizado."""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Custom Provider (Embeddings Aleatorios)")
    print("=" * 60)

    # Simular un proveedor custom con embeddings determinísticos
    # En la práctica, esto podría ser Hugging Face, Cohere, etc.

    def custom_embed_function(text: str) -> np.ndarray:
        """
        Función custom de embedding (ejemplo simplificado).

        En un caso real, aquí llamarías a tu modelo custom,
        por ejemplo: sentence-transformers, Cohere, etc.
        """
        # Para demostración, usamos un embedding "determinístico" basado en longitud
        embedding = np.random.rand(384)

        # Hacer que textos similares tengan embeddings similares
        # (esto es solo para demo, en realidad usarías un modelo real)
        if "ayuda" in text.lower() or "problema" in text.lower():
            embedding[0:10] = 0.9  # Alta similitud en primeras dimensiones
        elif "precio" in text.lower() or "comprar" in text.lower():
            embedding[10:20] = 0.9
        else:
            embedding[20:30] = 0.9

        return embedding

    def custom_batch_function(texts: list) -> np.ndarray:
        """Función batch para múltiples textos."""
        return np.array([custom_embed_function(text) for text in texts])

    # Crear el proveedor personalizado
    custom_provider = CustomEmbeddingProvider(
        embed_fn=custom_embed_function,
        batch_fn=custom_batch_function,
        dimension=384,
        model_name="demo-random-embeddings"
    )

    paths = [
        Path(
            name="soporte",
            utterances=["necesito ayuda", "tengo un problema", "error en sistema"]
        ),
        Path(
            name="ventas",
            utterances=["quiero comprar", "cuál es el precio", "información de productos"]
        )
    ]

    router = VectorPath(paths=paths, provider=custom_provider, score_threshold=0.6)
    print(router.get_path_info())

    # Probar el routing
    result = router.evaluate_query("tengo un problema con mi cuenta", verbose=True)


def example_switching_providers():
    """Ejemplo comparando resultados entre diferentes proveedores."""
    print("\n" + "=" * 60)
    print("Ejemplo 4: Comparar Proveedores")
    print("=" * 60)

    paths = [
        Path(name="tecnico", utterances=["bug", "error", "falla"]),
        Path(name="general", utterances=["hola", "ayuda", "consulta"])
    ]

    providers = {
        "Azure OpenAI": AzureOpenAIProvider(model="text-embedding-3-small"),
        # "OpenAI": OpenAIProvider(model="text-embedding-3-small"),  # Requiere API key
    }

    query = "tengo un error en el sistema"

    for name, provider in providers.items():
        print(f"\n🔍 Probando con {name}:")
        router = VectorPath(paths=paths, provider=provider, score_threshold=0.6)
        result = router(query)
        print(f"   Query: '{query}'")
        print(f"   Resultado: {result.name} (score: {result.score:.4f})")


def main():
    """Ejecutar todos los ejemplos."""
    print("\n" + "🚀 " + "=" * 58)
    print("    Vector-Path: Ejemplos de Múltiples Proveedores")
    print("=" * 60)

    try:
        # Ejemplo 1: Azure OpenAI
        example_azure_openai()

        # Ejemplo 2: OpenAI directo (comentado por defecto, requiere OPENAI_API_KEY)
        # example_openai()

        # Ejemplo 3: Proveedor personalizado
        example_custom_provider()

        # Ejemplo 4: Comparar proveedores
        # example_switching_providers()

        print("\n" + "=" * 60)
        print("✅ Ejemplos completados!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplos: {e}")
        print("Asegúrate de tener configurado .env con credenciales de Azure OpenAI.")


if __name__ == "__main__":
    main()
