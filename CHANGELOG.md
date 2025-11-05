# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-04

### Added
- **Arquitectura modular agnóstica**: Soporte para múltiples proveedores de embeddings
- `EmbeddingProvider` clase base abstracta para proveedores personalizados
- `AzureOpenAIProvider` para Azure OpenAI (mantiene compatibilidad con v0.1.0)
- `OpenAIProvider` para OpenAI directo (no Azure)
- `CustomEmbeddingProvider` para proveedores personalizados (Hugging Face, Cohere, etc.)
- Ejemplo `multiple_providers.py` demostrando uso de diferentes proveedores
- Tests actualizados para nueva arquitectura modular

### Changed
- **BREAKING**: `VectorPath` ahora acepta un parámetro `provider` (EmbeddingProvider)
- Los parámetros legacy (`api_key`, `azure_endpoint`, etc.) siguen funcionando pero están deprecated
- La función `create_vector_path()` ahora acepta un parámetro `provider`
- Output de `get_path_info()` ahora muestra "Proveedor" en lugar de "Modelo"

### Deprecated
- Parámetros de inicialización legacy en `VectorPath.__init__`:
  - `model`, `api_key`, `azure_endpoint`, `api_version`
  - Usar `provider` parameter en su lugar
- Estos parámetros seguirán funcionando para compatibilidad hacia atrás

### Migration Guide v0.1.0 → v0.2.0

#### Antes (v0.1.0):
```python
router = VectorPath(
    paths=paths,
    model="text-embedding-3-small",
    api_key="key",
    azure_endpoint="https://..."
)
```

#### Ahora (v0.2.0 - Recomendado):
```python
from vector_path.providers import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    model="text-embedding-3-small",
    api_key="key",
    azure_endpoint="https://..."
)
router = VectorPath(paths=paths, provider=provider)
```

#### O usar OpenAI directo:
```python
from vector_path.providers import OpenAIProvider

provider = OpenAIProvider(model="text-embedding-3-small")
router = VectorPath(paths=paths, provider=provider)
```

## [0.1.0] - 2025-11-04

### Added
- Initial release of vector-path library
- Core `VectorPath` class for semantic routing
- `Path` dataclass for defining semantic routes
- `PathChoice` dataclass for routing results
- Azure OpenAI embeddings integration
- Cosine similarity-based matching
- Top-k score aggregation for robust routing
- Configurable similarity thresholds
- Dynamic path addition support
- Batch embedding generation for efficiency
- Comprehensive test suite with mocked API calls
- Example scripts for basic and advanced usage
- Full type hints throughout the codebase
- Detailed documentation in README.md

### Features
- Semantic routing based on Azure OpenAI embeddings
- Support for custom metadata on paths
- Verbose evaluation mode for debugging
- Helper function `create_vector_path()` for quick setup
- Environment variable configuration support

[0.1.0]: https://github.com/yourusername/vector-path/releases/tag/v0.1.0
