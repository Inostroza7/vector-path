# vector-path

Una librería ligera de Python para enrutamiento semántico **agnóstica de proveedor**. Soporta múltiples proveedores de embeddings (Azure OpenAI, OpenAI, custom) para enrutar consultas de usuario a paths predefinidos basándose en similitud semántica.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Características

- 🎯 **Enrutamiento Semántico**: Empareja consultas con paths basándose en significado, no en palabras clave exactas
- 🔌 **Múltiples Proveedores**: Soporta Azure OpenAI, OpenAI, y proveedores personalizados
- 🚀 **Fácil de Usar**: API simple con configuración mínima requerida
- ⚡ **Eficiente**: Generación de embeddings en batch y cálculos de similitud optimizados
- 🔧 **Flexible**: Umbrales configurables, scoring top-k, y adición dinámica de paths
- 🧪 **Bien Probado**: Suite de tests comprensiva con llamadas API simuladas
- 📦 **Type Safe**: Type hints completos para mejor soporte de IDE
- 🔄 **Retrocompatible**: Los parámetros legacy de v0.1.0 siguen funcionando

## Instalación

### Desde el Código Fuente

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/vector-path.git
cd vector-path

# Instalar en modo desarrollo
pip install -e .

# O instalar con dependencias de desarrollo
pip install -e ".[dev]"
```

### Desde PyPI (cuando se publique)

```bash
pip install vector-path
```

## Inicio Rápido

### 1. Configurar Credenciales de Azure OpenAI

Crea un archivo `.env` en la raíz de tu proyecto:

```bash
AZURE_OPENAI_API_KEY='tu-api-key'
AZURE_OPENAI_ENDPOINT='https://tu-recurso.openai.azure.com/'
AZURE_OPENAI_API_VERSION='2024-02-01'
AZURE_OPENAI_MODEL='text-embedding-3-small'
```

### 2. Crear un VectorPath Simple

```python
from vector_path import VectorPath, Path

# Definir paths con utterances de ejemplo
paths = [
    Path(
        name="soporte",
        utterances=[
            "necesito ayuda",
            "soporte técnico",
            "tengo un problema"
        ],
        description="Consultas de soporte al cliente"
    ),
    Path(
        name="facturacion",
        utterances=[
            "pregunta sobre factura",
            "problema de pago",
            "costo de suscripción"
        ],
        description="Consultas de facturación y pagos"
    ),
    Path(
        name="ventas",
        utterances=[
            "información de precios",
            "quiero comprar",
            "adquirir producto"
        ],
        description="Consultas de ventas"
    )
]

# Crear el path
router = VectorPath(
    paths=paths,
    score_threshold=0.7,  # Score mínimo de similitud (0-1)
    top_k=5              # Considerar las 5 utterances más similares
)

# Enrutar una consulta
result = router("Tengo problemas con mi cuenta")

if result.name:
    print(f"Path emparejado: {result.name}")
    print(f"Confianza: {result.score:.2%}")
else:
    print("No se encontró ningún path")
```

### 3. Evaluar con Salida Detallada

```python
# Obtener salida verbosa con todos los scores de paths
result = router.evaluate_query(
    "¿Cuánto cuesta el plan premium?",
    verbose=True
)
```

Salida:
```
🔍 Query: '¿Cuánto cuesta el plan premium?'
📈 Scores por path:
  ✅ ventas: 0.8234
     facturacion: 0.7123
     soporte: 0.5891

🎯 Path seleccionado: 'ventas' (score: 0.8234)
```

## Múltiples Proveedores de Embeddings

vector-path v0.2.0+ soporta múltiples proveedores de embeddings. Elige el que mejor se adapte a tus necesidades:

### Azure OpenAI (Por Defecto)

```python
from vector_path import VectorPath, Path
from vector_path.providers import AzureOpenAIProvider

# Opción 1: Usar proveedores explícitamente (recomendado)
provider = AzureOpenAIProvider(
    model="text-embedding-3-small",
    # api_key, azure_endpoint se toman de .env automáticamente
)
router = VectorPath(paths=paths, provider=provider)

# Opción 2: Compatibilidad legacy (aún funciona)
router = VectorPath(paths=paths, model="text-embedding-3-small")
```

### OpenAI Directo

```python
from vector_path.providers import OpenAIProvider

provider = OpenAIProvider(
    model="text-embedding-3-small",
    api_key="sk-..."  # O desde OPENAI_API_KEY en .env
)
router = VectorPath(paths=paths, provider=provider)
```

### Proveedor Personalizado

Puedes usar cualquier servicio de embeddings (Hugging Face, Cohere, modelos locales, etc.):

```python
from vector_path.providers import CustomEmbeddingProvider
import numpy as np

# Tu función de embedding custom
def my_embedding_function(text: str) -> np.ndarray:
    # Aquí usarías tu modelo (ej: sentence-transformers)
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # return model.encode(text)
    return np.random.rand(384)  # Ejemplo simplificado

def batch_function(texts: list) -> np.ndarray:
    return np.array([my_embedding_function(t) for t in texts])

provider = CustomEmbeddingProvider(
    embed_fn=my_embedding_function,
    batch_fn=batch_function,
    dimension=384,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

router = VectorPath(paths=paths, provider=provider)
```

Ver [examples/multiple_providers.py](examples/multiple_providers.py) para más ejemplos.

## Uso Avanzado

### Adición Dinámica de Paths

```python
# Agregar un nuevo path en tiempo de ejecución
new_path = Path(
    name="reembolsos",
    utterances=["cancelar suscripción", "obtener reembolso", "devolución de dinero"]
)
router.add_path(new_path)
```

### Configuración Personalizada

```python
from vector_path import create_vector_path

# Usar la función helper para configuración rápida
router = create_vector_path(
    paths=paths,
    score_threshold=0.75,
    model="text-embedding-3-large"  # Usar un modelo de embedding diferente
)
```

### Usando Metadata de Path

```python
paths = [
    Path(
        name="soporte_vip",
        utterances=["ayuda urgente", "asistencia VIP"],
        metadata={
            "priority": "high",
            "sla": "1 hour",
            "department": "premium_support"
        }
    )
]

router = VectorPath(paths=paths)
result = router("Necesito ayuda urgente")

if result.name:
    matched_path = next(p for p in router.paths if p.name == result.name)
    print(f"Prioridad: {matched_path.metadata['priority']}")
    print(f"SLA: {matched_path.metadata['sla']}")
```

## Configuración

### Parámetros de VectorPath

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `paths` | `List[Path]` | Requerido | Lista de objetos Path definiendo rutas |
| `model` | `str` | `"text-embedding-3-small"` | Modelo de embedding de Azure OpenAI |
| `score_threshold` | `float` | `0.7` | Score mínimo de similitud (0-1) para emparejar |
| `top_k` | `int` | `5` | Número de utterances top para promediar en scoring |
| `api_key` | `str` | `None` | API key de Azure (usa variable de entorno si no se provee) |
| `azure_endpoint` | `str` | `None` | Endpoint de Azure (usa variable de entorno si no se provee) |
| `api_version` | `str` | `None` | Versión de API (usa variable de entorno si no se provee) |

### Variables de Entorno

- `AZURE_OPENAI_API_KEY`: Tu API key de Azure OpenAI (requerido)
- `AZURE_OPENAI_ENDPOINT`: URL del endpoint de Azure OpenAI (requerido)
- `AZURE_OPENAI_API_VERSION`: Versión de API (por defecto: `"2024-02-01"`)
- `AZURE_OPENAI_MODEL`: Nombre del modelo de embedding (por defecto: `"text-embedding-3-small"`)

## Ejemplos

Revisa el directorio [examples/](examples/) para ejemplos de uso más detallados:

- [basic_usage.py](examples/basic_usage.py) - Enrutamiento simple con paths predefinidos
- [advanced_usage.py](examples/advanced_usage.py) - Paths dinámicos, configuración personalizada y metadata

Ejecutar ejemplos:
```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Desarrollo

### Configurar Entorno de Desarrollo

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/vector-path.git
cd vector-path

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar con dependencias de desarrollo
pip install -e ".[dev]"
```

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=vector_path --cov-report=html

# Ejecutar archivo de test específico
pytest tests/test_core.py
```

### Calidad de Código

```bash
# Formatear código con black
black vector_path tests examples

# Lint con ruff
ruff check vector_path tests examples

# Type check con mypy
mypy vector_path
```

## Cómo Funciona

1. **Generación de Embeddings**: Al inicializarse, el router genera embeddings para todas las utterances de todos los paths usando la API de embeddings de Azure OpenAI.

2. **Procesamiento de Consultas**: Cuando llega una consulta, se convierte a un embedding usando el mismo modelo.

3. **Cálculo de Similitud**: El sistema calcula la similitud de coseno entre el embedding de la consulta y todos los embeddings de utterances.

4. **Agregación de Scores**: Para cada path, toma el promedio de las top-k utterances más similares (por defecto k=5).

5. **Emparejamiento por Umbral**: Si el score agregado más alto excede el umbral (por defecto 0.7), ese path es retornado. De lo contrario, no hay match.

Este enfoque hace el enrutamiento más robusto que el emparejamiento de una sola utterance y maneja variaciones en la entrada del usuario de manera elegante.

## Referencia de API

### Clases

#### `Path`
Representa una ruta semántica con utterances de ejemplo.

**Atributos:**
- `name` (str): Identificador único para el path
- `utterances` (List[str]): Frases de ejemplo que deben enrutar a este path
- `description` (Optional[str]): Descripción legible
- `metadata` (Dict[str, Any]): Metadata personalizada adicional

#### `PathChoice`
Resultado de una operación de enrutamiento.

**Atributos:**
- `name` (Optional[str]): Nombre del path emparejado (None si no hay match)
- `score` (float): Score de confianza del match (0-1)
- `scores` (Dict[str, float]): Scores para todos los paths evaluados

#### `VectorPath`
Clase principal del router semántico.

**Métodos:**
- `__call__(query: str) -> PathChoice`: Enruta una consulta a un path
- `evaluate_query(query: str, verbose: bool = True) -> PathChoice`: Enruta con salida detallada
- `add_path(path: Path)`: Agrega un nuevo path dinámicamente
- `get_path_info() -> str`: Obtiene información sobre los paths configurados

### Funciones

#### `create_vector_path(paths, score_threshold=0.7, model="text-embedding-3-small")`
Función helper para crear rápidamente una instancia de VectorPath.

## Contribuir

¡Las contribuciones son bienvenidas! Siéntete libre de enviar un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Agradecimientos

- Construido con [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- Inspirado por patrones de enrutamiento semántico en aplicaciones modernas de IA

## Soporte

Para problemas, preguntas o contribuciones, por favor abre un issue en GitHub.
