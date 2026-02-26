# Data València Agent (Analista de Datos Abiertos de València)

**Data València Agent** es una aplicación web interactiva desarrollada con Streamlit y potenciada por Modelos de Lenguaje Grandes (LLMs) como Google Gemini y Llama 3. Su objetivo es permitir a cualquier usuario explorar el [catálogo de Datos Abiertos del Ayuntamiento de València](https://valencia.opendatasoft.com/pages/home/?flg=es-es) utilizando lenguaje natural.

La aplicación encuentra el dataset más relevante para la consulta del usuario, lo descarga, lo analiza y genera visualizaciones y resúmenes de forma automática, actuando como un analista de datos virtual.

---

## Características Principales

- **Búsqueda Semántica**: Utiliza embeddings de sentencias (`sentence-transformers`) y un índice vectorial (FAISS) para encontrar el dataset más relevante para una consulta en lenguaje natural.
- **Multi-LLM**: Permite cambiar entre diferentes proveedores de LLM (Google Gemini, Llama 3 a través de Groq, codestral a través de Ollama) para la planificación y generación de insights.
- **Análisis Automático de Datos**: Identifica automáticamente tipos de columnas (numéricas, categóricas, geoespaciales, temporales).
- **Generación de Visualizaciones**: El LLM planifica y sugiere los gráficos más adecuados (mapas, barras, líneas, etc.) para responder a la consulta del usuario.
- **Creación de Insights**: Un agente LLM interpreta los gráficos y los datos para generar un resumen ejecutivo en texto.
- **Interfaz Interactiva**: Construida con Streamlit para una experiencia de usuario fluida y conversacional.

---

## 🛠️ Arquitectura y Tecnologías

El proyecto sigue una arquitectura modular basada en agentes, donde cada componente tiene una responsabilidad clara:

- **Frontend**: `Streamlit`
- **Búsqueda y RAG (Retrieval-Augmented Generation)**:
  - **Embeddings**: `sentence-transformers` (modelo `paraphrase-MiniLM-L6-v2`)
  - **Índice Vectorial**: `FAISS`
- **Modelos de Lenguaje (LLMs)**:
  - `Google Gemini` (a través de `google-genai`)
  - `Groq` (a través de `groq`)
  - `codestral` (a través de `ollama`)
- **Análisis y Manipulación de Datos**: `Pandas`, `NumPy`
- **Visualización**: `Plotly Express`
- **Gestión de Dependencias**: `uv`
- **Gestión de Secrets**: `pydantic-settings`

> [!IMPORTANT]
> El proyecto está diseñado para que solo se pueda usar un proveedor de LLM a la vez.
>
> Para configurar el proveedor, se utilizan variables de entorno.

---

## ⚙️ Instalación y Ejecución Local

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### Prerrequisitos

- Python 3.13
- Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. Instalar el gestor de dependencias `uv`

`uv` es un gestor de dependencias moderno y rápido para Python, escrito en Rust.

Puedes instalarlo con pipx:

```bash
pipx install uv
```

También puedes consultar la documentación oficial de [uv](https://docs.astral.sh/uv) para el instalador standalone.

### 3. Instalar las dependencias

Una vez instalado `uv`, puedes instalar las dependencias del proyecto con:

```bash
uv sync
```

Esto creará un entorno virtual `.venv` en la raíz del proyecto e instalará todas las dependencias listadas en `pyproject.toml`.

### 4. Configurar las variables de entorno

El proyecto necesita que se configure un proveedor de LLM para funcionar. Actualmente soporta Ollama (local), Google Gemini y Groq.

Para configurar las variables de entorno, copia el archivo `.env.example` a `.env` y agrega los valores que necesites.

Variables clave (nombres actuales que usa la aplicación):

- LLM_PROVIDER: Selecciona el proveedor (`ollama`, `gemini`, `groq`).
- LLM_MODEL: Modelo a usar en el proveedor seleccionado.
- LLM_PROVIDER_API_KEY: (Opcional) clave API para proveedores que la requieran (Gemini / Groq).
 - OLLAMA_HOST: URL de la instancia de Ollama si usas Ollama (por defecto `http://localhost:11434`).

Se recomiendan usar los siguientes modelos:

- Ollama: `codestral`
- Google Gemini: `gemini-1.5-flash-latest`
- Groq: `llama3-70b-8192`

#### 4.1 Configuración de Ollama (solo si usas Ollama)

Si quieres usar Ollama, primero necesitas instalarlo y configurar tu modelo localmente. Puedes seguir la guía oficial de [Ollama](https://ollama.com/docs/installation) para instalarlo. A continuación, descarga el modelo `codestral` con el siguiente comando:

```bash
ollama pull codestral
```

En tu archivo `.env`, configura las variables, por ejemplo:

```env
LLM_PROVIDER=ollama
LLM_MODEL=codestral
OLLAMA_HOST=http://localhost:11434
```

### 5. Construir el Índice FAISS (Solo la primera vez)

Para que la búsqueda funcione, necesitas crear el índice vectorial localmente. El script está en `scripts/build_index.py` y se encargará de descargar la información de los datasets, generar los embeddings y construir el índice FAISS, que se guardará localmente en `data/`.

Antes de ejecutar el script, crea el directorio `data/` en la raíz del proyecto (el contenido no se sube al repositorio):

```bash
mkdir data
```

Ejecuta el script desde la raíz del proyecto (o dentro del contenedor si despliegas en Docker):

```bash
python scripts/build_index.py
```

El proceso generará los archivos `faiss_metadata.json` y `faiss_opendata_valencia.idx` dentro de `data/`.

#### 💡 Uso

1. Escribe una consulta en lenguaje natural en el campo de texto principal (p. ej.: "¿Dónde hay aparcamientos para bicis?").
2. Haz clic en "Analizar Consulta".
3. El agente buscará el dataset más relevante, lo analizará y te presentará visualizaciones e insights.
4. Puedes realizar preguntas de seguimiento sobre el dataset activo.

#### 📈 Posibles mejoras futuras

- Implementar un sistema de caché más avanzado para los resultados de la API.
- Permitir al usuario seleccionar manualmente un dataset si la búsqueda semántica no es precisa.
- Añadir soporte para más tipos de visualizaciones.
- Mejorar la gestión de memoria para datasets muy grandes.

### 6. Ejecutar la aplicación

¡Ya está todo listo! Inicia la aplicación Streamlit con este comando desde la raíz del proyecto:

```bash
streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en una nueva pestaña de tu navegador.

## Despliegue

Para desplegar la aplicación se proporciona un `Dockerfile` que puedes usar para crear una imagen Docker de la aplicación. Está optimizado para producción, utilizando una imagen base de Python ligera y configurando el entorno de manera eficiente.

Cuando despliegues la aplicación, asegúrate de configurar las variables de entorno necesarias para el proveedor de LLM que hayas elegido y de ejecutar el script `scripts/build_index.py` desde el contenedor para generar el índice FAISS antes de iniciar la aplicación.

## Agradecimientos

- A OpenData València por proporcionar los datos.
- A las comunidades de Streamlit, Hugging Face y FAISS.
- A @vicentcorrecher, creador de [BIaaS](https://github.com/vicentcorrecher/BIaaS), cuyo trabajo fue el punto de partida de esta evolución del proyecto.
