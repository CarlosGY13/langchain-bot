# Chatbot con Google Gemini + RAG + Memoria

Un chatbot inteligente que utiliza Google Gemini 2.5 Flash con capacidades de RAG (Retrieval-Augmented Generation) automático y memoria de conversación. Responde preguntas basándose en documentos PDF y mantiene el contexto de la conversación.

## 🚀 Características Principales

- **Google Gemini 2.5 Flash**: Modelo de última generación de Google
- **RAG Automático**: Integra información de documentos PDF cuando es relevante
- **Memoria de Conversación**: Recuerda el historial completo de la charla
- **Embeddings Inteligentes**: OpenAI primario con fallback a HuggingFace local
- **FAISS Optimizado**: Vector store con cache local para evitar recálculos
- **Respuestas Concisas**: Configurado para respuestas directas y al grano
- **Sin Prefijos**: RAG se activa automáticamente según relevancia

## 📋 Requisitos

- Python 3.8+
- API Key de Google AI (gratuita)
- API Key de OpenAI (opcional, usa HuggingFace como fallback)
- Documentos PDF en carpeta `data/`

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/CarlosGY13/langchain-bot.git
cd langchain-bot
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar API Keys
Crea un archivo `.env` con:
```bash
# Obligatorio
GOOGLE_API_KEY=GOOGLE_KEY

# Opcional (usa HuggingFace si no tienes)
OPENAI_API_KEY=OPENAI_KEY
```

**Obtener claves:**
- Google AI: https://aistudio.google.com/app/apikey (gratis)
- OpenAI: https://platform.openai.com/api-keys (opcional)

### 5. Agregar documentos
```bash
mkdir data
# Colocar archivos PDF en la carpeta data/
```

## 🚀 Uso

### Ejecutar el chatbot
```bash
python app.py
```

### Comandos disponibles
- **Conversación normal**: Escribe cualquier pregunta
- **`memoria`**: Ver historial de conversación
- **`salir`**: Terminar el programa

### Ejemplos de uso

```
Tú: ¿Qué productos tiene AJE?
Gemini: AJE Group produce principalmente Big Cola, Cifrut (jugos), 
        Agua Cielo, Volt (energética) y Sporade (deportiva).
        [Fuente: estrategia.pdf (p.1)]

Tú: ¿Cómo funciona su estrategia?
Gemini: Su estrategia se basa en democratización del consumo, 
        precios accesibles y expansión en mercados emergentes.
        [Fuente: estrategia.pdf (p.3)]

Tú: ¿Cómo estás?
Gemini: Estoy funcionando perfectamente, listo para ayudarte.
```

## 🏗️ Arquitectura

### Sistema RAG Optimizado

1. **Preprocesamiento de PDF**:
   - PyPDFLoader para extracción
   - Chunking: 750 tokens, overlap 150
   - Metadata enriquecida (archivo, página, índice)

2. **Embeddings Inteligentes**:
   - Primario: OpenAI Embeddings
   - Fallback: sentence-transformers/all-MiniLM-L6-v2
   - Embeddings normalizados

3. **Vector Store FAISS**:
   - Cache local en `faiss_index/`
   - Carga rápida en ejecuciones posteriores
   - Búsqueda de similitud con k=3

4. **RAG Automático**:
   - Detección inteligente de relevancia
   - Integración transparente con conversación
   - Citas de fuentes automáticas

### Memoria de Conversación
- Lista simple de intercambios (usuario, bot)
- Contexto de últimos 3 intercambios
- Sin warnings de deprecación

## 📁 Estructura del Proyecto

```
langchain-bot/
├── app.py                  # Chatbot principal
├── data/                   # PDFs para RAG
│   └── *.pdf
├── faiss_index/           # Vector store (generado)
├── .env                   # API keys
├── .gitignore            
├── requirements.txt       # Dependencias
└── README.md             # Este archivo
```

## ⚙️ Configuración Avanzada

### Ajustar chunks de PDF
```python
# En preprocess_pdf()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,      # Cambiar tamaño
    chunk_overlap=150,   # Cambiar overlap
)
```

### Cambiar número de documentos RAG
```python
# En create_rag_chain()
retriever = db.as_retriever(
    search_kwargs={"k": 3}  # Cambiar número
)
```

### Ajustar concisión de respuestas
```python
# En los prompts
"Máximo 4 oraciones"  # Cambiar límite
```

## 🔧 Solución de Problemas

### Error de cuota OpenAI
- **Síntoma**: "insufficient_quota"
- **Solución**: Se activa automáticamente HuggingFace fallback

### No encuentra documentos
- **Síntoma**: "No se encontraron PDFs"
- **Solución**: Agregar archivos PDF a carpeta `data/`

### FAISS no carga
- **Síntoma**: Error cargando vector store
- **Solución**: Se recrea automáticamente

### Respuestas vacías
- **Síntoma**: Gemini no responde
- **Solución**: Verificar API key de Google

**Desarrollado con ❤️ usando Google Gemini, LangChain y FAISS**