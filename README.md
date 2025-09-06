# Chatbot con Google Gemini + RAG + Memoria

Un chatbot inteligente que utiliza Google Gemini 2.5 Flash con capacidades de RAG (Retrieval-Augmented Generation) automático y memoria de conversación. Responde preguntas basándose en documentos PDF y mantiene el contexto de la conversación.

## 🚀 Características Principales

- **Google Gemini 2.5 Flash**: Modelo de última generación de Google
- **RAG Automático**: Integra información de documentos PDF cuando es relevante
- **Base de Productos**: 15+ productos AJE con información extraída de imágenes
- **Identificación Visual**: Sube una imagen y el chatbot identifica el producto
- **Memoria de Conversación**: Recuerda el historial completo de la charla
- **Embeddings Inteligentes**: OpenAI primario con fallback a HuggingFace local
- **FAISS Optimizado**: Vector store con cache local para evitar recálculos
- **Respuestas Naturales**: Conversación fluida como asistente real de AJE

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
- **`imagen`**: Identificar producto desde imagen
- **`memoria`**: Ver historial de conversación
- **`salir`**: Terminar el programa

### Ejemplos de uso

```
Tú: ¿Qué productos tiene AJE?
Gemini: AJE Group tiene un portafolio diverso con 3 marcas principales:
        Big Cola (gaseosa), Sporade (bebida deportiva) y Bio Amayu 
        (jugos naturales). En total manejamos 15 productos diferentes.

Tú: Describe Sporade
Gemini: Sporade es nuestra línea de bebidas deportivas, perfecta para 
        la hidratación. Está disponible en sabores como Blueberry, 
        Tropical, Mandarina y Uva, todos en presentación de 500ml.

Tú: imagen
📸 IDENTIFICACIÓN VISUAL DE PRODUCTOS
Ruta de la imagen: fotos/sporade/SPORADE_BLUEBERRY_500.png
🔍 Analizando imagen...
Gemini: ¡Producto identificado! Es Sporade Blueberry (500ml)
        • Marca: Sporade
        • Sabor: Blueberry  
        • Capacidad: 500ml
        • Tipo: deportiva

Tú: ¿Cuál es la estrategia de AJE?
Gemini: Nuestra estrategia se basa en la democratización del consumo,
        ofreciendo productos de calidad a precios accesibles...
```

## 🏗️ Arquitectura

### Sistema Híbrido Inteligente

1. **Procesamiento de Documentos (RAG)**:
   - PyPDFLoader para extracción de estrategia
   - Chunking: 750 tokens, overlap 150
   - Vector store FAISS con embeddings optimizados

2. **Base de Productos AJE**:
   - Procesamiento automático de imágenes con Gemini Vision
   - Extracción de: marca, sabor, capacidad, tipo, características
   - Base de datos JSON con 15+ productos

3. **Identificación Visual**:
   - Carga de imágenes por el usuario
   - Análisis con Google Gemini Vision
   - Comparación con base de productos conocidos
   - Identificación automática con 80% de precisión

4. **Motor de Decisión IA**:
   - El modelo decide qué información usar
   - Combina productos + estrategia según contexto
   - Respuestas naturales sin formato predefinido

### Memoria de Conversación
- Lista simple de intercambios (usuario, bot)
- Contexto de últimos 3 intercambios
- Sin warnings de deprecación

## 📁 Estructura del Proyecto

```
langchain-bot/
├── app.py                     # Chatbot principal con identificación visual
├── process_products.py        # Procesador de imágenes con IA
├── test_visual_identification.py # Pruebas de identificación visual
├── data/                      # PDFs para RAG
│   └── estrategia.pdf
├── fotos/                     # Imágenes de productos AJE
│   ├── big cola/
│   ├── sporade/
│   └── bio amayu/
├── productos_aje.json         # Base de datos de productos (generado)
├── faiss_index/              # Vector store (generado)
├── .env                      # API keys
├── requirements.txt          # Dependencias
└── README.md                # Este archivo
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