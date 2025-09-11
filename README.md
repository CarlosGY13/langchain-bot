# 🥤 Chatbot AJE Group - Asistente Virtual Inteligente

Un chatbot avanzado para AJE Group que combina **RAG (Retrieval-Augmented Generation)** con **identificación visual de productos** usando Google Gemini 2.5 Flash. Responde preguntas sobre estrategia de internacionalización y productos AJE.

## 🚀 Características Principales

### 💬 **Conversación Inteligente**
- **Google Gemini 2.5 Flash**: Modelo de última generación
- **RAG Automático**: Consulta documentos PDF de estrategia empresarial
- **Memoria de Conversación**: Mantiene contexto completo del chat
- **Respuestas Naturales**: Conversación fluida como asistente real

### 📸 **Identificación Visual de Productos**
- **Sube imágenes** y obtén identificación automática
- **15+ productos AJE** en la base de datos
- **Tolerancia de volumen** ±100ml para variaciones de etiquetado
- **Análisis con IA**: Extrae características visuales automáticamente

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
Crear archivo `.env` en la raíz del proyecto:
```bash
# Obligatorio - Obtener en https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=tu_google_api_key_aqui

# Opcional - Si no tienes, usa HuggingFace automáticamente
OPENAI_API_KEY=tu_openai_api_key_aqui
```

### 5. Agregar documentos (opcional)
```bash
mkdir data
# Colocar archivos PDF sobre estrategia de AJE en la carpeta data/
```

## 🚀 Ejecución

### 🌐 Interfaz Web (Recomendado)

#### Opción 1: Lanzador automático
```bash
python run_streamlit.py
```

#### Opción 2: Streamlit directo
```bash
streamlit run streamlit_app.py
```

#### Opción 3: Puerto personalizado
```bash
streamlit run streamlit_app.py --server.port 8080
```

**Acceder a:** `http://localhost:8501` (o el puerto especificado)

### 💻 Interfaz de Consola
```bash
python app.py
```

## 🎯 Funcionalidades Web

### 💬 **Chat de Texto**
- Escribe preguntas sobre productos o estrategia de AJE
- Presiona **📤** para enviar
- El input se limpia automáticamente
- Historial completo de conversación

### 📸 **Identificación Visual**
- Arrastra y suelta imágenes de productos AJE
- Formatos soportados: PNG, JPG, JPEG (máx. 200MB)
- Identificación automática con información detallada
- Preview temporal que desaparece después del análisis

## 📝 Ejemplos de Uso

### Consultas de Texto
```
👤 Usuario: ¿Cuál es la estrategia de internacionalización de AJE?

🤖 AJE Assistant: ¡Excelente pregunta! La estrategia de AJE Group se centra 
en llegar a la gran mayoría de la población en países emergentes, 
enfocándonos especialmente en consumidores de medianos y bajos recursos...

👤 Usuario: ¿Qué productos tienen?

🤖 AJE Assistant: Nuestro portafolio incluye tres marcas principales:
• Big Cola: Nuestra gaseosa emblema en 350ml, 1000ml y 3000ml
• Sporade: Bebida deportiva en sabores Blueberry, Tropical, Mandarina...
• Bio Amayu: Jugos naturales con sabores como Arándano, Manzana Camu...
```

### Identificación Visual
```
👤 Usuario: [Sube imagen de producto]

🤖 AJE Assistant: 🎉 ¡Producto identificado!

🥤 Producto: Sporade Tropical
📏 Capacidad: 500ml
🏷️ Tipo: Deportiva

👁️ Descripción visual:
Bebida de color rojo/naranja en botella plástica transparente con tapa 
verde. Etiqueta predominantemente negra y verde con 'SPORADE' en letras 
blancas y 'tropical' en franja roja.
```

### 🧠 **Componentes Principales**

#### 1. **RAG (Retrieval-Augmented Generation)**
- **PyPDFLoader**: Extrae texto de PDFs de estrategia
- **RecursiveCharacterTextSplitter**: Chunks de 750 tokens, overlap 150
- **FAISS Vector Store**: Búsqueda semántica optimizada
- **OpenAI/HuggingFace Embeddings**: Fallback automático

#### 2. **Base de Productos**
- **Procesamiento automático**: `process_products.py` analiza imágenes
- **Extracción de metadatos**: Marca, sabor, capacidad desde nombres de archivo
- **Análisis visual con IA**: Colores, envase, características especiales
- **JSON estructurado**: `productos_aje.json` con 15+ productos

#### 3. **Identificación Visual**
- **Google Gemini Vision**: Análisis de imágenes subidas por usuario
- **Matching inteligente**: Compara con base de productos conocidos
- **Tolerancia de volumen**: ±100ml para variaciones de etiquetado
- **Scoring system**: Prioriza coincidencias exactas de marca y sabor

#### 4. **Motor de Decisión IA**
- **Contexto híbrido**: Combina RAG + productos según relevancia
- **Prompt engineering**: Instrucciones para respuestas naturales
- **Sin parsing rígido**: El modelo decide qué información usar
- **Memoria conversacional**: Mantiene contexto de intercambios previos
