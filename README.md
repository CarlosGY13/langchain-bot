# Chatbot Estrategia de Internacionalización

Un chatbot especializado que responde preguntas sobre la estrategia de internacionalización empresarial utilizando LangChain, embeddings locales de Hugging Face y RAG (Retrieval-Augmented Generation).

## 🚀 Características

- **RAG (Retrieval-Augmented Generation)**: Respuestas basadas en documentos PDF
- **100% Gratuito**: Sin APIs externas, todo funciona localmente
- **Embeddings Locales**: Usando modelos de Hugging Face
- **Interfaz Web Moderna**: Desarrollada con Streamlit
- **Búsqueda Inteligente**: Encuentra información relevante en el documento

## 📋 Requisitos

- Python 3.8+
- PDF de estrategia de internacionalización
- Conexión a internet (solo para descargar modelos la primera vez)

## 🛠️ Instalación

1. **Clonar el repositorio:**
```bash
git clone <tu-repositorio>
cd langchain-bot
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Preparar datos:**
- Colocar el PDF de estrategia en `data/estrategia.pdf`

5. **Configuración opcional:**
```bash
# Copiar archivo de ejemplo de variables de entorno
copy env_example.txt .env
# Editar .env si necesitas configuración personalizada
```

## 🚀 Uso

1. **Generar embeddings (solo la primera vez):**
```bash
python ingest.py
```

2. **Ejecutar el chatbot:**
```bash
streamlit run app.py
```

3. **Abrir en navegador:**
El chatbot estará disponible en `http://localhost:8501`

## 📊 Estructura del Proyecto

```
langchain-bot/
├── app.py              # Aplicación principal del chatbot
├── ingest.py           # Procesamiento de PDF y generación de embeddings
├── requirements.txt    # Dependencias del proyecto
├── README.md          # Este archivo
├── env_example.txt    # Ejemplo de variables de entorno
├── data/
│   └── estrategia.pdf # PDF de estrategia de internacionalización
└── faiss_index/       # Índice de embeddings (generado automáticamente)
```

## 💡 Ejemplos de Preguntas

### Sobre Estrategia:
- ¿Cuál es la estrategia de internacionalización?
- ¿Cómo funciona el modelo de negocio?
- ¿Cuáles son los pilares estratégicos?
- ¿En qué países opera la empresa?
- ¿Cuáles son los factores de éxito?
- ¿Qué mercados son objetivos prioritarios?
- ¿Cómo se realiza la expansión internacional?

## 🔧 Tecnologías Utilizadas

- **LangChain**: Framework para aplicaciones de IA
- **Hugging Face**: Modelos de embeddings y lenguaje
- **FAISS**: Biblioteca para búsqueda de similitud
- **Streamlit**: Interfaz web
- **Sentence Transformers**: Embeddings locales
- **Transformers**: Modelos de lenguaje

## ⚙️ Configuración Avanzada

### Variables de Entorno (Opcional)

Crea un archivo `.env` basado en `env_example.txt`:

```bash
# Configuración opcional para Hugging Face
HUGGINGFACE_API_KEY=tu_api_key_aqui

# Configuración de modelos
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-medium

# Configuración de búsqueda
SEARCH_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Cambiar Modelo de Embeddings

En `ingest.py` y `app.py`, puedes cambiar:
```python
model_name="sentence-transformers/all-MiniLM-L6-v2"
```

### Cambiar Modelo de Lenguaje

En `app.py`, puedes cambiar:
```python
model_name = "microsoft/DialoGPT-medium"
```

### Ajustar Parámetros de Búsqueda

```python
retriever=db.as_retriever(search_kwargs={"k": 3})
```

## 🔍 Solución de Problemas

### Error: "No module named 'langchain_community'"
```bash
pip install langchain-community langchain-huggingface
```

### Error: "No se encontró el archivo data/estrategia.pdf"
- Asegúrate de colocar el PDF en la carpeta `data/`
- El archivo debe llamarse exactamente `estrategia.pdf`

### Error: "CUDA out of memory"
- Los modelos se ejecutan en CPU por defecto
- Si tienes GPU y quieres usarla, modifica `device='cuda'` en el código

### Error: "Model download failed"
- Verifica tu conexión a internet
- Los modelos se descargan automáticamente la primera vez
- Puede tardar varios minutos dependiendo de tu conexión

### Error: "FAISS index not found"
- Ejecuta primero: `python ingest.py`
- Asegúrate de que el PDF esté en la carpeta `data/`

## 📝 Notas Importantes

- **100% Gratuito**: No se requieren APIs externas
- **Primera ejecución**: Puede tardar más tiempo debido a la descarga de modelos
- **Modelos locales**: Se descargan automáticamente la primera vez
- **Espacio en disco**: Asegúrate de tener suficiente espacio para los modelos (~2GB)
- **Memoria RAM**: Se recomienda al menos 4GB de RAM disponible

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
