import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import glob
import json
import difflib

def preprocess_pdf(pdf_path):
    """
    Preprocesamiento optimizado del PDF con metadata
    """
    print(f"Procesando PDF: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_documents([page])
        for i, chunk in enumerate(page_chunks):
            chunk.metadata.update({
                "source_file": os.path.basename(pdf_path),
                "page_number": page.metadata.get("page", 0),
                "chunk_index": i,
                "total_chunks": len(page_chunks)
            })
            chunks.append(chunk)
    
    print(f"Creados {len(chunks)} chunks del PDF")
    return chunks

def create_embeddings():
    """
    Configurar embeddings: OpenAI primario, HuggingFace fallback
    """
    try:
        if os.getenv("OPENAI_API_KEY"):
            print("Intentando OpenAI Embeddings...")
            embeddings = OpenAIEmbeddings()
            embeddings.embed_query("test")
            print("OpenAI Embeddings configurado exitosamente")
            return embeddings
        else:
            raise Exception("No OpenAI key")
    except Exception as e:
        if "insufficient_quota" in str(e) or "429" in str(e):
            print("Cuota de OpenAI agotada, usando sentence-transformers (local)")
        else:
            print("Usando sentence-transformers (local)")
        
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def create_or_load_faiss_db():
    """
    Construir o cargar base FAISS optimizada
    """
    faiss_path = "faiss_index"
    embeddings = create_embeddings()
    
    if os.path.exists(faiss_path):
        try:
            print("Cargando base FAISS existente...")
            db = FAISS.load_local(
                faiss_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Base FAISS cargada exitosamente")
            return db, embeddings
        except Exception as e:
            print(f"Error cargando FAISS: {e}. Recreando...")
    
    print("Creando nueva base FAISS...")
    
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("No se encontraron PDFs en la carpeta 'data/'")
        return None, embeddings
    
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = preprocess_pdf(pdf_file)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("No se pudieron procesar documentos")
        return None, embeddings
    
    print(f"Creando base FAISS con {len(all_chunks)} chunks...")
    db = FAISS.from_documents(all_chunks, embeddings)
    
    db.save_local(faiss_path)
    print("Base FAISS guardada exitosamente")
    
    return db, embeddings

def load_products_database(products_file="productos_aje.json"):
    """
    Cargar base de datos de productos AJE
    """
    try:
        if os.path.exists(products_file):
            with open(products_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Base de productos cargada: {data['metadata']['total_productos']} productos")
                return data['productos']
        else:
            print(f"Archivo {products_file} no encontrado. Ejecuta 'python process_products.py' primero.")
            return []
    except Exception as e:
        print(f"Error cargando productos: {e}")
        return []

def search_products(query, products_db, max_results=3):
    """
    Buscar productos por nombre, marca, sabor o características
    """
    if not products_db:
        return []
    
    query_lower = query.lower()
    matches = []
    
    query_words = query_lower.split()
    
    for product in products_db:
        score = 0
        match_details = []
        
        searchable_text = f"{product['marca']} {product['producto']} {product['sabor']} {product['capacidad']}".lower()
        
        if any(word in product['marca'].lower() for word in query_words):
            score += 10
            match_details.append("marca")
        
        if any(word in product['sabor'].lower() for word in query_words):
            score += 8
            match_details.append("sabor")
        
        if any(word in str(product['capacidad']).lower() for word in query_words):
            score += 5
            match_details.append("capacidad")
        
        if any(word in product.get('tipo_bebida', '').lower() for word in query_words):
            score += 6
            match_details.append("tipo")
        
        if any(word in product.get('caracteristicas_especiales', '').lower() for word in query_words):
            score += 4
            match_details.append("características")
        
        for word in query_words:
            if len(word) > 3:
                similarity = difflib.SequenceMatcher(None, word, product['sabor'].lower()).ratio()
                if similarity > 0.6:
                    score += int(similarity * 5)
                    match_details.append("similitud")
        
        if score > 0:
            matches.append({
                'product': product,
                'score': score,
                'match_details': match_details
            })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return [match['product'] for match in matches[:max_results]]


def create_rag_chain(db, llm):
    """
    Configurar RetrievalQA con parámetros optimizados
    """
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    from langchain.prompts import PromptTemplate
    
    template = """Basándote en el contexto proporcionado, responde de manera concisa y directa.
Máximo 3-4 oraciones. Ve al grano sin perder información importante.

Contexto: {context}

Pregunta: {question}

Respuesta concisa:"""

    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=False
    )
    
    return qa_chain

def get_response_with_rag(user_input, rag_chain, model, memory, products_context=""):
    """
    Función que integra RAG automáticamente sin prefijos
    """
    try:
        docs = rag_chain.retriever.invoke(user_input)
        
        if docs and any(doc.page_content.strip() for doc in docs):
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            chat_history = memory.buffer_as_str if hasattr(memory, 'buffer_as_str') else ""
            
            prompt = f"""Eres el asistente virtual de AJE Group. Tienes acceso a información sobre la estrategia de la empresa y también sobre nuestros productos. Usa la información más relevante para responder de manera natural y conversacional.

Información de estrategia empresarial:
{context}

Información de productos (si es relevante):
{products_context}

Conversación previa:
{chat_history}

Pregunta: {user_input}

Responde de forma natural y amigable como un experto de AJE Group:"""
            
            response = model.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            sources = set()
            for doc in docs[:3]:
                if doc.metadata.get('source_file'):
                    page = doc.metadata.get('page_number', 'N/A')
                    sources.add(f"{doc.metadata['source_file']} (p.{page})")
            
            
            return response_content, True
        else:
            return None, False
            
    except Exception as e:
        print(f"Error en RAG: {e}")
        return None, False

def main():
    """Función principal del chatbot"""
    print("Iniciando chatbot con Google Gemini 2.5 Flash...")
    
    try:
        print("Conectando con Google Gemini...")
        model = init_chat_model(
            "gemini-2.5-flash", 
            model_provider="google_genai",
            temperature=0.7
        )
        
        conversation_history = []
        
        print("Inicializando sistema RAG...")
        db, embeddings = create_or_load_faiss_db()
        
        rag_chain = None
        if db:
            rag_chain = create_rag_chain(db, model)
            print("Sistema RAG configurado exitosamente")
        else:
            print("Funcionando sin RAG - agrega PDFs a la carpeta 'data/'")
        
        print("Cargando base de datos de productos...")
        products_db = load_products_database()
        
        test_response = model.invoke("Hola, responde brevemente que estás funcionando")
        print("Conexión exitosa!")
        print(f"Prueba: {test_response.content}")
    
        print(f"\n{'='*60}")
        rag_status = "CON RAG" if rag_chain else "SIN RAG"
        products_status = f"+ {len(products_db)} PRODUCTOS" if products_db else ""
        print(f"\n¡Hola! Soy el asistente virtual de AJE Group")
        print("="*50)
        print("¡Pregúntame lo que necesites!")
        print("-"*50)
        
        conversation_count = 0
        
        while True:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n¡Hasta luego! / Goodbye!")
                break
            
            if user_input.lower() == 'memoria':
                if conversation_history:
                    print(f"\nHistorial de conversación ({len(conversation_history)} intercambios):")
                    for i, (user_msg, bot_msg) in enumerate(conversation_history, 1):
                        user_preview = user_msg[:80] + "..." if len(user_msg) > 80 else user_msg
                        bot_preview = bot_msg[:80] + "..." if len(bot_msg) > 80 else bot_msg
                        print(f"{i}. Tú: {user_preview}")
                        print(f"   Gemini: {bot_preview}")
                else:
                    print("\nNo hay historial de conversación aún.")
                continue
            
            if not user_input:
                print("Por favor escribe algo...")
                continue
            
            print("\nGemini: ", end="", flush=True)
            try:
                response_text = ""
                
                products_context = ""
                if products_db:
                    found_products = search_products(user_input, products_db, max_results=5)
                    if found_products:
                        products_context = "\n\nInformación de productos AJE disponibles:\n"
                        for p in found_products:
                            products_context += f"- {p['marca']} {p['sabor']} ({p['capacidad']}) - {p.get('tipo_bebida', 'bebida')}"
                            if p.get('caracteristicas_especiales') and p['caracteristicas_especiales'] != "No visible":
                                products_context += f" - {p['caracteristicas_especiales']}"
                            if p.get('descripcion_visual'):
                                products_context += f" - {p['descripcion_visual'][:100]}..."
                            products_context += "\n"
                    
                    elif any(word in user_input.lower() for word in ['productos', 'catálogo', 'portafolio', 'marcas']):
                        products_context = "\n\nPortafolio completo de productos AJE:\n"
                        by_brand = {}
                        for p in products_db:
                            brand = p['marca']
                            if brand not in by_brand:
                                by_brand[brand] = []
                            by_brand[brand].append(p)
                        
                        for brand, products in by_brand.items():
                            products_context += f"\n{brand}:\n"
                            unique_products = {}
                            for p in products:
                                key = f"{p['sabor']}_{p['capacidad']}"
                                if key not in unique_products:
                                    unique_products[key] = p
                            
                            for p in unique_products.values():
                                products_context += f"- {p['sabor']} ({p['capacidad']}) - {p.get('tipo_bebida', 'bebida')}\n"
                
                if rag_chain:
                    class SimpleMemory:
                        def __init__(self, history):
                            self.history = history
                        
                        @property
                        def buffer_as_str(self):
                            if not self.history:
                                return ""
                            recent = self.history[-3:]
                            return "\n".join([
                                f"Usuario: {user}\nAsistente: {bot}" 
                                for user, bot in recent
                            ])
                    
                    memory_obj = SimpleMemory(conversation_history)
                    rag_response, used_rag = get_response_with_rag(user_input, rag_chain, model, memory_obj, products_context)
                    
                    if used_rag and rag_response:
                        response_text = rag_response
                    else:
                        recent_history = conversation_history[-3:]
                        history_context = ""
                        if recent_history:
                            history_context = "\n".join([
                                f"Usuario: {user}\nAsistente: {bot}" 
                                for user, bot in recent_history
                            ])
                        
                        prompt = f"""Eres el asistente virtual de AJE Group. Responde de manera natural y conversacional como si fueras un experto en la empresa.

Conversación previa:
{history_context}

{products_context}

Usuario: {user_input}

Responde de forma natural y amigable:"""
                        
                        response = model.invoke(prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                else:
                    recent_history = conversation_history[-3:]
                    history_context = ""
                    if recent_history:
                        history_context = "\n".join([
                            f"Usuario: {user}\nAsistente: {bot}" 
                            for user, bot in recent_history
                        ])
                    
                    prompt = f"""Eres el asistente virtual de AJE Group. Responde de manera natural y conversacional como si fueras un experto en la empresa.

Conversación previa:
{history_context}

{products_context}

Usuario: {user_input}

Responde de forma natural y amigable:"""
                    
                    response = model.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                
                print(response_text)
                
                conversation_history.append((user_input, response_text))
                conversation_count += 1
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error: {error_msg}")
                
                if "quota" in error_msg.lower():
                    print("Cuota de API agotada")
                elif "rate" in error_msg.lower():
                    print("Límite de velocidad alcanzado")
                elif "key" in error_msg.lower() or "auth" in error_msg.lower():
                    print("Problema con la API key")
                else:
                    print("Intenta reformular tu pregunta")
        
    except KeyboardInterrupt:
        print("\n\n¡Programa interrumpido por el usuario!")
    except Exception as e:
        print(f"Error al inicializar: {str(e)}")
        
if __name__ == "__main__":
    main()

