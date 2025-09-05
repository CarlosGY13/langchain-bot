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

def get_response_with_rag(user_input, rag_chain, model, memory):
    """
    Función que integra RAG automáticamente sin prefijos
    """
    try:
        docs = rag_chain.retriever.invoke(user_input)
        
        if docs and any(doc.page_content.strip() for doc in docs):
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            chat_history = memory.buffer_as_str if hasattr(memory, 'buffer_as_str') else ""
            
            prompt = f"""Basándote en el contexto del documento y la conversación previa, responde de manera concisa (máximo 4 oraciones).

Contexto del documento:
{context}

Conversación previa:
{chat_history}

Pregunta: {user_input}

Respuesta concisa:"""
            
            response = model.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            sources = set()
            for doc in docs[:3]:
                if doc.metadata.get('source_file'):
                    page = doc.metadata.get('page_number', 'N/A')
                    sources.add(f"{doc.metadata['source_file']} (p.{page})")
            
            if sources:
                response_content += f"\n\n[Fuente: {', '.join(list(sources)[:2])}]"
            
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
        
        test_response = model.invoke("Hola, responde brevemente que estás funcionando")
        print("Conexión exitosa!")
        print(f"Prueba: {test_response.content}")
    
        print(f"\n{'='*60}")
        rag_status = "CON RAG" if rag_chain else "SIN RAG"
        print(f"CHATBOT GEMINI 2.5 FLASH + MEMORIA + {rag_status}")
        print("="*60)
        print("El chatbot integra automáticamente información de documentos cuando es relevante")
        print("Escribe 'salir' para terminar")
        print("Escribe 'memoria' para ver el historial") 
        print("-"*60)
        
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
                    rag_response, used_rag = get_response_with_rag(user_input, rag_chain, model, memory_obj)
                    
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
                        
                        prompt = f"""Responde de manera concisa y directa (máximo 4 oraciones).

Conversación previa:
{history_context}

Usuario: {user_input}

Respuesta concisa:"""
                        
                        response = model.invoke(prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                else:
                    # Sin RAG - solo conversación normal
                    recent_history = conversation_history[-3:]
                    history_context = ""
                    if recent_history:
                        history_context = "\n".join([
                            f"Usuario: {user}\nAsistente: {bot}" 
                            for user, bot in recent_history
                        ])
                    
                    prompt = f"""Responde de manera concisa y directa (máximo 4 oraciones).

Conversación previa:
{history_context}

Usuario: {user_input}

Respuesta concisa:"""
                    
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

