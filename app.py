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
import base64
from pathlib import Path
import re

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
    
    return chunks

def create_embeddings():
    """
    Configurar embeddings: OpenAI primario, HuggingFace fallback
    """
    try:
        if os.getenv("OPENAI_API_KEY"):
            embeddings = OpenAIEmbeddings()
            embeddings.embed_query("test")
            return embeddings
        else:
            raise Exception("No OpenAI key")
    except Exception:
        pass
    
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
            db = FAISS.load_local(
                faiss_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return db, embeddings
        except Exception:
            pass
    
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        return None, embeddings
    
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = preprocess_pdf(pdf_file)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return None, embeddings
    
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local(faiss_path)
    
    return db, embeddings

def load_products_database(products_file="productos_aje.json"):
    """
    Cargar base de datos de productos AJE
    """
    try:
        if os.path.exists(products_file):
            with open(products_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['productos']
        else:
            return []
    except Exception:
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

def extract_volume_from_text(text):
    """Extraer volumen en ml de un texto"""
    try:
        patterns = [
            r'(\d+[,.]?\d*)\s*L',
            r'(\d+)\s*ml',
            r'(\d+[,.]?\d*)\s*litros?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                volume_str = matches[0].replace(',', '.')
                volume = float(volume_str)
                
                if 'l' in pattern.lower() and 'ml' not in pattern.lower():
                    volume = volume * 1000
                
                return int(volume)
        
        return None
    except:
        return None

def extract_volume_from_capacity(capacity_str):
    """Extraer volumen numérico de la capacidad (ej: '1000ml' -> 1000)"""
    try:
        if not capacity_str:
            return None
        
        numbers = re.findall(r'\d+', capacity_str)
        if numbers:
            return int(numbers[0])
        
        return None
    except:
        return None

def encode_image_for_vision(image_path):
    """Codificar imagen para análisis con Gemini Vision"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception:
        return None

def identify_product_from_image(image_path, products_db, model):
    """
    Identificar producto AJE desde imagen usando Gemini Vision
    """
    try:
        base64_image = encode_image_for_vision(image_path)
        if not base64_image:
            return None, "No se pudo procesar la imagen"
        
        products_list = ""
        for p in products_db:
            products_list += f"- {p['marca']} {p['sabor']} ({p['capacidad']}) - {p.get('tipo_bebida', 'bebida')}\n"
        
        prompt = f"""Analiza esta imagen de producto de bebida y determina si corresponde a alguno de los productos AJE de la siguiente lista:

PRODUCTOS AJE DISPONIBLES:
{products_list}

Instrucciones:
1. Examina cuidadosamente la imagen
2. Identifica la marca, sabor y capacidad visible en la etiqueta
3. IMPORTANTE: Lee el volumen exacto que aparece en el envase (ej: 1,035L, 500ml, 3,0L)
4. Compara con la lista de productos AJE disponibles
5. Si encuentras una coincidencia de marca y sabor, responde: "PRODUCTO_IDENTIFICADO: [Marca] [Sabor] [Volumen exacto leído]"
6. Si no encuentras coincidencia o no es un producto AJE, responde: "NO_IDENTIFICADO: [descripción de lo que ves]"

Ejemplos de respuesta correcta:
- "PRODUCTO_IDENTIFICADO: Big Cola Cola 1,035 L"
- "PRODUCTO_IDENTIFICADO: Sporade Blueberry 500ml"
- "NO_IDENTIFICADO: Coca-Cola 350ml (no es producto AJE)"

Respuesta:"""

        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        )
        
        response = model.invoke([message])
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Procesar respuesta - manejar tanto formato "PRODUCTO_IDENTIFICADO:" como detección directa
        if "PRODUCTO_IDENTIFICADO:" in response_text:
            product_info = response_text.split("PRODUCTO_IDENTIFICADO:")[1].strip()
        else:
            # Si no hay formato específico, usar toda la respuesta para análisis
            product_info = response_text.strip()
            
        best_match = None
        best_score = 0
        detected_volume = extract_volume_from_text(product_info)
        
        for p in products_db:
            score = 0
            
            # Coincidencia exacta de nombre
            product_name = f"{p['marca']} {p['sabor']} ({p['capacidad']})"
            if product_name.lower() in product_info.lower():
                return p, f"¡Producto identificado! Es {product_name}"
            
            # Coincidencia con tolerancia de volumen (ignorar espacios)
            marca_match = p['marca'].lower().replace(' ', '') in product_info.lower().replace(' ', '')
            sabor_match = p['sabor'].lower().replace(' ', '') in product_info.lower().replace(' ', '')
            
            if marca_match and sabor_match:
                score += 10
                
                # Extraer volumen detectado y comparar con tolerancia
                product_volume = extract_volume_from_capacity(p['capacidad'])
                
                if detected_volume and product_volume:
                    volume_diff = abs(detected_volume - product_volume)
                    
                    if volume_diff <= 100:  # Tolerancia de 100ml
                        score += 5
                        if volume_diff <= 50:  # Bonus por mayor precisión
                            score += 3
            
            if score > best_score:
                best_score = score
                best_match = p
        
        if best_match and best_score >= 10:
            # Mostrar información sobre la tolerancia si aplica
            product_volume = extract_volume_from_capacity(best_match['capacidad'])
            
            message = f"¡Producto identificado! Es {best_match['marca']} {best_match['sabor']} ({best_match['capacidad']})"
            
            if detected_volume and product_volume and detected_volume != product_volume:
                volume_diff = abs(detected_volume - product_volume)
                message += f" (detectado: {detected_volume}ml, diferencia: ±{volume_diff}ml)"
            
            return best_match, message
        
        return None, f"Producto detectado: {product_info}, pero no encontrado en base de datos (score: {best_score})"
            
    except Exception as e:
        return None, f"Error analizando imagen: {str(e)}"

def handle_image_upload():
    """
    Manejar carga de imagen para identificación
    """
    print("\nIDENTIFICACIÓN VISUAL DE PRODUCTOS")
    print("=" * 50)
    print("Puedes subir una imagen de un producto AJE para identificarlo.")
    print("Formatos soportados: .png, .jpg, .jpeg")
    print("Escribe 'cancelar' para volver al chat normal.")
    print("-" * 50)
    
    while True:
        image_path = input("\nRuta de la imagen (o 'cancelar'): ").strip()
        
        if image_path.lower() == 'cancelar':
            return None, "Identificación visual cancelada"
        
        if not image_path:
            print("Por favor ingresa una ruta válida")
            continue
        
        # Verificar si el archivo existe
        if not os.path.exists(image_path):
            print(f"Archivo no encontrado: {image_path}")
            continue
        
        # Verificar extensión
        valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        if not any(image_path.endswith(ext) for ext in valid_extensions):
            print("Formato no soportado. Use: .png, .jpg, .jpeg")
            continue
        
        return image_path, "Imagen cargada correctamente"


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
    try:
        model = init_chat_model(
            "gemini-2.5-flash", 
            model_provider="google_genai",
            temperature=0.7
        )
        
        conversation_history = []
        
        # Inicializar sistema RAG
        db, embeddings = create_or_load_faiss_db()
        rag_chain = create_rag_chain(db, model) if db else None
        
        # Cargar base de productos
        products_db = load_products_database()
    
        print(f"\n¡Hola! Soy el asistente virtual de AJE Group")
        print("="*50)
        print("Puedo ayudarte con:")
        print("• Información sobre productos y estrategia")
        print("• Identificación visual de productos (sube una imagen)")
        print("¡Pregúntame lo que necesites!")
        print("\nComandos especiales:")
        print("• 'imagen' - Identificar producto desde imagen")
        print("• 'salir' - Terminar conversación")
        print("-"*50)
        
        conversation_count = 0
        
        while True:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n¡Hasta luego! / Goodbye!")
                break
            
            if user_input.lower() in ['imagen', 'img', 'photo', 'foto']:
                image_path, message = handle_image_upload()
                if image_path:
                    print(f"\n🔍 Analizando imagen: {Path(image_path).name}")
                    print("Esto puede tomar unos segundos...")
                    
                    identified_product, result_message = identify_product_from_image(image_path, products_db, model)
                    
                    if identified_product:
                        response_text = f"{result_message}\n\n"
                        response_text += f"**Información del producto:**\n"
                        response_text += f"• Marca: {identified_product['marca']}\n"
                        response_text += f"• Sabor: {identified_product['sabor']}\n"
                        response_text += f"• Capacidad: {identified_product['capacidad']}\n"
                        response_text += f"• Tipo: {identified_product.get('tipo_bebida', 'No especificado')}\n"
                        
                        if identified_product.get('caracteristicas_especiales') and identified_product['caracteristicas_especiales'] != "No visible":
                            response_text += f"• Características: {identified_product['caracteristicas_especiales']}\n"
                        
                        if identified_product.get('descripcion_visual'):
                            response_text += f"• Descripción: {identified_product['descripcion_visual'][:150]}...\n"
                        
                        print(f"\nGemini: {response_text}")
                        conversation_history.append(("Identificación de imagen", response_text))
                    else:
                        print(f"\nGemini: {result_message}")
                        conversation_history.append(("Identificación de imagen", result_message))
                else:
                    print(f"\n{message}")
                
                conversation_count += 1
                continue
            
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
                if "quota" in error_msg.lower():
                    print("Error: Cuota de API agotada")
                elif "rate" in error_msg.lower():
                    print("Error: Límite de velocidad alcanzado")
                elif "key" in error_msg.lower() or "auth" in error_msg.lower():
                    print("Error: Problema con la API key")
                else:
                    print(f"Error: {error_msg}")
        
    except KeyboardInterrupt:
        print("\n\n¡Programa interrumpido por el usuario!")
    except Exception as e:
        print(f"Error al inicializar: {str(e)}")
        
if __name__ == "__main__":
    main()

