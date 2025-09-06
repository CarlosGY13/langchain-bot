import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from app import (
    load_products_database, 
    create_or_load_faiss_db, 
    create_rag_chain,
    get_response_with_rag,
    identify_product_from_image,
    search_products
)
from langchain.chat_models import init_chat_model

load_dotenv()

st.set_page_config(
    page_title="Chatbot AJE Group",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
    }
    
    .assistant-bubble {
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .product-info {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,107,53,0.3);
    }
    
    .product-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .product-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .product-detail {
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,107,53,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,107,53,0.4);
    }
    
    .stButton > button:disabled {
        background: #cccccc;
        transform: none;
        box-shadow: none;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        height: 50px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF6B35 !important;
        box-shadow: 0 0 0 3px rgba(255,107,53,0.1) !important;
    }
    
    .stFileUploader > div {
        border-radius: 15px !important;
        border: 2px dashed #FF6B35 !important;
        background: rgba(255,107,53,0.05) !important;
        min-height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stFileUploader > div > div > div > div {
        font-size: 0.9rem !important;
        color: #666 !important;
    }
    
    .stButton > button {
        height: 50px !important;
        width: 50px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.2rem !important;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-style: italic;
        margin: 1rem 0;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #FF6B35;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .stTextInput input {
        border-radius: 25px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    try:
        model = init_chat_model(
            "gemini-2.0-flash-exp",
            model_provider="google",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        products_db = load_products_database()
        
        try:
            db = create_or_load_faiss_db()
            rag_chain = create_rag_chain(model, db)
        except Exception as e:
            st.warning(f"⚠️ RAG no disponible: {e}")
            rag_chain = None
        
        return model, rag_chain, products_db
        
    except Exception as e:
        st.error(f"❌ Error inicializando chatbot: {e}")
        return None, None, None

def process_text_query(user_input, model, rag_chain, products_db, conversation_history):
    try:
        relevant_products = search_products(user_input, products_db, max_results=3)
        
        products_context = ""
        if relevant_products:
            products_context = "Productos AJE relevantes:\n"
            for product in relevant_products:
                products_context += f"- {product['marca']} {product['sabor']} ({product['capacidad']})\n"
        
        if rag_chain:
            response = get_response_with_rag(
                user_input, rag_chain, model, conversation_history, products_context
            )
        else:
            response = model.invoke(f"Contexto: {products_context}\n\nPregunta: {user_input}")
        
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        return f"Error procesando consulta: {str(e)}"

def process_image_query(uploaded_file, model, products_db):
    try:
        if uploaded_file.size > 10 * 1024 * 1024:
            return None, "La imagen es demasiado grande. Máximo 10MB."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            identified_product, result_message = identify_product_from_image(temp_path, products_db, model)
        except Exception as vision_error:
            os.unlink(temp_path)
            return None, f"Error en análisis de imagen: {str(vision_error)}"
        
        os.unlink(temp_path)
        
        return identified_product, result_message
        
    except Exception as e:
        return None, f"Error procesando imagen: {str(e)}"

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🥤 Chatbot AJE Group</h1>
        <p>Asistente virtual inteligente con identificación visual de productos</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, rag_chain, products_db = initialize_chatbot()
    
    if not model:
        st.error("❌ No se pudo inicializar el chatbot. Verifica tu configuración.")
        return
    
    with st.sidebar:
        st.markdown("### 📋 ¿Qué puedo hacer?")
        st.write("• Responder sobre estrategia de internacionalización")
        st.write("• Consultar información de productos AJE")
        st.write("• Identificar productos desde imágenes")
        st.write("• Mantener conversación natural")
        
        st.markdown("---")
        st.write("**Nuestras marcas:**")
        st.write("🥤 Big Cola")
        st.write("🏃 Sporade")
        st.write("🍃 Bio Amayu")
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False
    
    if st.session_state.conversation_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, (user_msg, bot_msg) in enumerate(st.session_state.conversation_history):
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">{user_msg}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-bubble">{bot_msg.replace('<', '&lt;').replace('>', '&gt;')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="assistant-message">
            <div class="assistant-bubble">
                ¡Hola! Soy el asistente virtual de AJE Group. 👋<br><br>
                Puedo ayudarte con información sobre nuestros productos, estrategia de internacionalización, 
                o identificar productos AJE desde imágenes.<br><br>
                ¿En qué puedo ayudarte hoy?
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([6, 2, 1])
    
    with col1:
        user_input = st.text_input(
            "Mensaje",
            placeholder="Escribe tu mensaje y presiona el botón 📤...",
            key=f"text_input_{st.session_state.input_key}",
            disabled=st.session_state.processing,
            label_visibility="collapsed"
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "Imagen",
            type=['png', 'jpg', 'jpeg'],
            help="Subir imagen",
            disabled=st.session_state.processing,
            label_visibility="collapsed",
            key=f"image_uploader_{st.session_state.input_key}"
        )
    
    with col3:
        if uploaded_file is not None:
            send_button = st.button(
                "📸", 
                key="send_button",
                disabled=st.session_state.processing,
                help="Identificar producto"
            )
        elif user_input and user_input.strip():
            send_button = st.button(
                "📤", 
                key="send_button",
                disabled=st.session_state.processing,
                help="Enviar mensaje"
            )
        else:
            send_button = st.button(
                "📤", 
                key="send_button",
                disabled=True,
                help="Escribe un mensaje"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and not st.session_state.processing:
        st.markdown("**📸 Imagen seleccionada:**")
        st.image(uploaded_file, caption=f"📁 {uploaded_file.name}", width=200)
    
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(function(mutations) {
            const textInput = document.querySelector('input[data-testid*="text_input"]');
            if (textInput && !textInput.hasAttribute('data-enter-listener')) {
                textInput.setAttribute('data-enter-listener', 'true');
                textInput.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && !event.shiftKey && this.value.trim()) {
                        event.preventDefault();
                        const sendButton = document.querySelector('button[data-testid*="send"]');
                        if (sendButton && !sendButton.disabled) {
                            sendButton.click();
                        }
                    }
                });
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
    </script>
    """, unsafe_allow_html=True)
    
    if send_button and not st.session_state.processing:
        st.session_state.processing = True
        
        try:
            if uploaded_file is not None:
                with st.spinner("🔍 Analizando imagen..."):
                    identified_product, result_message = process_image_query(
                        uploaded_file, model, products_db
                    )
                
                if identified_product:
                    response = f"🎉 ¡Producto identificado!\n\n"
                    response += f"🥤 Producto: {identified_product['marca']} {identified_product['sabor']}\n"
                    response += f"📏 Capacidad: {identified_product['capacidad']}\n"
                    response += f"🏷️ Tipo: {identified_product.get('tipo_bebida', 'No especificado')}\n\n"
                    
                    if identified_product.get('caracteristicas_especiales') and identified_product['caracteristicas_especiales'] != "No visible":
                        response += f"⭐ Características: {identified_product['caracteristicas_especiales']}\n\n"
                    
                    if identified_product.get('descripcion_visual'):
                        response += f"👁️ Descripción visual:\n{identified_product['descripcion_visual']}"
                else:
                    response = f"❌ No pude identificar este producto. {result_message}"
                
                st.session_state.conversation_history.append(("📸 Imagen de producto", response))
            
            elif user_input and user_input.strip():
                with st.spinner("🤔 Pensando..."):
                    response = process_text_query(
                        user_input, 
                        model, 
                        rag_chain, 
                        products_db, 
                        st.session_state.conversation_history
                    )
                
                st.session_state.conversation_history.append((user_input, response))
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            if uploaded_file is not None:
                st.session_state.conversation_history.append(("📸 Imagen de producto", error_msg))
            else:
                st.session_state.conversation_history.append((user_input or "Consulta", error_msg))
        
        finally:
            st.session_state.processing = False
            st.session_state.input_key += 1
            
            st.markdown("""
            <script>
            setTimeout(function() {
                const inputs = document.querySelectorAll('input[type="text"]');
                inputs.forEach(input => {
                    if (input.placeholder && input.placeholder.includes('📤')) {
                        input.value = '';
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                });
            }, 100);
            </script>
            """, unsafe_allow_html=True)
            
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🚀 Chatbot AJE Group - Desarrollado con Google Gemini Vision + LangChain + Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()