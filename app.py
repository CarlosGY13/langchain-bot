import getpass
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar API key de Google Gemini
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

def main():
    """Función principal del chatbot"""
    print("Iniciando chatbot con Google Gemini 2.5 Flash...")
    
    try:
        # Inicializar el modelo
        print("Conectando con Google Gemini...")
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Inicializar memoria de conversación
        memory = ConversationBufferMemory(return_messages=True)
        
        # Prueba de conexión
        test_response = model.invoke("Hola, responde brevemente que estás funcionando")
        print("Conexión exitosa!")
        print(f"Prueba: {test_response.content}")
        
        # Iniciar chat interactivo
        print(f"\n{'='*60}")
        print("CHATBOT CON GOOGLE GEMINI 2.5 FLASH (CON MEMORIA)")
        print("="*60)
        print("Puedes escribir en español, inglés u otros idiomas")
        print("Escribe 'salir' para terminar")
        print("Escribe 'info' para ver información del modelo")
        print("Escribe 'memoria' para ver el historial")
        print("-"*60)
        
        conversation_count = 0
        
        while True:
            # Obtener input del usuario
            user_input = input("\nTú: ").strip()
            
            # Verificar comandos especiales
            if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n¡Hasta luego! / Goodbye!")
                break
            
            if user_input.lower() == 'info':
                print(f"\nModelo: Google Gemini 2.5 Flash")
                print(f"Conversaciones: {conversation_count}")
                print("Idiomas: Multilingüe completo")
                print("Velocidad: Ultra rápido")
                print("Memoria: Activada (historial completo)")
                continue
            
            if user_input.lower() == 'memoria':
                chat_history = memory.chat_memory.messages
                if chat_history:
                    print(f"\nHistorial de conversación ({len(chat_history)} mensajes):")
                    for i, msg in enumerate(chat_history, 1):
                        role = "Tú" if isinstance(msg, HumanMessage) else "Gemini"
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"{i}. {role}: {content}")
                else:
                    print("\nNo hay historial de conversación aún.")
                continue
            
            if not user_input:
                print("Por favor escribe algo...")
                continue
            
            # Generar respuesta con memoria
            print("\nGemini: ", end="", flush=True)
            try:
                # Obtener historial de la memoria
                chat_history = memory.chat_memory.messages
                
                # Crear lista de mensajes con historial + nuevo mensaje
                messages = chat_history + [HumanMessage(content=user_input)]
                
                # Enviar al modelo con contexto completo
                response = model.invoke(messages)
                
                # Guardar el intercambio en la memoria
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(response.content)
                
                print(response.content)
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
        print("\nSoluciones posibles:")
        print("   1. Verificar tu API key de Google AI")
        print("   2. Asegurar conexión a internet")
        print("   3. Instalar: pip install langchain-google-genai")

if __name__ == "__main__":
    main()

