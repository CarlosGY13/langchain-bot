import subprocess
import sys
import os

def check_requirements():
    try:
        import streamlit
        import langchain
        from dotenv import load_dotenv
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False

def check_env_file():
    if not os.path.exists('.env'):
        print("⚠️  Archivo .env no encontrado")
        print("💡 Crea un archivo .env con:")
        print("   GOOGLE_API_KEY=tu_api_key_aqui")
        print("   OPENAI_API_KEY=tu_api_key_aqui (opcional)")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'GOOGLE_API_KEY' not in content:
            print("⚠️  GOOGLE_API_KEY no encontrada en .env")
            print("💡 Agrega: GOOGLE_API_KEY=tu_api_key_aqui")
            return False
    
    return True

def check_data_files():
    missing_files = []
    
    if not os.path.exists('productos_aje.json'):
        missing_files.append('productos_aje.json')
    
    if not os.path.exists('fotos'):
        missing_files.append('fotos/')
    
    if not os.path.exists('data'):
        missing_files.append('data/')
    
    if missing_files:
        print("⚠️  Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        
        if 'productos_aje.json' in missing_files:
            print("💡 Ejecuta: python process_products.py")
        
        return False
    
    return True

def main():
    print("🚀 LANZADOR DEL CHATBOT AJE - STREAMLIT")
    print("=" * 50)
    
    print("🔍 Verificando dependencias...")
    if not check_requirements():
        sys.exit(1)
    
    print("🔍 Verificando configuración...")
    if not check_env_file():
        sys.exit(1)
    
    print("🔍 Verificando archivos de datos...")
    if not check_data_files():
        print("\n💡 Algunos archivos faltan, pero la aplicación puede funcionar parcialmente.")
        response = input("¿Continuar de todos modos? (s/n): ").lower()
        if response not in ['s', 'si', 'sí', 'y', 'yes']:
            sys.exit(1)
    
    print("\n✅ Todas las verificaciones pasaron!")
    print("🌐 Lanzando aplicación Streamlit...")
    print("-" * 50)
    print("📱 La aplicación se abrirá en tu navegador")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Para detener: Ctrl+C")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 ¡Aplicación detenida por el usuario!")
    except Exception as e:
        print(f"\n❌ Error lanzando aplicación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()