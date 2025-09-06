import os
import json
import re
import glob
from pathlib import Path
import base64
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class ProductProcessor:
    def __init__(self):
        print("Inicializando Google Gemini Vision...")
        self.vision_model = init_chat_model(
            "gemini-2.5-flash", 
            model_provider="google_genai",
            temperature=0.3
        )
        print("Modelo de visión listo!")
        
    def parse_filename(self, filepath):
        folder_name = Path(filepath).parent.name
        filename = Path(filepath).stem
        
        marca = folder_name.replace('_', ' ').title()
        
        parts = filename.split('_')
        
        capacidad = None
        sabor_parts = []
        
        for i, part in enumerate(parts):
            if re.match(r'^\d+$', part):
                capacidad = f"{part}ml"
                
                if marca.lower() == "big cola":
                    producto = "Big Cola"
                    sabor_parts = parts[1:i] if i > 1 else ["Original"]
                elif marca.lower() == "bio amayu":
                    producto = "Bio Amayu"
                    sabor_parts = parts[2:i] if i > 2 else ["Original"]
                elif marca.lower() == "sporade":
                    producto = "Sporade"
                    sabor_parts = parts[1:i] if i > 1 else ["Original"]
                else:
                    producto = parts[0].replace('_', ' ').title()
                    sabor_parts = parts[1:i] if i > 1 else ["Original"]
                break
        
        if not capacidad:
            if marca.lower() == "big cola":
                producto = "Big Cola"
                sabor_parts = parts[2:] if len(parts) > 2 else ["Original"]
            elif marca.lower() == "bio amayu":
                producto = "Bio Amayu"
                sabor_parts = parts[2:] if len(parts) > 2 else ["Original"]
            elif marca.lower() == "sporade":
                producto = "Sporade"
                sabor_parts = parts[1:] if len(parts) > 1 else ["Original"]
            else:
                producto = parts[0].replace('_', ' ').title()
                sabor_parts = parts[1:] if len(parts) > 1 else ["Original"]
        
        if sabor_parts and sabor_parts != ["Original"]:
            sabor = ' '.join(sabor_parts).replace('_', ' ').title()
        else:
            sabor = "Original"
        
        return {
            "marca": marca,
            "producto": producto, 
            "sabor": sabor,
            "capacidad": capacidad,
            "archivo_original": filename
        }
    
    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error codificando imagen {image_path}: {e}")
            return None
    
    def analyze_image_with_vision(self, image_path):
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return {}
            
            prompt = f"""
Analiza esta imagen de producto de bebida AJE y extrae SOLO la información que puedes ver claramente.

Responde en formato JSON con esta estructura exacta:
{{
    "tipo_bebida": "tipo de bebida que observas (gaseosa, jugo, energética, deportiva, agua, etc.)",
    "tipo_envase": "tipo de envase (botella plástica, lata, tetrapack, etc.)",
    "colores_principales": ["color1", "color2", "color3"],
    "texto_visible": "cualquier texto o eslogan visible en el envase",
    "caracteristicas_especiales": "características nutricionales visibles (sin azúcar, light, natural, etc.)",
    "descripcion_visual": "descripción breve del producto basada en lo que ves"
}}

Si no puedes determinar algún campo con certeza, usa "No visible" como valor.
Responde SOLO el JSON, sin texto adicional.
"""

            # Crear mensaje con imagen
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
            
            response = self.vision_model.invoke([message])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            try:
                json_text = response_text.strip()
                if json_text.startswith('```json'):
                    json_text = json_text[7:-3]
                elif json_text.startswith('```'):
                    json_text = json_text[3:-3]
                
                vision_data = json.loads(json_text)
                print(f"Análisis exitoso: {Path(image_path).name}")
                return vision_data
                
            except json.JSONDecodeError as e:
                print(f"Error parseando JSON para {image_path}: {e}")
                print(f"Respuesta recibida: {response_text[:200]}...")
                return {}
                
        except Exception as e:
            print(f"Error analizando imagen {image_path}: {e}")
            return {}
    
    def group_product_images(self, folder_path):
        image_files = glob.glob(os.path.join(folder_path, "*.png"))
        image_files.extend(glob.glob(os.path.join(folder_path, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
        
        products = {}
        
        for img_path in image_files:
            filename = Path(img_path).stem
            
            base_name = re.sub(r'_\d+$', '', filename)
            
            if base_name not in products:
                products[base_name] = []
            
            products[base_name].append(img_path)
        
        return products
    
    def process_all_products(self, fotos_folder="fotos"):
        if not os.path.exists(fotos_folder):
            print(f"Carpeta {fotos_folder} no encontrada")
            return []
        
        all_products = []
        
        for brand_folder in os.listdir(fotos_folder):
            brand_path = os.path.join(fotos_folder, brand_folder)
            
            if not os.path.isdir(brand_path):
                continue
                
            print(f"\nProcesando marca: {brand_folder}")
            
            product_groups = self.group_product_images(brand_path)
            
            for base_name, image_paths in product_groups.items():
                print(f"  🔍 Procesando producto: {base_name}")
                
                main_image = image_paths[0]
                
                file_data = self.parse_filename(main_image)
                
                print(f"Analizando imagen con IA...")
                vision_data = self.analyze_image_with_vision(main_image)
                
                product = {
                    "id": f"{file_data['marca'].lower().replace(' ', '_')}_{file_data['sabor'].lower().replace(' ', '_')}_{file_data['capacidad'] or 'unknown'}",
                    "marca": file_data["marca"],
                    "producto": file_data["producto"],
                    "sabor": file_data["sabor"],
                    "capacidad": file_data["capacidad"],
                    
                    "tipo_bebida": vision_data.get("tipo_bebida", "No determinado"),
                    "tipo_envase": vision_data.get("tipo_envase", "No determinado"),
                    "colores_principales": vision_data.get("colores_principales", []),
                    "texto_visible": vision_data.get("texto_visible", ""),
                    "caracteristicas_especiales": vision_data.get("caracteristicas_especiales", ""),
                    "descripcion_visual": vision_data.get("descripcion_visual", ""),
                    
                    "imagenes": [Path(img).name for img in image_paths],
                    "imagen_principal": Path(main_image).name,
                    "total_imagenes": len(image_paths)
                }
                
                all_products.append(product)
                print(f"Producto procesado: {product['marca']} {product['sabor']} {product['capacidad']}")
        
        return all_products
    
    def save_products_database(self, products, output_file="productos_aje.json"):
        database = {
            "metadata": {
                "total_productos": len(products),
                "marcas": list(set(p["marca"] for p in products)),
                "tipos_bebida": list(set(p["tipo_bebida"] for p in products if p["tipo_bebida"] != "No determinado")),
                "generado_con": "Google Gemini Vision + Parsing de archivos",
                "version": "1.0"
            },
            "productos": products
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False)
        
        print(f"\nBase de datos guardada en: {output_file}")
        print(f"Total productos: {len(products)}")
        print(f"Marcas encontradas: {', '.join(database['metadata']['marcas'])}")
        
        return output_file

def main():
    processor = ProductProcessor()
    
    products = processor.process_all_products()
    
    if products:
        db_file = processor.save_products_database(products)
        
        print(f"\nArchivo generado: {db_file}")
        print(f"Listo para integrar en el chatbot")
    else:
        print("No se encontraron productos para procesar")

if __name__ == "__main__":
    main()
