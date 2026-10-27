
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

print(f"Cargando modelo desde {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("¡Modelo cargado correctamente!\n")

# Mapeo de clases (orden alfabético por carpetas)
CLASS_NAMES = ["Gato (Cat)", "Perro (Dog)"]

def predict_animal(image_path: str):
    """
    Carga una imagen, la redimensiona a 200x200 y predice si es perro o gato.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: No se encontró la imagen en {img_path}")
        return None
    
    # Cargar y redimensionar a 200x200 (sin rescale, tal como espera EfficientNet)
    img = load_img(img_path, target_size=(200, 200))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Inferencia
    preds = model.predict(img_batch, verbose=0)[0]
    predicted_idx = np.argmax(preds)
    confidence = preds[predicted_idx] * 100
    
    predicted_class = CLASS_NAMES[predicted_idx]
    
    print("=" * 50)
    print(f"🖼️  Imagen: {img_path.name}")
    print(f"🐾  Predicción: {predicted_class}")
    print(f"📈  Confianza:  {confidence:.2f}%")
    print(f"📊  Probabilidades -> Gato: {preds[0]*100:.2f}% | Perro: {preds[1]*100:.2f}%")
    print("=" * 50)
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Prueba automática con una imagen de test si existen
    sample_dog = BASE_DIR / "data" / "processed" / "test" / "dogs"
    sample_cat = BASE_DIR / "data" / "processed" / "test" / "cats"
    
    if sample_dog.exists() and len(os.listdir(sample_dog)) > 0:
        first_dog = sample_dog / os.listdir(sample_dog)[0]
        predict_animal(first_dog)
        
    if sample_cat.exists() and len(os.listdir(sample_cat)) > 0:
        first_cat = sample_cat / os.listdir(sample_cat)[0]
        predict_animal(first_cat)
