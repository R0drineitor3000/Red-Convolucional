# cnn.py
import os
import json  # <-- Importamos json para guardar las etiquetas de texto
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Importamos las funciones corregidas de image.py
from image import resize, get_image_file, get_folder_path, get_file

dataset = None

def define_dataset():
    global dataset
    dataset = {}
    while True:
        dataset_name = input("Enter the name of the dataset class (example: dogs, cats, etc.): ")
        dataset_path = get_folder_path()
        if not dataset_path:
            print("No folder selected. Try again.")
            continue
        dataset[dataset_name] = dataset_path
        if int(input("Would you like to add another dataset? (1: Sí, 0: No): ")) == 0:
            break

def train_network(image_width, image_height, epochs=10):
    if not dataset:
        print("First, you need to define the dataset.")
        return None, None

    data = []
    labels = []
    label_map = {name: i for i, name in enumerate(dataset.keys())}
    print(f"Tag mapping: {label_map}")

    # 1. Recolección de datos
    for dataset_name, dataset_path in dataset.items():
        print(f"Processing: {dataset_name}...")
        label = label_map[dataset_name]
        
        for image_file in os.listdir(dataset_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                continue
                
            image_path = os.path.join(dataset_path, image_file)
            
            try:
                image_bytes = get_image_file(image_path)
                img_array = resize(image_bytes, image_width, image_height)
                img_array = img_array / 255.0
                
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Could not process image {image_file}: {e}")

    if len(data) == 0:
        print("No valid images found in the specified directories.")
        return None, None

    X = np.array(data)
    y = np.array(labels)

    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    X = X[indexes]
    y = y[indexes]

    num_classes = len(label_map)
    if num_classes == 2:
        output_neurons = 1
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:
        output_neurons = num_classes
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'

    # 2. Arquitectura de la CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_neurons, activation=output_activation)
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    print("Starting training...")
    model.fit(X, y, epochs=epochs, batch_size=32)
    print("Training completed.")
    
    # CORRECCIÓN: Ahora retornamos el modelo Y el diccionario de clases
    return model, label_map

def save_model(model, label_map, filename="my_model.keras"):
    """Guarda el modelo .keras y un archivo .json con las etiquetas al lado."""
    print("Select the folder where you want to save the model...")
    save_dir = get_folder_path()
    if save_dir is None:
        print("Model saving cancelled.")
        return
    
    if not filename.endswith('.keras'):
        filename += '.keras'
        
    full_path = os.path.join(save_dir, filename)
    
    # 1. Guardar el modelo Keras de forma limpia
    model.save(full_path)
    
    # 2. Guardar el archivo JSON en la misma ruta
    json_path = full_path.replace('.keras', '_labels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
        
    print(f"Model saved at: {full_path}")
    print(f"Labels saved successfully at: {json_path}")


def load_model():
    """Carga el archivo .keras y su archivo de etiquetas .json asociado."""
    print("Select the .keras model file to load...")
    full_path = get_file()
    if full_path is None:
        print("Model loading cancelled.")
        return None, None
    
    model = models.load_model(full_path)
    print(f"Model loaded successfully from: {full_path}")
    
    # Buscar de forma automática el archivo JSON asociado
    json_path = full_path.replace('.keras', '_labels.json')
    label_map = None
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print("Labels loaded successfully.")
    else:
        print("Warning: Associated '_labels.json' file not found. Numeric indexes will be used.")
        
    return model, label_map

if __name__ == "__main__":
    model = None
    label_map = None

    choice = int(input("Would you like to train or load a model? (1: Train, 0: Load): "))
    if choice == 1:
        define_dataset()
        # Capturamos ambas salidas de la función
        model, label_map = train_network(image_width=128, image_height=128, epochs=10)
        if model and label_map:
            save_model(model, label_map)
    else:
        # Cargamos el modelo junto con sus etiquetas guardadas en disco
        model, label_map = load_model()

    if model is None:
        print("No model available for evaluation.")
        exit()

    # Si por alguna razón no hay label_map (ej. se borró el json), creamos uno numérico por defecto
    if label_map is None:
        inverse_label_map = {0: "Class 0", 1: "Class 1"}
    else:
        # Al venir del JSON: k es el texto (ej. "dogs") y v es el entero (ej. 0)
        inverse_label_map = {int(v): str(k) for k, v in label_map.items()}
    
    # Evaluate the model
    while True:
        print("Select the image file to test the model...")
        test_image_path = get_file()
        if test_image_path and model:
            try:
                image_bytes = get_image_file(test_image_path)
                img_array = resize(image_bytes, 128, 128) / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de batch
                
                predictions = model.predict(img_array)
                if predictions.shape[1] > 1:
                    predicted_class = np.argmax(predictions)
                else:
                    predicted_class = 1 if predictions[0][0] > 0.5 else 0
                
                class_obtained = inverse_label_map.get(predicted_class, f"Unknown Index ({predicted_class})")
                print(f"Predicted class: {class_obtained}")

            except Exception as e:
                print(f"Could not process the test image: {e}")
                
        choice = int(input("Would you like to test another image? (1: Yes, 0: No): "))
        if choice == 0:
            break