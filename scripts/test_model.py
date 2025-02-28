import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Увімкнути інтерактивний бек-енд WSL2
import matplotlib
matplotlib.use("TkAgg")  # Можна змінити на "Qt5Agg"

# Увімкнути інтерактивний режим
plt.ion()

def predict_image(image_path, model, class_labels) -> None:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label}")
    
    plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    
    args = parser.parse_args()

    # Завантажуємо модель
    model = tf.keras.models.load_model(args.model)

    # Отримуємо мітки класів із моделі
    if hasattr(model, "class_indices"):
        class_labels = list(model.class_indices.keys())
    else:
        num_classes = model.output_shape[-1]
        class_labels = [f"Class_{i}" for i in range(num_classes)]

    # Виконуємо передбачення
    predict_image(args.image_path, model, class_labels)
