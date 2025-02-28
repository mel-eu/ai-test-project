import tensorflow as tf
from typing import Dict


# Шляхи до папок
train_dir = "./dataset/train"
val_dir = "./dataset/validation"
test_dir = "./dataset/test"

# Параметри для VGG16
img_size = (224, 224)
batch_size = 32


def load_dataset() -> Dict[str, tf.keras.preprocessing.image.DirectoryIterator]:
    # Аугментація для тренувального набору
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,         # Нормалізація пікселів
        rotation_range=20,       # Обертання
        width_shift_range=0.2,   # Зсув по ширині
        height_shift_range=0.2,  # Зсув по висоті
        shear_range=0.2,         # Зсув
        zoom_range=0.2,          # Масштабування
        horizontal_flip=True     # Віддзеркалення
    )

    # Для валідації та тесту тільки нормалізація
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Завантаження зображень з папок
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'test_generator': test_generator
    }