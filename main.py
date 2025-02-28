import os
from scripts.augmentation import augment_images
from typing import Dict, List

from scripts.build_vgg16 import build_vgg16
from scripts.evaluate_model import evaluate_model
from keras.src.models.functional import Functional
from scripts.load_dataset import load_dataset
from scripts.test_model import predict_image
from scripts.train_vgg16 import train_vgg16
from scripts.use_cuda import set_cuda


augmented_train = "./augmented_data"

# Шлях до тестового зображення
test_image_path = "./dataset/test/buttons/btn_1.png"

if __name__ == '__main__':
    # Налаштувати CUDA
    set_cuda()
    
    # Завантажити та підготувати дані
    dataset: Dict = load_dataset()
    
    if not os.path.isdir(augmented_train):
        # Виконати аугментацію
        augment_images()
    
    # Побудувати модель VGG16
    model: Functional = build_vgg16(dataset['train_generator'])
    
    # Запуск навчання VGG16 (з CUDA)
    train_vgg16(
        model,
        dataset['train_generator'],
        dataset['val_generator']
    )
    
    # Оцінка моделі
    class_labels: List[str] = evaluate_model(
        model,
        dataset['train_generator']
    )
    
    # Тестування на окремих зображеннях
    predict_image(
        model=model,
        image_path=test_image_path,
        class_labels=class_labels
    )