import albumentations as A
import cv2
import os


# Папка зображень та збереження аугментованих даних
input_dir = "./dataset/train"
output_dir = "./augmented_data"

# Створюємо трансформації
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),  # Зміна освітлення та контрасту
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Розмиття
    A.GaussNoise(var_limit=(10, 50), p=0.3),  # Додавання шуму
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Деформації
    A.HorizontalFlip(p=0.5),  # Дзеркальне відображення
    A.Rotate(limit=30, p=0.5)  # Випадковий поворот
])

# Функція для обробки зображень
def augment_images(input_folder=input_dir, output_folder=output_dir, num_augmented=3) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for i in range(num_augmented):  # Кількість нових зразків на кожне зображення
                augmented = transform(image=image)['image']
                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                aug_img_path = os.path.join(output_class_path, aug_img_name)
                cv2.imwrite(aug_img_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
