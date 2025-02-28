import tensorflow as tf
from keras.src.models.functional import Functional

# Завантаження VGG16 без верхніх шарів (з попередньо навченими вагами ImageNet)
base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Заморожуємо ваги попередніх шарів
base_model.trainable = False

def build_vgg16(train_generator) -> Functional:
    # Додаємо власні повнозв’язні шари
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

    # Створюємо фінальну модель
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

    # Компільовуємо модель
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Виводимо архітектуру
    model.summary()
    
    # Повертаємо модель
    return model