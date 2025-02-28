import tensorflow as tf

def set_cuda() -> None:
    # Перевіряємо, чи TensorFlow використовує GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Доступні GPU: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU налаштовано для використання")
        except RuntimeError as e:
            print(e)
    else:
        print("❌ CUDA не знайдено, перевірте налаштування")