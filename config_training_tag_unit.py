# config_training_tag_unit.py
"""
Конфигурация для обучения моделей tag, unit1, unit2.
Все архитектурные параметры вынесены сюда.
"""

# ======================== ParNetTag =============================
TAG_MODEL_HIDDEN_DIM = 256
TAG_MODEL_NUM_LAYERS = 4
TAG_MODEL_DROPOUT = 0.1          # dropout между слоями MLP (0 = без dropout)

# ======================== UnitParnet1 ===========================
UNIT1_IN_CHANNELS = 4
UNIT1_OUT_CHANNELS = 4

# Основной извлекатель признаков (img_conv)
UNIT1_IMG_CONV_HIDDEN = 128        # размерность скрытого слоя
UNIT1_IMG_CONV_LAYERS = 4         # количество свёрток в блоке (каждая с GELU)
UNIT1_IMG_CONV_DROPOUT = 0.0

# Кодировщик текста (text_encoder)
UNIT1_TEXT_HIDDEN = 128            # скрытая размерность в свёртке 1x1
UNIT1_TEXT_DROPOUT = 0.0

# Пространственное внимание
UNIT1_ATTENTION_DROPOUT = 0.0

# Выходной блок (out_conv)
UNIT1_OUT_CONV_HIDDEN = 128        # скрытая размерность
UNIT1_OUT_CONV_LAYERS = 4         # число свёрток перед финальным слоем
UNIT1_OUT_CONV_DROPOUT = 0.0

# ======================== UnitParnet2 ===========================
UNIT2_IN_CHANNELS = 4
UNIT2_OUT_CHANNELS = 4

# Кодировщик шума (encoder)
UNIT2_ENCODER_HIDDEN = 64
UNIT2_ENCODER_LAYERS = 4          # число свёрток
UNIT2_ENCODER_DROPOUT = 0.0

# Блок предсказания остаточного шума (residual)
UNIT2_RESIDUAL_HIDDEN = 64
UNIT2_RESIDUAL_LAYERS = 4         # число свёрток перед финальным предсказанием
UNIT2_RESIDUAL_DROPOUT = 0.0

# ─── Данные ───────────────────────────────────────────────────────
DATASET_DIR_TAG = "./prepared_dataset_tag_image"  # папка с .pt и .txt
TAG_SEQ_MAX_LEN = 32          # максимальное количество тегов
TAG_BYTE_LEN = 128            # длина байтовой последовательности одного тега
PAD_BYTE_VALUE = 0            # байт паддинга коротких тегов

# ─── Шум ──────────────────────────────────────────────────────────
NOISE_SEED = 42               # базовый сид (фактически заменён хэшем имени)
NOISE_STD = 0.5               # стандартное отклонение гауссова шума

# ─── Обучение ──────────────────────────────────────────────────────
BATCH_SIZE = 3
LEARNING_RATE = 0.00001
NUM_EPOCHS = 10000
RANDOM_SEED = 1234

# ─── Чекпоинты ────────────────────────────────────────────────────
MODELS_DIR_TAG_UNIT = "./models_tag_unit"
MAX_CHECKPOINTS = 5



MAX_TRAIN_IMAGES = 427          # ограничение числа тренировочных примеров (None – все)
# ─── Валидация ────────────────────────────────────────────
VALIDATION_SPLIT = 10          # сколько примеров оставить на валидацию
VAL_EVERY_EPOCHS = 2
# ─── тесты ────────────────────────────────────────────
TEST_EVERY_EPOCHS = 5
NUM_TEST_EXAMPLES = 10         # число примеров для визуализации
TEST_SEED = 5678

SAVE_EVERY_EPOCHS = 1
CLEAR_CACHE_EACH_BATCH = True

# Папки для сохранения результатов
TESTS_DIR = "./tests"         # тесты (примеры из обучающей выборки)
VAL_TESTS_DIR = "./val_tests" # валидация (примеры из валидационной выборки)

# Для визуализации: путь к инференс-декодеру (TorchScript)
DECODER_INFERENCE_PATH = "./models/decoder_inference.pt"
# Путь к декомпрессору для преобразования сжатого парнета в полный 3-канальный
DECOMPRESSOR_INFERENCE_PATH = "./models_compressor/decompressor_inference.pt"

# Потеря
PARNET_DIFF_LOSS_WEIGHT = 100.0  # вес для гипотетической потери
