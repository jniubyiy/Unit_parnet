# config_training_tag_unit.py

"""
Конфигурация для обучения моделей tag, unit1, unit2.

Все параметры задаются здесь.
"""

# ─── Режим обучения ───────────────────────────────────────────────
# В этом файле реализовано только обучение tag_unit.
TRAINING_MODE = "tag_unit"

# ─── Параметры моделей ────────────────────────────────────────────
# ParNetTag
TAG_MODEL_HIDDEN_DIM = 256
TAG_MODEL_NUM_LAYERS = 3
TAG_MODEL_ACTIVATION = "gelu"

# UnitParnet1
UNIT1_HIDDEN_DIM = 64

# UnitParnet2
UNIT2_HIDDEN_DIM = 32

# ─── Данные ───────────────────────────────────────────────────────
DATASET_DIR_TAG = "./prepared_dataset_tag_image"   # папка с .pt и .txt
TAG_SEQ_MAX_LEN = 32                               # максимальное количество тегов
TAG_BYTE_LEN = 128                                 # длина байтовой последовательности одного тега
PAD_BYTE_VALUE = 0                                 # байт паддинга коротких тегов

# ─── Шум ──────────────────────────────────────────────────────────
NOISE_SEED = 42                                    # базовый сид для генерации шума
NOISE_STD = 0.5                                    # стандартное отклонение гауссова шума

# ─── Обучение ──────────────────────────────────────────────────────
BATCH_SIZE = 3
LEARNING_RATE = 0.00001
NUM_EPOCHS = 10000
RANDOM_SEED = 1234

# ─── Чекпоинты ────────────────────────────────────────────────────
MODELS_DIR_TAG_UNIT = "./models_tag_unit"
MAX_CHECKPOINTS = 5

# ─── Валидация / тесты ────────────────────────────────────────────
VALIDATION_SPLIT = 0               # сколько примеров оставить на валидацию
MAX_TRAIN_IMAGES = 3             # ограничение числа тренировочных примеров (None – все)
VAL_EVERY_EPOCHS = 100
TEST_EVERY_EPOCHS = 200
SAVE_EVERY_EPOCHS = 10
NUM_TEST_EXAMPLES = 3               # число примеров для визуализации
TEST_SEED = 5678
CLEAR_CACHE_EACH_BATCH = True

# Папки для сохранения результатов
TESTS_DIR = "./tests"               # тесты (примеры из обучающей выборки)
VAL_TESTS_DIR = "./val_tests"       # валидация (примеры из валидационной выборки)

# Для визуализации: путь к инференс-декодеру (TorchScript)
DECODER_INFERENCE_PATH = "./models/decoder_inference.pt"

# Потеря
PARNET_DIFF_LOSS_WEIGHT = 100.0       # вес для гипотетической потери

# Максимальное количество временных шагов на батч
MAX_TIME_STEPS = 2