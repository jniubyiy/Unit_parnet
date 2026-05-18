# config_training_tag_unit.py
"""
Конфигурация для обучения моделей tag, unified_unit.
"""

# ======================== ParNetTag =============================
TAG_MODEL_HIDDEN_DIM = 256
TAG_MODEL_NUM_LAYERS = 3
TAG_MODEL_DROPOUT = 0.1          # dropout между слоями MLP (0 = без dropout)

# ======================== UnifiedUnitParnet =====================
UNIFIED_IN_CHANNELS = 4
UNIFIED_OUT_CHANNELS = 4

# Текстовая ветвь (поиск «что нельзя удалять»)
UNIFIED_TXT_FEAT_HIDDEN = 64
UNIFIED_TXT_FEAT_LAYERS = 2
UNIFIED_TXT_FEAT_DROPOUT = 0.0

UNIFIED_TXT_ENCODER_HIDDEN = 64
UNIFIED_TXT_ENCODER_DROPOUT = 0.0

UNIFIED_ATTENTION_DROPOUT = 0.0

# Шумовая ветвь (поиск «что нужно удалить»)
UNIFIED_NOISE_ENC_HIDDEN = 32
UNIFIED_NOISE_ENC_LAYERS = 2
UNIFIED_NOISE_ENC_DROPOUT = 0.0

# Объединение признаков
UNIFIED_FUSION_HIDDEN = 32
UNIFIED_FUSION_LAYERS = 2
UNIFIED_FUSION_DROPOUT = 0.0

# Предсказание остаточного шума
UNIFIED_RESIDUAL_HIDDEN = 32
UNIFIED_RESIDUAL_LAYERS = 2
UNIFIED_RESIDUAL_DROPOUT = 0.0

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
