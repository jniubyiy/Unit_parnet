# preparing_the_dataset_parnet.py

import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import concurrent.futures

import config_preparing_the_dataset_parnet as cfg


def load_encoder(model_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """Загружает TorchScript-модель энкодера."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель энкодера не найдена: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def process_single_image(args_tuple):
    """
    Обрабатывает одно изображение: открывает, преобразует, прогоняет через энкодер.
    Сохраняет .pt файл с ключами 'parnet' и 'mask'.
    Принимает кортеж:
        (file_path_str, target_resolution, output_dir_str, encoder_model_path)
    """
    file_path_str, target_resolution, output_dir_str, encoder_model_path = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)

    # Загружаем энкодер один раз для процесса
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        encoder = load_encoder(encoder_model_path, device)
    except Exception as e:
        return (file_path.name, f"ERROR: failed to load encoder: {e}")

    try:
        # Открываем изображение
        img = Image.open(file_path).convert("RGB")
        w, h = img.size
        max_side = max(w, h)

        # Масштабирование, если нужно
        if max_side > target_resolution:
            ratio = target_resolution / max_side
            new_w = int(round(w * ratio))
            new_h = int(round(h * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        # Центрирование с чёрным паддингом
        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        offset_x = (target_resolution - w) // 2
        offset_y = (target_resolution - h) // 2
        canvas.paste(img, (offset_x, offset_y))

        # Бинарная маска
        mask = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
        mask[offset_y:offset_y + h, offset_x:offset_x + w] = 1

        # Преобразование в тензор [0, 1]
        img_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0

        # Нормализация в [-1, 1] (как ожидает энкодер)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(device)   # [1, 3, H, W]

        # Прогон через энкодер
        with torch.no_grad():
            parnet_tensor = encoder(img_tensor)          # [1, 3, H, W] в [-1, 1]

        # Переносим обратно на CPU, убираем batch-измерение
        parnet_tensor = parnet_tensor.squeeze(0).cpu()   # [3, H, W]
        mask_tensor = torch.from_numpy(mask).float()     # [H, W]

        # Сохранение
        number = file_path.stem
        save_path = output_path / f"{number}.pt"
        torch.save({"parnet": parnet_tensor, "mask": mask_tensor}, save_path)

        return (file_path.name, "OK")
    except Exception as e:
        return (file_path.name, f"ERROR: {e}")


def prepare_dataset_parnet(
    target_resolution: int,
    dataset_dir: str,
    output_dir: str,
    encoder_model_path: str
):
    """
    Параллельно обрабатывает изображения, конвертируя их в парнеты через энкодер.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg'}
    file_paths = []
    for f in sorted(dataset_path.iterdir()):
        if f.suffix.lower() in image_extensions:
            try:
                int(f.stem)
            except ValueError:
                print(f"Пропущен файл {f}: имя не является целым числом")
                continue
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих изображений в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} изображений. Целевое разрешение: {target_resolution}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    # Готовим аргументы для каждого процесса
    tasks = [(str(p), target_resolution, str(output_path), encoder_model_path) for p in file_paths]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            fname = Path(futures[future]).name
            try:
                result = future.result()
                if result[1] == "OK":
                    print(f"OK: {result[0]}")
                else:
                    print(f"FAIL: {result[0]} – {result[1]}")
            except Exception as e:
                print(f"FAIL: {fname} – исключение в процессе: {e}")

    print(f"Готово. Парнеты сохранены в {output_path}")


if __name__ == "__main__":
    if cfg.TARGET_RESOLUTION % 32 != 0:
        print("Предупреждение: разрешение не кратно 32, архитектура рассчитана на кратность 32.")

    prepare_dataset_parnet(
        target_resolution=cfg.TARGET_RESOLUTION,
        dataset_dir=cfg.DATASET_DIR,
        output_dir=cfg.OUTPUT_DIR,
        encoder_model_path=cfg.ENCODER_MODEL_PATH
    )