# training_models_TagUnit.py
import os, re, glob, math, random, gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from model_parnet_tag import ParNetTag
from model_Unit_parnet import UnifiedUnitParnet
import config_training_tag_unit as cfg

# ---------------------- Устройства ----------------------
DEVICE_TAG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_UNIT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_DECODER = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Devices – Tag: {DEVICE_TAG}, UnifiedUnit: {DEVICE_UNIT}")

# ---------------------- Датасет ----------------------
class ParnetTagDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pt_files = {}
        self.txt_files = {}
        for pt_path in glob.glob(os.path.join(root_dir, "*.pt")):
            fname = os.path.splitext(os.path.basename(pt_path))[0]
            self.pt_files[fname] = pt_path
        for txt_path in glob.glob(os.path.join(root_dir, "*.txt")):
            fname = os.path.splitext(os.path.basename(txt_path))[0]
            self.txt_files[fname] = txt_path
        common_keys = sorted(set(self.pt_files.keys()) & set(self.txt_files.keys()),
                             key=lambda x: x)
        self.keys = common_keys
        if not self.keys:
            raise RuntimeError(f"No matching .pt and .txt pairs in {root_dir}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = torch.load(self.pt_files[key], map_location='cpu', weights_only=False)
        parnet_compressed = data['parnet_compressed']
        with open(self.txt_files[key], 'r', encoding='utf-8') as f:
            raw_line = f.read().strip()
            tags = [tag.strip() for tag in raw_line.split(',') if tag.strip()] if raw_line else []
        return {
            'parnet_compressed': parnet_compressed,
            'tags': tags,
            'example_id': key
        }

# ---------------------- Кодирование тегов ----------------------
def encode_tags(tags_list, max_len=32, byte_len=128, pad_byte=0):
    encoded = []
    for tag in tags_list:
        tag_bytes = tag.encode('utf-8')[:byte_len]
        if len(tag_bytes) < byte_len:
            tag_bytes = tag_bytes + bytes([pad_byte] * (byte_len - len(tag_bytes)))
        encoded.append(torch.tensor(list(tag_bytes), dtype=torch.float32))
    if not encoded:
        encoded.append(torch.zeros(byte_len))
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        while len(encoded) < max_len:
            encoded.append(torch.zeros(byte_len))
    return torch.stack(encoded, dim=0)

# ---------------------- Коллат-функция ----------------------
def collate_tag_fn(batch):
    parnets = []
    tags_batch = []
    noises = []
    example_ids = []
    num_tags_list = []
    for item in batch:
        parnet = item['parnet_compressed']
        tags = item['tags']
        example_id = item['example_id']
        seed = hash(example_id) % (2**31)
        g = torch.Generator()
        g.manual_seed(seed)
        noise = torch.randn(parnet.shape, generator=g) * cfg.NOISE_STD
        tags_tensor = encode_tags(tags, max_len=cfg.TAG_SEQ_MAX_LEN, byte_len=cfg.TAG_BYTE_LEN, pad_byte=cfg.PAD_BYTE_VALUE)
        num_real = min(len(tags), cfg.TAG_SEQ_MAX_LEN) if tags else 1
        parnets.append(parnet)
        noises.append(noise)
        tags_batch.append(tags_tensor)
        example_ids.append(example_id)
        num_tags_list.append(num_real)
    parnet_clean = torch.stack(parnets, dim=0)
    parnet_noisy = parnet_clean + torch.stack(noises, dim=0)
    noise_batch = torch.stack(noises, dim=0)
    tags_batch = torch.stack(tags_batch, dim=0)
    num_tags_tensor = torch.tensor(num_tags_list, dtype=torch.long)
    return {
        'parnet_clean': parnet_clean,
        'parnet_noisy': parnet_noisy,
        'noise': noise_batch,
        'tags_raw': tags_batch,
        'num_tags': num_tags_tensor,
        'example_ids': example_ids
    }

# ---------------------- Потери ----------------------
def difference_loss(pred, target):
    return torch.mean(torch.log(1.0 + torch.abs(pred - target)))

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(2.0) - 10 * math.log10(mse.item())

# ---------------------- Подготовка tag_parnets ----------------------
def compute_tag_parnets(tags_raw, num_tags, tag_model, device):
    B, max_len, byte_len = tags_raw.shape
    real_tags_list = []
    for i in range(B):
        n = num_tags[i].item()
        real_tags_list.append(tags_raw[i, :n])
    if real_tags_list:
        real_tags_batch = torch.cat(real_tags_list, dim=0).to(device)
    else:
        real_tags_batch = torch.empty(0, byte_len, device=device)
    if real_tags_batch.numel() > 0:
        tag_parnets_real = tag_model(real_tags_batch)
    else:
        tag_parnets_real = real_tags_batch
    tag_parnets = torch.zeros(B, byte_len, max_len, device=device)
    idx = 0
    for i in range(B):
        n = num_tags[i].item()
        if n > 0:
            tag_parnets[i, :, :n] = tag_parnets_real[idx:idx + n].transpose(0, 1)
            idx += n
    return tag_parnets

# ---------------------- Чекпоинты ----------------------
def get_model_path(name, epoch, models_dir):
    return os.path.join(models_dir, f"{name}_epoch{epoch}.pth")

def find_latest_checkpoint(name, models_dir):
    files = glob.glob(os.path.join(models_dir, f"{name}_epoch*.pth"))
    if not files:
        return None, 0
    def extract_epoch(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=extract_epoch)
    return latest, extract_epoch(latest)

def cleanup_old_checkpoints(models_dir, keep=cfg.MAX_CHECKPOINTS):
    for name in ["tag", "unified"]:
        files = glob.glob(os.path.join(models_dir, f"{name}_epoch*.pth"))
        if len(files) <= keep:
            continue
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass

def save_checkpoints(epoch, tag_model, opt_tag, unified_model, opt_unified, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for name, model, opt in [("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)]:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, get_model_path(name, epoch, models_dir))
    cleanup_old_checkpoints(models_dir)

def load_checkpoints_if_exist(tag_model, opt_tag, unified_model, opt_unified, models_dir):
    loaded_epoch = 0
    for name, model, opt in [("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)]:
        path, epoch = find_latest_checkpoint(name, models_dir)
        if path:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"Loaded {name} from epoch {epoch}")
            if loaded_epoch == 0:
                loaded_epoch = epoch
            else:
                assert epoch == loaded_epoch, f"Epoch mismatch for {name}"
    return loaded_epoch

# ---------------------- Загрузка инференс-моделей ----------------------
def load_inference_decoder(decoder_path):
    if not os.path.exists(decoder_path):
        return None
    model = torch.jit.load(decoder_path, map_location=DEVICE_DECODER)
    model.eval()
    return model

def load_inference_decompressor(decompressor_path):
    if not os.path.exists(decompressor_path):
        return None
    model = torch.jit.load(decompressor_path, map_location=DEVICE_DECODER)
    model.eval()
    return model

# ---------------------- Конвертация сжатого парнета в PIL ----------------------
def compressed_parnet_to_pil(compressed_parnet, decompressor, decoder):
    with torch.no_grad():
        x = compressed_parnet.unsqueeze(0).to(DEVICE_DECODER)
        full_parnet = decompressor(x)
        rgb = decoder(full_parnet)
        rgb = rgb.squeeze(0).cpu()
        arr = (rgb.clamp(-1, 1) + 1) / 2 * 255
        arr = arr.permute(1, 2, 0).to(torch.uint8).numpy()
        return Image.fromarray(arr)

# ---------------------- Тренировочная эпоха (две фазы) ----------------------
def train_epoch_tag_unit(tag_model, unified_model, train_loader, opt_tag, opt_unified):
    tag_model.train()
    unified_model.train()
    total_loss_epoch = 0.0
    n_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        parnet_clean = batch['parnet_clean'].to(DEVICE_UNIT)
        parnet_noisy = batch['parnet_noisy'].to(DEVICE_UNIT)
        noise = batch['noise'].to(DEVICE_UNIT)
        tags_raw = batch['tags_raw'].to(DEVICE_TAG)
        num_tags = batch['num_tags'].to(DEVICE_TAG)

        # --- Фаза 1: UnifiedUnitParnet ---
        for p in tag_model.parameters():
            p.requires_grad = False
        for p in unified_model.parameters():
            p.requires_grad = True

        with torch.no_grad():
            tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT)
        pred = unified_model(parnet_noisy, tag_parnets, noise)
        loss1 = difference_loss(pred, parnet_clean)
        loss1.backward()
        grads_unified = {name: param.grad.clone().cpu() for name, param in unified_model.named_parameters() if param.grad is not None}
        tag_model.zero_grad(); unified_model.zero_grad()
        l1 = loss1.item()

        # --- Фаза 2: ParNetTag ---
        for p in tag_model.parameters():
            p.requires_grad = True
        for p in unified_model.parameters():
            p.requires_grad = False

        tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT)
        pred = unified_model(parnet_noisy, tag_parnets, noise)
        loss2 = difference_loss(pred, parnet_clean)
        loss2.backward()
        grads_tag = {name: param.grad.clone().cpu() for name, param in tag_model.named_parameters() if param.grad is not None}
        tag_model.zero_grad(); unified_model.zero_grad()
        l2 = loss2.item()

        # Применяем градиенты
        for name, param in unified_model.named_parameters():
            param.grad = grads_unified[name].to(param.device) if name in grads_unified else None
        opt_unified.step(); opt_unified.zero_grad()

        for name, param in tag_model.named_parameters():
            param.grad = grads_tag[name].to(param.device) if name in grads_tag else None
        opt_tag.step(); opt_tag.zero_grad()

        total_loss_epoch += l1 + l2
        print(f"Batch {batch_idx+1}/{n_batches} | Ph1(Unified):{l1:.6f} Ph2(Tag):{l2:.6f}")

        if cfg.CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss_epoch / (2 * n_batches)

# ---------------------- Сбор данных для валидации ----------------------
def collect_validation_examples(tag_model, unified_model, val_loader, num_examples):
    tag_model.eval(); unified_model.eval()
    sum_loss = 0.0; sum_psnr = 0.0; n_batches = 0
    examples = []

    with torch.no_grad():
        for batch in val_loader:
            parnet_clean = batch['parnet_clean'].to(DEVICE_UNIT)
            parnet_noisy = batch['parnet_noisy'].to(DEVICE_UNIT)
            noise = batch['noise'].to(DEVICE_UNIT)
            tags_raw = batch['tags_raw'].to(DEVICE_TAG)
            num_tags = batch['num_tags'].to(DEVICE_TAG)
            B = parnet_clean.size(0)

            tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT)
            pred = unified_model(parnet_noisy, tag_parnets, noise)

            loss = difference_loss(pred, parnet_clean)
            sum_loss += loss.item()
            sum_psnr += compute_psnr(pred, parnet_clean)
            n_batches += 1

            for i in range(B):
                if len(examples) >= num_examples:
                    break
                ex_id = batch['example_ids'][i]
                ex_loss = difference_loss(pred[i].unsqueeze(0), parnet_clean[i].unsqueeze(0)).item()
                ex_psnr = compute_psnr(pred[i].unsqueeze(0), parnet_clean[i].unsqueeze(0))
                examples.append({
                    'id': ex_id,
                    'orig': parnet_clean[i].cpu(),
                    'noisy': parnet_noisy[i].cpu(),
                    'pred': pred[i].cpu(),
                    'loss': ex_loss,
                    'psnr': ex_psnr
                })

    avg_loss = sum_loss / n_batches
    avg_psnr = sum_psnr / n_batches
    return avg_loss, avg_psnr, examples

# ---------------------- Сохранение примера ----------------------
def save_example_images(base_dir, ex, decompressor, decoder):
    os.makedirs(base_dir, exist_ok=True)

    orig_img = compressed_parnet_to_pil(ex['orig'], decompressor, decoder)
    orig_img.save(os.path.join(base_dir, "original_decoded.png"))

    pred_img = compressed_parnet_to_pil(ex['pred'], decompressor, decoder)
    pred_img.save(os.path.join(base_dir, "predicted_decoded.png"))
    diff_pred = np.abs(np.array(orig_img).astype(np.float32) - np.array(pred_img).astype(np.float32)).astype(np.uint8)
    Image.fromarray(diff_pred).save(os.path.join(base_dir, "difference_predicted_decoded.png"))

    with open(os.path.join(base_dir, "metrics.txt"), 'w') as f:
        f.write(f"Loss: {ex['loss']:.6f}\nPSNR: {ex['psnr']:.2f} dB\n")

# ---------------------- Валидация ----------------------
def run_validation_tag_unit(tag_model, unified_model, val_loader, epoch, models_dir,
                            opt_tag, opt_unified, decoder, decompressor):
    num_examples = cfg.NUM_TEST_EXAMPLES
    avg_loss, avg_psnr, examples = collect_validation_examples(tag_model, unified_model, val_loader, num_examples)
    print(f"Validation Epoch {epoch}: Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f} dB")

    if not examples or decompressor is None or decoder is None:
        if decompressor is None:
            print("Визуализация пропущена: декомпрессор не загружен.")
        tag_model.train(); unified_model.train()
        return avg_loss, avg_psnr, tag_model, unified_model, opt_tag, opt_unified

    print("Данные для валидации собраны и перенесены в RAM. Выгружаю обучающие модели...")

    temp_paths = []
    for name, model, opt in [("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)]:
        path = os.path.join(models_dir, f"temp_val_{name}_restore.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, path)
        temp_paths.append(path)

    del tag_model, unified_model, opt_tag, opt_unified
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ex in examples:
        base_dir = os.path.join(cfg.VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
        save_example_images(base_dir, ex, decompressor, decoder)

    tag_model = ParNetTag().to(DEVICE_TAG)
    unified_model = UnifiedUnitParnet().to(DEVICE_UNIT)
    opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
    opt_unified = optim.Adam(unified_model.parameters(), lr=cfg.LEARNING_RATE)

    for (name, model, opt), path in zip([("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)], temp_paths):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        os.remove(path)

    tag_model.train(); unified_model.train()
    return avg_loss, avg_psnr, tag_model, unified_model, opt_tag, opt_unified

# ---------------------- Сбор данных для тестирования ----------------------
def collect_test_examples(tag_model, unified_model, dataset, num_examples):
    tag_model.eval(); unified_model.eval()
    random.seed(cfg.TEST_SEED)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    examples = []

    for idx in indices:
        item = dataset[idx]
        parnet = item['parnet_compressed'].unsqueeze(0).to(DEVICE_UNIT)
        tags = item['tags']
        example_id = item['example_id']

        seed = hash(example_id) % (2**31)
        g = torch.Generator()
        g.manual_seed(seed)
        noise = torch.randn(parnet.shape, generator=g).to(DEVICE_UNIT) * cfg.NOISE_STD
        noisy_parnet = parnet + noise

        tags_tensor = encode_tags(tags, max_len=cfg.TAG_SEQ_MAX_LEN, byte_len=cfg.TAG_BYTE_LEN, pad_byte=cfg.PAD_BYTE_VALUE)
        tags_tensor = tags_tensor.unsqueeze(0).to(DEVICE_TAG)
        num_tags = torch.tensor([min(len(tags), cfg.TAG_SEQ_MAX_LEN) if tags else 1], device=DEVICE_TAG)

        with torch.no_grad():
            tag_parnets = compute_tag_parnets(tags_tensor, num_tags, tag_model, DEVICE_UNIT)
            pred = unified_model(noisy_parnet, tag_parnets, noise)

        loss = difference_loss(pred, parnet).item()
        psnr_val = compute_psnr(pred, parnet)

        examples.append({
            'id': example_id,
            'orig': parnet.squeeze(0).cpu(),
            'noisy': noisy_parnet.squeeze(0).cpu(),
            'pred': pred.squeeze(0).cpu(),
            'loss': loss,
            'psnr': psnr_val
        })

    return examples

# ---------------------- Тестирование ----------------------
def run_tests_tag_unit(tag_model, unified_model, train_dataset, epoch, models_dir,
                       opt_tag, opt_unified, decoder, decompressor):
    num_examples = cfg.NUM_TEST_EXAMPLES
    examples = collect_test_examples(tag_model, unified_model, train_dataset, num_examples)
    if not examples or decompressor is None or decoder is None:
        if decompressor is None:
            print("Тестовая визуализация пропущена: декомпрессор не загружен.")
        tag_model.train(); unified_model.train()
        return tag_model, unified_model, opt_tag, opt_unified

    print("Тестовые данные собраны и находятся в RAM. Выгружаю обучающие модели...")

    temp_paths = []
    for name, model, opt in [("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)]:
        path = os.path.join(models_dir, f"temp_test_{name}_restore.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, path)
        temp_paths.append(path)

    del tag_model, unified_model, opt_tag, opt_unified
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ex in examples:
        base_dir = os.path.join(cfg.TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
        save_example_images(base_dir, ex, decompressor, decoder)

    tag_model = ParNetTag().to(DEVICE_TAG)
    unified_model = UnifiedUnitParnet().to(DEVICE_UNIT)
    opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
    opt_unified = optim.Adam(unified_model.parameters(), lr=cfg.LEARNING_RATE)

    for (name, model, opt), path in zip([("tag", tag_model, opt_tag), ("unified", unified_model, opt_unified)], temp_paths):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        os.remove(path)

    tag_model.train(); unified_model.train()
    print(f"Test examples for epoch {epoch} saved.")
    return tag_model, unified_model, opt_tag, opt_unified

# ---------------------- Основная функция ----------------------
def train():
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    decoder = load_inference_decoder(cfg.DECODER_INFERENCE_PATH)
    if decoder is None:
        print("Warning: Decoder not found – visualization skipped.")
    decompressor = load_inference_decompressor(cfg.DECOMPRESSOR_INFERENCE_PATH)
    if decompressor is None:
        print("Warning: Decompressor not found – visualization will not be possible.")

    dataset = ParnetTagDataset(cfg.DATASET_DIR_TAG)
    n_total = len(dataset)
    n_val = min(cfg.VALIDATION_SPLIT, n_total)
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError("Not enough data for training.")

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_tag_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_tag_fn, num_workers=0, pin_memory=True) if n_val > 0 else None

    tag_model = ParNetTag().to(DEVICE_TAG)
    unified_model = UnifiedUnitParnet().to(DEVICE_UNIT)

    opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
    opt_unified = optim.Adam(unified_model.parameters(), lr=cfg.LEARNING_RATE)

    models_dir = cfg.MODELS_DIR_TAG_UNIT
    start_epoch = load_checkpoints_if_exist(tag_model, opt_tag, unified_model, opt_unified, models_dir) + 1

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_total = train_epoch_tag_unit(tag_model, unified_model, train_loader, opt_tag, opt_unified)
        print(f"Epoch {epoch:3d} Average Total: {avg_total:.6f}")

        if val_loader and epoch % cfg.VAL_EVERY_EPOCHS == 0:
            val_loss, val_psnr, tag_model, unified_model, opt_tag, opt_unified = \
                run_validation_tag_unit(
                    tag_model, unified_model, val_loader, epoch, models_dir,
                    opt_tag, opt_unified, decoder, decompressor
                )

        if epoch % cfg.TEST_EVERY_EPOCHS == 0:
            print(f"Running tests for epoch {epoch}...")
            tag_model, unified_model, opt_tag, opt_unified = \
                run_tests_tag_unit(
                    tag_model, unified_model, train_dataset, epoch, models_dir,
                    opt_tag, opt_unified, decoder, decompressor
                )

        if epoch % cfg.SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, tag_model, opt_tag, unified_model, opt_unified, models_dir)
            print(f"Checkpoints saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    train()