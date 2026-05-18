# training_models_TagUnit.py

import os, re, glob, math, random, gc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from model_parnet_tag import ParNetTag
from model_Unit_parnet_1 import UnitParnet1
from model_Unit_parnet_2 import UnitParnet2

import config_training_tag_unit as cfg

# ---------------------- Устройства ----------------------
DEVICE_TAG   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_UNIT1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_UNIT2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_DECODER = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Devices – Tag: {DEVICE_TAG}, Unit1: {DEVICE_UNIT1}, Unit2: {DEVICE_UNIT2}")

# ---------------------- Датасет ----------------------
class ParnetTagDataset(Dataset):
    """Загружает парнеты и теги из папки: файлы <id>.pt и <id>.txt"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pt_files = {}
        self.txt_files = {}
        all_pts = glob.glob(os.path.join(root_dir, "*.pt"))
        for pt_path in all_pts:
            fname = os.path.splitext(os.path.basename(pt_path))[0]
            self.pt_files[fname] = pt_path
        all_txts = glob.glob(os.path.join(root_dir, "*.txt"))
        for txt_path in all_txts:
            fname = os.path.splitext(os.path.basename(txt_path))[0]
            self.txt_files[fname] = txt_path
        common_keys = sorted(set(self.pt_files.keys()) & set(self.txt_files.keys()),
                             key=lambda x: int(x) if x.isdigit() else x)
        self.keys = common_keys
        if not self.keys:
            raise RuntimeError(f"No matching .pt and .txt pairs in {root_dir}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = torch.load(self.pt_files[key], map_location='cpu', weights_only=False)
        parnet = data['parnet']  # [3, H, W] в [-1,1]
        with open(self.txt_files[key], 'r', encoding='utf-8') as f:
            raw_line = f.read().strip()
        if raw_line:
            tags = [tag.strip() for tag in raw_line.split(',') if tag.strip()]
        else:
            tags = []
        return {
            'parnet': parnet,
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
        tag_tensor = torch.tensor(list(tag_bytes), dtype=torch.float32)
        encoded.append(tag_tensor)
    if len(encoded) == 0:
        encoded.append(torch.zeros(byte_len))
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        while len(encoded) < max_len:
            encoded.append(torch.zeros(byte_len))
    return torch.stack(encoded, dim=0)   # [max_len, byte_len]

# ---------------------- Коллат-функция ----------------------
def collate_tag_fn(batch):
    parnets = []
    tags_batch = []
    noises = []
    example_ids = []
    num_tags_list = []

    for item in batch:
        parnet = item['parnet']
        tags = item['tags']
        example_id = item['example_id']

        seed = cfg.NOISE_SEED + int(example_id) if example_id.isdigit() else cfg.NOISE_SEED
        g = torch.Generator()
        g.manual_seed(seed)
        noise = torch.randn(parnet.shape, generator=g) * cfg.NOISE_STD

        tags_tensor = encode_tags(tags, max_len=cfg.TAG_SEQ_MAX_LEN,
                                  byte_len=cfg.TAG_BYTE_LEN, pad_byte=cfg.PAD_BYTE_VALUE)
        num_real = min(len(tags), cfg.TAG_SEQ_MAX_LEN) if tags else 1

        parnets.append(parnet)
        noises.append(noise)
        tags_batch.append(tags_tensor)
        example_ids.append(example_id)
        num_tags_list.append(num_real)

    parnet_clean = torch.stack(parnets, dim=0)
    parnet_noisy = parnet_clean + torch.stack(noises, dim=0)
    noise_batch = torch.stack(noises, dim=0)
    tags_batch = torch.stack(tags_batch, dim=0)          # [B, 32, 128]
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
    """Преобразует теги в tag_parnets [B, 128, 32] на указанном устройстве."""
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

# ---------------------- Один временной шаг (без tag_model) ----------------------
def forward_step_with_tag_parnets(tag_parnets, parnet_input, noise, t_step_tensor, unit1, unit2):
    enriched = unit1(parnet_input, tag_parnets, t_step_tensor)
    pred = unit2(enriched, noise, t_step_tensor)
    return pred, enriched

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
    for name in ["tag", "unit1", "unit2"]:
        files = glob.glob(os.path.join(models_dir, f"{name}_epoch*.pth"))
        if len(files) <= keep:
            continue
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass

def save_checkpoints(epoch, tag_model, opt_tag, unit1, opt_unit1, unit2, opt_unit2, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for name, model, opt in [("tag", tag_model, opt_tag), ("unit1", unit1, opt_unit1), ("unit2", unit2, opt_unit2)]:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, get_model_path(name, epoch, models_dir))
    cleanup_old_checkpoints(models_dir)

def load_checkpoints_if_exist(tag_model, opt_tag, unit1, opt_unit1, unit2, opt_unit2, models_dir):
    loaded_epoch = 0
    for name, model, opt in [("tag", tag_model, opt_tag), ("unit1", unit1, opt_unit1), ("unit2", unit2, opt_unit2)]:
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

# ---------------------- Загрузка декодера ----------------------
def load_inference_decoder(decoder_path):
    if not os.path.exists(decoder_path):
        return None
    model = torch.jit.load(decoder_path, map_location=DEVICE_DECODER)
    model.eval()
    return model

# ---------------------- Визуализация ----------------------
def parnet_to_pil(t):
    arr = (t.clamp(-1, 1).cpu().numpy() + 1) / 2 * 255
    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(arr)

def save_example_images(base_dir, example_id,
                        original_parnet=None, predicted_parnet=None,
                        noisy_parnet=None, enriched_parnet=None,
                        loss_val=None, psnr_val=None, decoder=None,
                        step=None, max_steps=None, save_original=True):
    os.makedirs(base_dir, exist_ok=True)
    suffix = f"_step{step}" if step is not None else ""

    # Оригинал (RGB) – только если save_original и original_parnet передан
    if original_parnet is not None and save_original:
        parnet_to_pil(original_parnet).save(os.path.join(base_dir, f"{example_id}_original_parnet.png"))

    # Предсказание (RGB) – только если predicted_parnet передан
    if predicted_parnet is not None:
        parnet_to_pil(predicted_parnet).save(os.path.join(base_dir, f"{example_id}{suffix}_predicted_parnet.png"))
        if original_parnet is not None:
            diff_parnet = (predicted_parnet - original_parnet).abs()
            parnet_to_pil(diff_parnet).save(os.path.join(base_dir, f"{example_id}{suffix}_difference_parnet.png"))

    # Noisy (только один раз)
    if noisy_parnet is not None and (step is None or step == 1):
        parnet_to_pil(noisy_parnet).save(os.path.join(base_dir, f"{example_id}_noisy_parnet.png"))

    # Enriched (для каждого шага, если передан)
    if enriched_parnet is not None:
        parnet_to_pil(enriched_parnet).save(os.path.join(base_dir, f"{example_id}{suffix}_enriched_parnet.png"))

    # Декодированные изображения
    if decoder is not None:
        with torch.no_grad():
            # Декодированный оригинал (один раз)
            if original_parnet is not None and save_original and (step is None or step == 1):
                orig_dec = decoder(original_parnet.unsqueeze(0).to(DEVICE_DECODER)).squeeze(0).cpu()
                parnet_to_pil(orig_dec).save(os.path.join(base_dir, f"{example_id}_original_decoded.png"))

            # Декодированное предсказание (если есть predicted_parnet)
            if predicted_parnet is not None:
                pred_dec = decoder(predicted_parnet.unsqueeze(0).to(DEVICE_DECODER)).squeeze(0).cpu()
                parnet_to_pil(pred_dec).save(os.path.join(base_dir, f"{example_id}{suffix}_predicted_decoded.png"))
                if original_parnet is not None:
                    # Для разницы нужен оригинал; вычисляем его один раз (если ещё не вычислен) или повторно
                    orig_dec_for_diff = decoder(original_parnet.unsqueeze(0).to(DEVICE_DECODER)).squeeze(0).cpu()
                    diff_dec = (pred_dec - orig_dec_for_diff).abs()
                    parnet_to_pil(diff_dec).save(os.path.join(base_dir, f"{example_id}{suffix}_difference_decoded.png"))

            # Декодированный шумный (один раз)
            if noisy_parnet is not None and (step is None or step == 1):
                noisy_dec = decoder(noisy_parnet.unsqueeze(0).to(DEVICE_DECODER)).squeeze(0).cpu()
                parnet_to_pil(noisy_dec).save(os.path.join(base_dir, f"{example_id}_noisy_decoded.png"))

            # Декодированный enriched
            if enriched_parnet is not None:
                enriched_dec = decoder(enriched_parnet.unsqueeze(0).to(DEVICE_DECODER)).squeeze(0).cpu()
                parnet_to_pil(enriched_dec).save(os.path.join(base_dir, f"{example_id}{suffix}_enriched_decoded.png"))

    # Метрики только для последнего шага или если шаг не указан
    if loss_val is not None and psnr_val is not None:
        if step is None or (max_steps is not None and step == max_steps):
            with open(os.path.join(base_dir, f"{example_id}_metrics.txt"), 'w') as f:
                f.write(f"Loss: {loss_val:.6f}\nPSNR: {psnr_val:.2f} dB\n")

# ---------------------- Тренировочная эпоха ----------------------
def train_epoch_tag_unit(tag_model, unit1, unit2, train_loader, opt_tag, opt_unit1, opt_unit2):
    tag_model.train()
    unit1.train()
    unit2.train()
    total_loss_epoch = 0.0
    n_batches = len(train_loader)
    max_steps = cfg.MAX_TIME_STEPS

    for batch_idx, batch in enumerate(train_loader):
        parnet_clean = batch['parnet_clean'].to(DEVICE_UNIT2)
        parnet_noisy = batch['parnet_noisy'].to(DEVICE_UNIT2)
        noise = batch['noise'].to(DEVICE_UNIT2)
        tags_raw = batch['tags_raw'].to(DEVICE_TAG)
        num_tags = batch['num_tags'].to(DEVICE_TAG)
        B = parnet_clean.size(0)

        grads_tag = {}
        grads_unit1 = {}
        grads_unit2 = {}

        # --- Фаза 1: unit2 ---
        for p in tag_model.parameters(): p.requires_grad = False
        for p in unit1.parameters(): p.requires_grad = False
        for p in unit2.parameters(): p.requires_grad = True

        with torch.no_grad():
            tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT2)

        loss_phase1 = 0.0
        pred = None
        for step in range(1, max_steps+1):
            input_parnet = parnet_noisy if step == 1 else pred
            t_tensor = torch.full((B,), step, device=DEVICE_UNIT2, dtype=torch.float32)
            pred, _ = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
            loss_phase1 += difference_loss(pred, parnet_clean)
        loss_phase1.backward()

        for name, param in unit2.named_parameters():
            if param.grad is not None:
                grads_unit2[name] = param.grad.clone().cpu()
        tag_model.zero_grad(); unit1.zero_grad(); unit2.zero_grad()
        l1 = loss_phase1.item()

        # --- Фаза 2: unit1 ---
        for p in tag_model.parameters(): p.requires_grad = False
        for p in unit1.parameters(): p.requires_grad = True
        for p in unit2.parameters(): p.requires_grad = False

        with torch.no_grad():
            tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT2)

        loss_phase2 = 0.0
        pred = None
        for step in range(1, max_steps+1):
            input_parnet = parnet_noisy if step == 1 else pred
            t_tensor = torch.full((B,), step, device=DEVICE_UNIT2, dtype=torch.float32)
            pred, _ = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
            loss_phase2 += difference_loss(pred, parnet_clean)
        loss_phase2.backward()

        for name, param in unit1.named_parameters():
            if param.grad is not None:
                grads_unit1[name] = param.grad.clone().cpu()
        tag_model.zero_grad(); unit1.zero_grad(); unit2.zero_grad()
        l2 = loss_phase2.item()

        # --- Фаза 3: tag_model ---
        for p in tag_model.parameters(): p.requires_grad = True
        for p in unit1.parameters(): p.requires_grad = False
        for p in unit2.parameters(): p.requires_grad = False

        tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT2)

        loss_phase3 = 0.0
        pred = None
        for step in range(1, max_steps+1):
            input_parnet = parnet_noisy if step == 1 else pred
            t_tensor = torch.full((B,), step, device=DEVICE_UNIT2, dtype=torch.float32)
            pred, _ = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
            loss_phase3 += difference_loss(pred, parnet_clean)
        loss_phase3.backward()

        for name, param in tag_model.named_parameters():
            if param.grad is not None:
                grads_tag[name] = param.grad.clone().cpu()
        tag_model.zero_grad(); unit1.zero_grad(); unit2.zero_grad()
        l3 = loss_phase3.item()

        # Применение накопленных градиентов
        for name, param in unit2.named_parameters():
            param.grad = grads_unit2[name].to(param.device) if name in grads_unit2 else None
        opt_unit2.step(); opt_unit2.zero_grad(); grads_unit2.clear()

        for name, param in unit1.named_parameters():
            param.grad = grads_unit1[name].to(param.device) if name in grads_unit1 else None
        opt_unit1.step(); opt_unit1.zero_grad(); grads_unit1.clear()

        for name, param in tag_model.named_parameters():
            param.grad = grads_tag[name].to(param.device) if name in grads_tag else None
        opt_tag.step(); opt_tag.zero_grad(); grads_tag.clear()

        total_loss_epoch += l1 + l2 + l3
        print(f"Batch {batch_idx+1}/{n_batches} | Ph1(U2):{l1:.6f} Ph2(U1):{l2:.6f} Ph3(Tag):{l3:.6f}")

        if cfg.CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return total_loss_epoch / (3 * n_batches)

# ---------------------- Валидация ----------------------
def run_validation_tag_unit(tag_model, unit1, unit2, val_loader, epoch, models_dir,
                            opt_tag, opt_unit1, opt_unit2, decoder=None):
    tag_model.eval(); unit1.eval(); unit2.eval()
    sum_loss = 0.0; sum_psnr = 0.0; n_batches = 0
    examples = []
    max_steps = cfg.MAX_TIME_STEPS

    with torch.no_grad():
        for batch in val_loader:
            parnet_clean = batch['parnet_clean'].to(DEVICE_UNIT2)
            parnet_noisy = batch['parnet_noisy'].to(DEVICE_UNIT2)
            noise = batch['noise'].to(DEVICE_UNIT2)
            tags_raw = batch['tags_raw'].to(DEVICE_TAG)
            num_tags = batch['num_tags'].to(DEVICE_TAG)
            B = parnet_clean.size(0)

            tag_parnets = compute_tag_parnets(tags_raw, num_tags, tag_model, DEVICE_UNIT2)

            if len(examples) < cfg.NUM_TEST_EXAMPLES:
                preds_per_example = [[] for _ in range(B)]
                enricheds_per_example = [[] for _ in range(B)]
                pred = None
                for step in range(1, max_steps+1):
                    input_parnet = parnet_noisy if step == 1 else pred
                    t_tensor = torch.full((B,), step, device=DEVICE_UNIT2, dtype=torch.float32)
                    pred, enriched = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
                    for i in range(B):
                        preds_per_example[i].append(pred[i].cpu())
                        enricheds_per_example[i].append(enriched[i].cpu())
                loss = difference_loss(pred, parnet_clean)
                sum_loss += loss.item()
                sum_psnr += compute_psnr(pred, parnet_clean)
                n_batches += 1
                for i in range(B):
                    if len(examples) >= cfg.NUM_TEST_EXAMPLES: break
                    ex_id = batch['example_ids'][i]
                    examples.append({
                        'id': ex_id,
                        'orig': parnet_clean[i].cpu(),
                        'noisy': parnet_noisy[i].cpu(),
                        'preds': preds_per_example[i],
                        'enricheds': enricheds_per_example[i],
                        'loss': difference_loss(preds_per_example[i][-1].unsqueeze(0), parnet_clean[i].unsqueeze(0)).item(),
                        'psnr': compute_psnr(preds_per_example[i][-1].unsqueeze(0), parnet_clean[i].unsqueeze(0))
                    })
            else:
                pred = None
                for step in range(1, max_steps+1):
                    input_parnet = parnet_noisy if step == 1 else pred
                    t_tensor = torch.full((B,), step, device=DEVICE_UNIT2, dtype=torch.float32)
                    pred, _ = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
                loss = difference_loss(pred, parnet_clean)
                sum_loss += loss.item()
                sum_psnr += compute_psnr(pred, parnet_clean)
                n_batches += 1

    avg_loss = sum_loss / n_batches
    avg_psnr = sum_psnr / n_batches
    print(f"Validation Epoch {epoch}: Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f} dB")

    if decoder is not None and examples:
        temp_paths = []
        for name, model, opt in [("tag", tag_model, opt_tag),
                                 ("unit1", unit1, opt_unit1),
                                 ("unit2", unit2, opt_unit2)]:
            path = os.path.join(models_dir, f"temp_val_{name}_restore.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, path)
            temp_paths.append(path)

        del tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for ex in examples:
            base_dir = os.path.join(cfg.VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
            # Базовый вызов: оригинал, шумный парнет, метрики (без predicted)
            save_example_images(base_dir, ex['id'],
                                original_parnet=ex['orig'],
                                noisy_parnet=ex['noisy'],
                                loss_val=ex['loss'], psnr_val=ex['psnr'],
                                decoder=decoder, step=None, save_original=True)
            # Для каждого временного шага: предсказание и enriched
            for step in range(1, max_steps+1):
                save_example_images(base_dir, ex['id'],
                                    original_parnet=ex['orig'],
                                    predicted_parnet=ex['preds'][step-1],
                                    enriched_parnet=ex['enricheds'][step-1],
                                    decoder=decoder, step=step, max_steps=max_steps,
                                    save_original=False)

        del decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tag_model = ParNetTag(hidden_dim=cfg.TAG_MODEL_HIDDEN_DIM,
                              num_layers=cfg.TAG_MODEL_NUM_LAYERS,
                              activation=cfg.TAG_MODEL_ACTIVATION).to(DEVICE_TAG)
        unit1 = UnitParnet1(hidden_dim=cfg.UNIT1_HIDDEN_DIM).to(DEVICE_UNIT1)
        unit2 = UnitParnet2(hidden_dim=cfg.UNIT2_HIDDEN_DIM).to(DEVICE_UNIT2)
        opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
        opt_unit1 = optim.Adam(unit1.parameters(), lr=cfg.LEARNING_RATE)
        opt_unit2 = optim.Adam(unit2.parameters(), lr=cfg.LEARNING_RATE)

        for (name, model, opt), path in zip([("tag", tag_model, opt_tag),
                                             ("unit1", unit1, opt_unit1),
                                             ("unit2", unit2, opt_unit2)], temp_paths):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            os.remove(path)

        tag_model.train(); unit1.train(); unit2.train()
        return avg_loss, avg_psnr, tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2

    tag_model.train(); unit1.train(); unit2.train()
    return avg_loss, avg_psnr, tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2

# ---------------------- Тестирование ----------------------
def run_tests_tag_unit(tag_model, unit1, unit2, train_dataset, epoch, models_dir,
                       opt_tag, opt_unit1, opt_unit2, decoder=None):
    tag_model.eval(); unit1.eval(); unit2.eval()
    max_steps = cfg.MAX_TIME_STEPS
    random.seed(cfg.TEST_SEED)
    indices = random.sample(range(len(train_dataset)), min(cfg.NUM_TEST_EXAMPLES, len(train_dataset)))

    examples = []
    for idx in indices:
        item = train_dataset[idx]
        parnet = item['parnet'].unsqueeze(0).to(DEVICE_UNIT2)
        tags = item['tags']
        example_id = item['example_id']

        seed = cfg.NOISE_SEED + int(example_id) if example_id.isdigit() else cfg.NOISE_SEED
        g = torch.Generator()
        g.manual_seed(seed)
        noise = torch.randn(parnet.shape, generator=g).to(DEVICE_UNIT2) * cfg.NOISE_STD
        noisy_parnet = parnet + noise

        tags_tensor = encode_tags(tags, max_len=cfg.TAG_SEQ_MAX_LEN,
                                  byte_len=cfg.TAG_BYTE_LEN, pad_byte=cfg.PAD_BYTE_VALUE)
        tags_tensor = tags_tensor.unsqueeze(0).to(DEVICE_TAG)
        num_tags = torch.tensor([min(len(tags), cfg.TAG_SEQ_MAX_LEN) if tags else 1], device=DEVICE_TAG)

        with torch.no_grad():
            tag_parnets = compute_tag_parnets(tags_tensor, num_tags, tag_model, DEVICE_UNIT2)
            preds = []; enricheds = []
            pred = None
            for step in range(1, max_steps+1):
                input_parnet = noisy_parnet if step == 1 else pred
                t_tensor = torch.full((1,), step, device=DEVICE_UNIT2, dtype=torch.float32)
                pred, enriched = forward_step_with_tag_parnets(tag_parnets, input_parnet, noise, t_tensor, unit1, unit2)
                preds.append(pred.squeeze(0).cpu())
                enricheds.append(enriched.squeeze(0).cpu())
            loss = difference_loss(pred, parnet).item()
            psnr_val = compute_psnr(pred, parnet)

        examples.append({
            'id': example_id,
            'orig': parnet.squeeze(0).cpu(),
            'noisy': noisy_parnet.squeeze(0).cpu(),
            'preds': preds,
            'enricheds': enricheds,
            'loss': loss,
            'psnr': psnr_val
        })

    if decoder is not None and examples:
        temp_paths = []
        for name, model, opt in [("tag", tag_model, opt_tag),
                                 ("unit1", unit1, opt_unit1),
                                 ("unit2", unit2, opt_unit2)]:
            path = os.path.join(models_dir, f"temp_test_{name}_restore.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, path)
            temp_paths.append(path)

        del tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for ex in examples:
            base_dir = os.path.join(cfg.TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
            # Базовый вызов: оригинал, шум, метрики
            save_example_images(base_dir, ex['id'],
                                original_parnet=ex['orig'],
                                noisy_parnet=ex['noisy'],
                                loss_val=ex['loss'], psnr_val=ex['psnr'],
                                decoder=decoder, step=None, save_original=True)
            # Для каждого шага
            for step in range(1, max_steps+1):
                save_example_images(base_dir, ex['id'],
                                    original_parnet=ex['orig'],
                                    predicted_parnet=ex['preds'][step-1],
                                    enriched_parnet=ex['enricheds'][step-1],
                                    decoder=decoder, step=step, max_steps=max_steps,
                                    save_original=False)

        del decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tag_model = ParNetTag(hidden_dim=cfg.TAG_MODEL_HIDDEN_DIM,
                              num_layers=cfg.TAG_MODEL_NUM_LAYERS,
                              activation=cfg.TAG_MODEL_ACTIVATION).to(DEVICE_TAG)
        unit1 = UnitParnet1(hidden_dim=cfg.UNIT1_HIDDEN_DIM).to(DEVICE_UNIT1)
        unit2 = UnitParnet2(hidden_dim=cfg.UNIT2_HIDDEN_DIM).to(DEVICE_UNIT2)
        opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
        opt_unit1 = optim.Adam(unit1.parameters(), lr=cfg.LEARNING_RATE)
        opt_unit2 = optim.Adam(unit2.parameters(), lr=cfg.LEARNING_RATE)

        for (name, model, opt), path in zip([("tag", tag_model, opt_tag),
                                             ("unit1", unit1, opt_unit1),
                                             ("unit2", unit2, opt_unit2)], temp_paths):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            os.remove(path)

        tag_model.train(); unit1.train(); unit2.train()
        print(f"Test examples for epoch {epoch} saved.")
    else:
        tag_model.train(); unit1.train(); unit2.train()

    return tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2

# ---------------------- Основная функция ----------------------
def train():
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    decoder = load_inference_decoder(cfg.DECODER_INFERENCE_PATH)
    if decoder is None:
        print("Warning: Decoder not found – visualization skipped.")

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

    tag_model = ParNetTag(hidden_dim=cfg.TAG_MODEL_HIDDEN_DIM,
                          num_layers=cfg.TAG_MODEL_NUM_LAYERS,
                          activation=cfg.TAG_MODEL_ACTIVATION).to(DEVICE_TAG)
    unit1 = UnitParnet1(hidden_dim=cfg.UNIT1_HIDDEN_DIM).to(DEVICE_UNIT1)
    unit2 = UnitParnet2(hidden_dim=cfg.UNIT2_HIDDEN_DIM).to(DEVICE_UNIT2)

    opt_tag = optim.Adam(tag_model.parameters(), lr=cfg.LEARNING_RATE)
    opt_unit1 = optim.Adam(unit1.parameters(), lr=cfg.LEARNING_RATE)
    opt_unit2 = optim.Adam(unit2.parameters(), lr=cfg.LEARNING_RATE)

    models_dir = cfg.MODELS_DIR_TAG_UNIT
    start_epoch = load_checkpoints_if_exist(tag_model, opt_tag, unit1, opt_unit1, unit2, opt_unit2, models_dir) + 1

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_total = train_epoch_tag_unit(tag_model, unit1, unit2, train_loader, opt_tag, opt_unit1, opt_unit2)
        print(f"Epoch {epoch:3d} Average Total: {avg_total:.6f}")

        if val_loader and epoch % cfg.VAL_EVERY_EPOCHS == 0:
            val_loss, val_psnr, tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2 = run_validation_tag_unit(
                tag_model, unit1, unit2, val_loader, epoch, models_dir,
                opt_tag, opt_unit1, opt_unit2, decoder
            )

        if epoch % cfg.TEST_EVERY_EPOCHS == 0:
            print(f"Running tests for epoch {epoch}...")
            tag_model, unit1, unit2, opt_tag, opt_unit1, opt_unit2 = run_tests_tag_unit(
                tag_model, unit1, unit2, train_dataset, epoch, models_dir,
                opt_tag, opt_unit1, opt_unit2, decoder
            )

        if epoch % cfg.SAVE_EVERY_EPOCHS == 0:
            save_checkpoints(epoch, tag_model, opt_tag, unit1, opt_unit1, unit2, opt_unit2, models_dir)
            print(f"Checkpoints saved at epoch {epoch}")

    print("Training completed.")

if __name__ == "__main__":
    train()