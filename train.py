import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from typing import List, Optional
from tqdm import tqdm

from model import SlideDuplicateModel, NTXentLoss
from dataset import SlideDuplicateDataset

def find_image_files(directory_or_file: str) -> List[str]:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if os.path.isfile(directory_or_file): return [directory_or_file]

    files = []
    for root, _, filenames in os.walk(directory_or_file):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts: files.append(os.path.join(root, f))
    return sorted(files)

def _default_num_workers() -> int:
    return 0

def train_slide_model(
    data_path: str,
    output_checkpoint: str = "slide_duplicate_model.pt",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    temperature: float = 0.07,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    use_amp: Optional[bool] = None
):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"

    # Throughput-oriented backend flags
    if device == "cuda":
        # Let cuDNN autotune conv kernels for the fixed 224x224 input size.
        torch.backends.cudnn.benchmark = True
        # Allow TF32 matmuls on Ampere+ (harmless on older GPUs).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if num_workers is None: num_workers = _default_num_workers()
    if use_amp is None: use_amp = (device == "cuda")

    # Pick AMP compute dtype. fp16 is the fastest on Turing (RTX 20xx, sm_75) and
    # most consumer GPUs; bf16 is only natively accelerated on Ampere+ (sm_80+).
    # Note: torch.cuda.is_bf16_supported() returns True on Turing too, but bf16 is
    # merely emulated there and ends up slower than fp16, so we gate on the real
    # hardware capability. The NT-Xent loss upcasts its similarity matrix to fp32,
    # so fp16 is numerically safe.
    if use_amp and device == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float16

    print(f"[+] Iniciando treinamento do modelo especializado em slides...")
    print(f"   Dispositivo: {device}")
    print(f"   Origem dos dados: {data_path}")

    all_files = find_image_files(data_path)
    if not all_files:
        raise ValueError(f"Nenhuma imagem válida encontrada em: {data_path}")

    if max_samples and len(all_files) > max_samples:
        print(f"   Limitando a {max_samples} imagens para este ciclo de treinamento (total disponível: {len(all_files)}).")
        all_files = all_files[:max_samples]
    else:
        print(f"   Total de imagens a serem utilizadas: {len(all_files)}")

    # Build dataset and dataloader
    dataset = SlideDuplicateDataset(all_files, target_size=224, cache_film_crops=True)

    # Pre-warm the film-crop cache once, in parallel, BEFORE the training loop.
    # This moves the expensive cold-cache work (14MP JPEG decode + OpenCV contour
    # detection) out of the loop using a thread pool, avoiding the Windows
    # process-spawn penalty of num_workers>0. After this, __getitem__ only does
    # cheap augmentation and the loop stays GPU-bound.
    prewarm_workers = min(8, os.cpu_count() or 4)
    t_prewarm = time.time()
    n_decoded = dataset.prewarm_cache(max_workers=prewarm_workers)
    print(f"   Pré-aquecendo cache de {n_decoded} crops em {time.time() - t_prewarm:.1f}s "
          f"({prewarm_workers} threads).")

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=num_workers,
    )
    if num_workers > 0:
        # Overlap CPU data preparation with GPU compute.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    if device == "cuda":
        # Speed up host->device transfers.
        loader_kwargs["pin_memory"] = True

    dataloader = DataLoader(dataset, **loader_kwargs)
    print(f"   DataLoader: num_workers={num_workers}, pin_memory={loader_kwargs.get('pin_memory', False)}, AMP={'on' if use_amp else 'off'}")

    # Instantiate model and loss
    model = SlideDuplicateModel(model_name="facebook/dinov2-small", projection_dim=256)
    model.to(device)

    # channels_last is faster on CUDA tensor cores for conv-based backbones.
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # We fine-tune the projection head and the last block of ViT backbone for stability & domain adaptation
    # Freeze initial blocks to preserve general vision foundation features
    for name, param in model.backbone.named_parameters():
        if "encoder.layer.11" not in name and "encoder.layer.10" not in name:
            param.requires_grad = False

    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )

    # AMP GradScaler (bf16 autocast does not need loss scaling; enabled=False
    # keeps this a correct no-op when AMP is off or on CPU).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")

    model.train()
    print(f"   Parâmetros treináveis configurados.")
    print(f"   Épocas: {epochs} | Tamanho do Lote: {batch_size} | Temp: {temperature}\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        step_count = 0
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{epochs}")
        for view1, view2 in pbar:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if device == "cuda":
                view1 = view1.to(memory_format=torch.channels_last)
                view2 = view2.to(memory_format=torch.channels_last)

            # Concatenate both augmented views into a single batch and run ONE
            # forward/backward pass. This roughly halves kernel-launch overhead
            # and keeps the GPU better utilized than two small (B) passes.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                batch = torch.cat([view1, view2], dim=0)
                z = model(batch)
                z1, z2 = z.chunk(2, dim=0)
                loss, acc = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item() * 100:.1f}%"})

        avg_loss = epoch_loss / max(1, step_count)
        avg_acc = epoch_acc / max(1, step_count)
        print(f"   [OK] Fim da Epoca {epoch}/{epochs} - Loss Medio: {avg_loss:.4f} | Acuracia: {avg_acc * 100:.2f}%")

        # Save a checkpoint at the end of every completed epoch. The SAME file is
        # overwritten each epoch (no per-epoch snapshots), so training can be
        # resumed/inspected if interrupted and the file always holds the latest
        # weights for inference.
        save_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_name": "facebook/dinov2-small",
            "projection_dim": 256,
            "temperature": temperature,
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
        torch.save(save_data, output_checkpoint)
        print(f"       Checkpoint (epoca {epoch}) salvo em: {os.path.abspath(output_checkpoint)}")

    # The canonical output file already holds the latest epoch's weights (saved
    # inside the loop after each epoch), so no extra save is needed here.
    print(f"\n[SUCESSO] Treinamento concluido!")
    print(f"   Pesos salvos em: {os.path.abspath(output_checkpoint)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento auto-supervisionado contrastivo para slides")
    parser.add_argument("--data", type=str, default=r"C:\digitalização", help="Caminho da pasta com as fotos")
    parser.add_argument("--output", type=str, default="slide_duplicate_model.pt", help="Arquivo de saída do modelo treinado")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=500, help="Máximo de amostras para treino")
    parser.add_argument("--num_workers", type=int, default=None, help="Workers do DataLoader (padrão: 0; o cache é pré-aquecido com threads)")
    parser.add_argument("--no_amp", action="store_true", help="Desativa precision mista (AMP) na GPU")
    args = parser.parse_args()

    train_slide_model(
        data_path=args.data,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        use_amp=not args.no_amp
    )
