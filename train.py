import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from typing import List, Optional
from tqdm import tqdm

from model import SlideDuplicateModel, NTXentLoss

def find_image_files(directory_or_file: str) -> List[str]:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if os.path.isfile(directory_or_file): return [directory_or_file]

    files = []
    for root, _, filenames in os.walk(directory_or_file):
        for f in filenames:
            # Ignora resource-fork do macOS (._*.jpg), thumbs.db, arquivos
            # ocultos e arquivos minúsculos (0 bytes / placeholder quebrado).
            # Um desses estourou o prewarm com UnidentifiedImageError.
            if f.startswith('._') or f.startswith('.'):
                continue
            if f.lower() in {'thumbs.db', 'desktop.ini'}:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_exts:
                continue
            full = os.path.join(root, f)
            try:
                if os.path.getsize(full) < 1024:
                    continue
            except OSError:
                continue
            files.append(full)
    return sorted(files)

def _default_num_workers() -> int:
    try: cpu = os.cpu_count() or 4
    except Exception: cpu = 4
    return min(4, max(1, cpu // 2))

def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _set_trainable_last_layers(model: SlideDuplicateModel, n_last: int) -> int:
    # Head sempre treinável
    for p in model.projector.parameters(): p.requires_grad = True
    # Backbone: só layers (12 - n_last) .. 11
    keep_from = 12 - max(1, n_last)
    n_train = sum(p.numel() for p in model.projector.parameters())
    for name, param in model.backbone.named_parameters():
        trainable = False
        for i in range(keep_from, 12):
            if f"encoder.layer.{i}" in name:
                trainable = True
                break
        # norm final / pooler acompanham o fine-tune quando descongelamos fundo
        if n_last >= 4 and ("layernorm" in name.lower() or "norm" in name.lower().split(".")[-1]):
            # mantém layernorm final treinável em descongelamento profundo
            if "encoder.layer" not in name: trainable = True
        param.requires_grad = trainable
        if trainable: n_train += param.numel()
    return n_train


def _build_optimizer(model: SlideDuplicateModel, lr: float, backbone_mult: float,
                     weight_decay: float) -> torch.optim.Optimizer:
    # Decay só em pesos 2D; bias/norm ficam sem decay (padrão ViT/BERT)
    head_decay, head_nodecay = [], []
    back_decay, back_nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_backbone = name.startswith("backbone.")
        is_decay = p.dim() >= 2
        if is_backbone:(back_decay if is_decay else back_nodecay).append(p)
        else: (head_decay if is_decay else head_nodecay).append(p)
    groups = []
    if head_decay: groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
    if head_nodecay: groups.append({"params": head_nodecay, "lr": lr, "weight_decay": 0.0})
    if back_decay: groups.append({"params": back_decay, "lr": lr * backbone_mult, "weight_decay": weight_decay})
    if back_nodecay: groups.append({"params": back_nodecay, "lr": lr * backbone_mult, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)

def train_slide_model(
    data_path: str,
    output_checkpoint: str = "slide_duplicate_model.pt",
    epochs: int = 5,
    batch_size: int = 24,
    lr: float = 3e-4,
    temperature: float = 0.1,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    use_amp: Optional[bool] = None,
    val_split: float = 0.05,
    seed: int = 42,
    accum_steps: int = 2,
    weight_decay: float = 0.05,
    backbone_lr_mult: float = 0.15,
    warmup_epochs: int = 2,
    unfreeze_at: Optional[int] = None,
    unfreeze_layers: int = 4,
    resume: bool = True,
):
    _set_seed(seed)
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
    if not all_files: raise ValueError(f"Nenhuma imagem válida encontrada em: {data_path}")

    # Amostragem aleatória (seedada) em vez de [:N] sobre lista ordenada —
    # [:N] enviesava o treino para as primeiras pastas/rolos do disco.
    import random as _py_random
    _rng = _py_random.Random(seed)
    _rng.shuffle(all_files)
    if max_samples and len(all_files) > max_samples:
        print(f"   Limitando a {max_samples} imagens para este ciclo de treinamento (total disponível: {len(all_files)}).")
        all_files = all_files[:max_samples]
    else:
        print(f"   Total de imagens a serem utilizadas: {len(all_files)}")

    # Split treino/validação estratificado por posição embaralhada.
    # Val serve para early signal + salvar o MELHOR checkpoint (antes salvava
    # sempre a última época, mesmo que pior).
    n_total = len(all_files)
    n_val = max(1, int(round(n_total * val_split))) if n_total >= 20 and val_split > 0 else 0
    if n_val > 0 and n_total - n_val < 8:
        n_val = max(0, n_total - 8)
    train_files = all_files[:n_total - n_val] if n_val else list(all_files)
    val_files = all_files[n_total - n_val:] if n_val else []
    print(f"   Split: {len(train_files)} treino / {len(val_files)} validação (seed={seed}).")

    # Build datasets and dataloaders
    from dataset import SlideDuplicateDataset as _SDD
    train_ds = _SDD(train_files, target_size=224, cache_film_crops=True)
    val_ds = _SDD(val_files, target_size=224, cache_film_crops=True) if val_files else None

    # Pre-warm dos caches de treino e val ANTES do loop (thread pool, sem
    # custo de spawn do Windows). Depois disso __getitem__ é só augment barato.
    prewarm_workers = min(8, os.cpu_count() or 4)
    t_prewarm = time.time()
    n_decoded = train_ds.prewarm_cache(max_workers=prewarm_workers)
    n_val_dec = val_ds.prewarm_cache(max_workers=prewarm_workers) if val_ds is not None else 0
    print(f"   Pré-aquecendo cache de {n_decoded} crops treino + {n_val_dec} val "
          f"em {time.time() - t_prewarm:.1f}s ({prewarm_workers} threads).")

    # O prewarm remove arquivos ilegíveis do dataset; se sobrar pouco (ou o
    # val zerar), ajusta aqui antes de criar os loaders.
    if len(train_ds) == 0:
        raise ValueError("Nenhuma imagem legível após filtrar arquivos corrompidos.")
    if val_ds is not None and len(val_ds) == 0:
        print("   [aviso] validação ficou sem imagens legíveis — seguindo sem val (best = train loss).")
        val_ds = None
    elif val_ds is not None and len(val_ds) != n_val_dec:
        print(f"   [dataset] val ajustada: {len(val_ds)} imagens legíveis.")

    # Batch físico limitado pela VRAM (RTX 2060 6GB); batch EFETIVO do
    # NT-Xent cresce via acumulação: eff = batch_size * accum_steps.
    # Ex.: 24 x 2 = 48 pares -> matriz 96x96, bem mais sinal que 8 (16x16).
    accum_steps = max(1, int(accum_steps))
    eff_batch = batch_size * accum_steps
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        num_workers=num_workers,
    )
    if num_workers > 0:
        # Overlap CPU data preparation with GPU compute.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    if device == "cuda":
        # Speed up host->device transfers.
        loader_kwargs["pin_memory"] = True

    dataloader = DataLoader(train_ds, **loader_kwargs)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=min(batch_size, len(val_ds)),
            shuffle=False, drop_last=False, num_workers=0,
            pin_memory=(device == "cuda"),
        )
    print(f"   DataLoader: num_workers={num_workers}, pin_memory={loader_kwargs.get('pin_memory', False)}, AMP={'on' if use_amp else 'off'}")
    print(f"   Batch físico={batch_size} x accum={accum_steps} -> batch efetivo={eff_batch} (temp={temperature}, wd={weight_decay})")

    # Instantiate model and loss
    model = SlideDuplicateModel(model_name="facebook/dinov2-small", projection_dim=256)
    model.to(device)

    # channels_last é para backbone conv; em ViT o ganho é ~0 e pode
    # atrapalhar layout — removido para estabilidade/velocidade real.
    # (mantido DataLoader + AMP fp16, que são os ganhos verdadeiros aqui)

    # Fine-tune progressivo: começa com head + 2 últimas camadas (estável) e
    # aprofunda para `unfreeze_layers` na época `unfreeze_at` (default: 60%
    # do treino). Antes ficava preso para sempre em 2 camadas.
    start_layers = 2
    n_trainable = _set_trainable_last_layers(model, start_layers)
    print(f"   Fine-tune inicial: head + últimas {start_layers} camadas ({n_trainable/1e6:.2f}M params treináveis).")
    if unfreeze_at is None:
        unfreeze_at = max(2, int(round(epochs * 0.6)))
    print(f"   Aprofundamento p/ {unfreeze_layers} camadas na época {unfreeze_at} (se ainda treinando).")

    criterion = NTXentLoss(temperature=temperature)
    optimizer = _build_optimizer(model, lr, backbone_lr_mult, weight_decay)

    # Scheduler: warmup linear + cosine decay. Antes o LR era constante, o que
    # estagnava o NT-Xent depois das primeiras épocas.
    steps_per_epoch = max(1, len(dataloader) // accum_steps)
    total_opt_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, min(total_opt_steps // 2, steps_per_epoch * max(0, warmup_epochs)))
    import math as _math
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.5 * (1.0 + _math.cos(_math.pi * prog))
    # LambdaLR com 1 lambda global aplicado proporcionalmente a cada grupo
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: _lr_lambda(s))
    for g, b in zip(optimizer.param_groups, base_lrs):
        g["initial_lr"] = b

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    ckpt_dir = os.path.dirname(os.path.abspath(output_checkpoint))
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = output_checkpoint
    start_epoch = 1
    best_val = float("inf")
    global_opt_step = 0
    if resume:
        resume_from = best_path if os.path.isfile(best_path) else None
        if resume_from is not None:
            try:
                ckpt = torch.load(resume_from, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=True)
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    bv = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
                    try:
                        best_val = float(bv)
                    except Exception:
                        best_val = float("inf")
                    if ckpt.get("optimizer_state_dict") is not None:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        except Exception:
                            pass
                    if ckpt.get("scheduler_state_dict") is not None:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                            # Blindagem: se checkpoint antigo salvou com Num grupos != lambdas, alinha
                            while len(scheduler.lr_lambdas) < len(scheduler.base_lrs):
                                scheduler.lr_lambdas.append(scheduler.lr_lambdas[0] if len(scheduler.lr_lambdas) else (lambda s: _lr_lambda(s)))
                        except Exception:
                            pass
                    try:
                        global_opt_step = int(ckpt.get("global_opt_step", 0))
                    except Exception:
                        pass
                    print(f"   [resume] continuando de '{resume_from}' (epoca {start_epoch}/{epochs}).")
            except Exception as e:
                print(f"   [resume] falha: {e} - comecando do zero.")

    print(f"   Parametros treinaveis ok.")
    print(f"   Epocas: {start_epoch}..{epochs} | Batch efetivo: {eff_batch} | LR head: {lr}\n")

    def _run_val():
        if val_loader is None:
            return None, None
        model.eval()
        tot_l, tot_a, n = 0.0, 0.0, 0
        with torch.no_grad():
            for v1, v2 in val_loader:
                v1 = v1.to(device, non_blocking=True)
                v2 = v2.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                    _z = model(torch.cat([v1, v2], dim=0))
                    _z1, _z2 = _z.chunk(2, dim=0)
                    _loss, _acc = criterion(_z1, _z2)
                tot_l += _loss.item()
                tot_a += _acc.item()
                n += 1
        model.train()
        return (tot_l / max(1, n)), (tot_a / max(1, n))

    for epoch in range(start_epoch, epochs + 1):
        if epoch == unfreeze_at and unfreeze_layers > start_layers:
            before = {id(p) for p in model.parameters() if p.requires_grad}
            n_trainable = _set_trainable_last_layers(model, unfreeze_layers)
            fresh = [p for p in model.parameters() if p.requires_grad and id(p) not in before]
            if fresh:
                dc = [p for p in fresh if p.dim() >= 2]
                nd = [p for p in fresh if p.dim() < 2]
                cur = _lr_lambda(global_opt_step)
                # LambdaLR guarda 1 lambda por grupo (lr_lambdas) + 1 base_lr por
                # grupo. Novos grupos exigem estender AMBAS as listas, senao o
                # get_lr() faz zip(strict) com tamanhos diferentes e quebra.
                def _extend_sched(base_lr):
                    try:
                        scheduler.base_lrs.append(base_lr)
                        ref = scheduler.lr_lambdas[0] if len(scheduler.lr_lambdas) else (lambda s: _lr_lambda(s))
                        scheduler.lr_lambdas.append(ref)
                    except Exception:
                        pass
                if dc:
                    optimizer.add_param_group({"params": dc, "lr": lr * backbone_lr_mult * cur,
                                               "weight_decay": weight_decay, "initial_lr": lr * backbone_lr_mult})
                    _extend_sched(lr * backbone_lr_mult)
                if nd:
                    optimizer.add_param_group({"params": nd, "lr": lr * backbone_lr_mult * cur,
                                               "weight_decay": 0.0, "initial_lr": lr * backbone_lr_mult})
                    _extend_sched(lr * backbone_lr_mult)
                print(f"   [unfreeze] epoca {epoch}: +{len(fresh)} tensores ({n_trainable/1e6:.2f}M params).")
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(dataloader, desc=f"Epoca {epoch}/{epochs}")
        for it, (view1, view2) in enumerate(pbar, start=1):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                batch = torch.cat([view1, view2], dim=0)
                z = model(batch)
                z1, z2 = z.chunk(2, dim=0)
                loss, acc = criterion(z1, z2)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            do_step = (it % accum_steps == 0) or (it == len(dataloader))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_opt_step += 1
                optimizer.zero_grad(set_to_none=True)

            tl = loss.item() * accum_steps
            epoch_loss += tl
            epoch_acc += acc.item()
            step_count += 1
            try:
                cl = optimizer.param_groups[0]["lr"]
            except Exception:
                cl = lr
            pbar.set_postfix({"loss": f"{tl:.4f}", "acc": f"{acc.item() * 100:.1f}%", "lr": f"{cl:.1e}"})

        avg_loss = epoch_loss / max(1, step_count)
        avg_acc = epoch_acc / max(1, step_count)
        val_loss, val_acc = _run_val()
        if val_loss is None:
            print(f"   [OK] Epoca {epoch}/{epochs} - train {avg_loss:.4f}/{avg_acc*100:.1f}% (sem val)")
            score = avg_loss
        else:
            print(f"   [OK] Epoca {epoch}/{epochs} - train {avg_loss:.4f}/{avg_acc*100:.1f}% | val {val_loss:.4f}/{val_acc*100:.1f}%")
            score = val_loss

        is_best = score < best_val
        if is_best:
            best_val = score
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_opt_step": global_opt_step,
            "model_name": "facebook/dinov2-small",
            "projection_dim": 256,
            "temperature": temperature,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "best_val": best_val,
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "eff_batch": eff_batch,
            "lr": lr,
        }
        try:
            payload["scaler_state_dict"] = scaler.state_dict() if scaler.is_enabled() else None
        except Exception:
            pass
        if val_loss is not None:
            payload["val_loss"] = val_loss
            payload["val_accuracy"] = val_acc
        torch.save(payload, best_path) if is_best else None
        if is_best:
            print(f"       * melhor ({score:.4f}) -> {os.path.basename(best_path)}")
        else:
            print(f"       (sem melhora; melhor {best_val:.4f}).")

    print(f"\n[SUCESSO] Melhor score: {best_val:.4f}")
    print(f"   Melhor: {os.path.abspath(best_path)}")
    return best_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento auto-supervisionado contrastivo para slides")
    parser.add_argument("--data", type=str, default=r"E:\backup\digitalização", help="Caminho da pasta com as fotos")
    parser.add_argument("--output", type=str, default="slide_duplicate_model.pt", help="Arquivo de saída do modelo treinado")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch físico (VRAM); efetivo = batch_size x accum")
    parser.add_argument("--accum", type=int, default=2, help="Passos de acumulação (batch efetivo p/ NT-Xent)")
    parser.add_argument("--lr", type=float, default=3e-4, help="LR do head (backbone usa fração)")
    parser.add_argument("--backbone_mult", type=float, default=0.15, help="LR backbone = lr x mult")
    parser.add_argument("--wd", type=float, default=0.05, help="Weight decay (só pesos 2D)")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperatura NT-Xent")
    parser.add_argument("--val_split", type=float, default=0.05, help="Fração p/ validação/best-model")
    parser.add_argument("--seed", type=int, default=42, help="Seed p/ shuffle/split")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Épocas de warmup do scheduler")
    parser.add_argument("--unfreeze_at", type=int, default=None, help="Epoca p/ aprofundar fine-tune (default 60pct)")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Camadas finais após aprofundar")
    parser.add_argument("--max_samples", type=int, default=500, help="Máximo de amostras para treino")
    parser.add_argument("--num_workers", type=int, default=None, help="Workers do DataLoader")
    parser.add_argument("--no_amp", action="store_true", help="Desativa precision mista (AMP) na GPU")
    parser.add_argument("--no_resume", action="store_true", help="Ignora checkpoints existentes")
    args = parser.parse_args()

    train_slide_model(
        data_path=args.data,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temp,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        val_split=args.val_split,
        seed=args.seed,
        accum_steps=args.accum,
        weight_decay=args.wd,
        backbone_lr_mult=args.backbone_mult,
        warmup_epochs=args.warmup_epochs,
        unfreeze_at=args.unfreeze_at,
        unfreeze_layers=args.unfreeze_layers,
        resume=not args.no_resume,
    )
