"""
Benchmark rapido de forward+backward para diferentes batch_size.
Usa o mesmo modelo/congelamento do train.py e mede tempo por step na CPU.
"""
import time
import torch
from model import SlideDuplicateModel, NTXentLoss

device = "cpu"
model = SlideDuplicateModel(model_name="facebook/dinov2-small", projection_dim=256)
model.to(device)

# Replica o congelamento do train.py (treina so ultimos 2 blocos + projector)
for name, param in model.backbone.named_parameters():
    if "encoder.layer.11" not in name and "encoder.layer.10" not in name:
        param.requires_grad = False

criterion = NTXentLoss(temperature=0.07)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=1e-4
)
model.train()

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Parametros treinaveis: {n_train:,} / totais: {n_total:,} ({100*n_train/n_total:.1f}%)\n")

print(f"{'batch':>6} | {'1o step (warmup)':>18} | {'media 3 steps':>14}")
print("-" * 46)

results = {}
for bs in [4, 8, 16, 32, 48, 64]:
    x1 = torch.randn(bs, 3, 224, 224)
    x2 = torch.randn(bs, 3, 224, 224)
    try:
        # warmup (1 step nao cronometrado)
        t0 = time.time()
        optimizer.zero_grad()
        z1 = model(x1); z2 = model(x2)
        loss = criterion(z1, z2)
        loss.backward(); optimizer.step()
        warm = time.time() - t0

        # media de 3 steps
        times = []
        for _ in range(3):
            t0 = time.time()
            optimizer.zero_grad()
            z1 = model(x1); z2 = model(x2)
            loss = criterion(z1, z2)
            loss.backward(); optimizer.step()
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        results[bs] = (warm, avg)
        print(f"{bs:>6} | {warm:>15.2f}s | {avg:>11.2f}s")
        del x1, x2
    except RuntimeError as e:
        print(f"{bs:>6} | ERRO: {str(e)[:40]}")
        break

print("\n--- Estimativa por epoca (steps/step) para amostras conhecidas ---")
print(f"{'samples':>8} | " + " | ".join(f"bs={b}" for b in results))
for samples in [100, 150, 300, 600, 1000]:
    row = [f"{samples:>8} |"]
    for b, (_, avg) in results.items():
        steps = max(1, samples // b)
        row.append(f"{steps*avg/60:>6.1f}min")
    print(" | ".join(row))
