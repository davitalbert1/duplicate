"""
Perfil real do loop de treino: mede tempo de dados vs GPU por iteracao,
replicando exatamente o pipeline de train.py (num_workers=0, cache ligado).
"""
import time, glob, torch
from torch.utils.data import DataLoader
from model import SlideDuplicateModel, NTXentLoss
from dataset import SlideDuplicateDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted(glob.glob(r"E:\backup\digitalização\pendrive\*.JPG"))[:16]
print("device", dev, "files", len(files))

ds = SlideDuplicateDataset(files, target_size=224, cache_film_crops=True)
dl = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, num_workers=0)

m = SlideDuplicateModel()
m.to(dev)
for n, p in m.backbone.named_parameters():
    p.requires_grad = ("encoder.layer.11" in n or "encoder.layer.10" in n)
crit = NTXentLoss()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
m.train()

it = iter(dl)
# warmup cache (1a iteracao inclui decode)
for step in range(3):
    t0 = time.time()
    v1, v2 = next(it)
    t_data = time.time() - t0
    v1 = v1.to(dev, non_blocking=True); v2 = v2.to(dev, non_blocking=True)
    t1 = time.time()
    opt.zero_grad()
    z1 = m(v1); z2 = m(v2)
    loss = crit(z1, z2); loss.backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.time() - t1
    print(f"step {step}: data={t_data*1000:8.1f}ms  gpu={t_gpu*1000:8.1f}ms  total={(t_data+t_gpu)*1000:8.1f}ms")
