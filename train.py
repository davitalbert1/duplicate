import os
import sys
import glob
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
    if os.path.isfile(directory_or_file):
        return [directory_or_file]
    
    files = []
    for root, _, filenames in os.walk(directory_or_file):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                files.append(os.path.join(root, f))
    return sorted(files)


def train_slide_model(
    data_path: str,
    output_checkpoint: str = "slide_duplicate_model.pt",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    temperature: float = 0.07,
    max_samples: Optional[int] = None,
    device: Optional[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=0
    )

    # Instantiate model and loss
    model = SlideDuplicateModel(model_name="facebook/dinov2-small", projection_dim=256)
    model.to(device)

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

    model.train()
    print(f"   Parâmetros treináveis configurados.")
    print(f"   Épocas: {epochs} | Tamanho do Lote: {batch_size} | Temp: {temperature}\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_count = 0
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{epochs}")
        for view1, view2 in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, step_count)
        print(f"   [OK] Fim da Epoca {epoch}/{epochs} - Loss Medio: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
    save_data = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "model_name": "facebook/dinov2-small",
        "projection_dim": 256,
        "temperature": temperature
    }
    torch.save(save_data, output_checkpoint)
    print(f"\n[SUCESSO] Treinamento concluido!")
    print(f"   Pesos salvos em: {os.path.abspath(output_checkpoint)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento auto-supervisionado contrastivo para slides")
    parser.add_argument("--data", type=str, default=r"E:\backup\digitalização", help="Caminho da pasta com as fotos")
    parser.add_argument("--output", type=str, default="slide_duplicate_model.pt", help="Arquivo de saída do modelo treinado")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=150, help="Máximo de amostras para treino")
    args = parser.parse_args()
    
    train_slide_model(
        data_path=args.data,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples
    )
