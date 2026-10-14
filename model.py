import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional

def _load_dinov2_backbone(model_name: str, token: Optional[str] = None):
    # huggingface_hub ja le HF_TOKEN do ambiente automaticamente, mas resolvemos
    # explicitamente para repassar ao download de fallback de forma previsivel.
    if token is None: token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        # Modo offline: usa apenas arquivos ja baixados -> nenhuma requisicao ao Hub.
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError):
        # Cache ausente/corrompido: baixa do Hub (usa o token se disponivel).
        print(f"[i] Modelo '{model_name}' nao encontrado no cache local. Baixando do Hugging Face Hub...")
        return AutoModel.from_pretrained(model_name, token=token)

class SlideDuplicateModel(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-small", projection_dim: int = 256, freeze_backbone: bool = False):
        super().__init__()
        self.model_name = model_name
        self.backbone = _load_dinov2_backbone(model_name)

        hidden_dim = self.backbone.config.hidden_size # 384 for dinov2-small

        if freeze_backbone:
            for param in self.backbone.parameters(): param.requires_grad = False

        # Non-linear projection head (SimCLR/DINO style)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # cls token represents global semantic and visual composition
        cls_token = outputs.last_hidden_state[:, 0, :]
        projected = self.projector(cls_token)
        # Normalize to unit sphere for cosine similarity and probability computation
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def get_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values)

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0) # (2B, D)

        # Compute the similarity matrix in float32 for numerical stability.
        # We disable autocast here because matmul/softmax would otherwise be cast
        # back to fp16, where the large negative mask value (-1e9) overflows fp16.
        # The matrix is tiny (2B x 2B), so the fp32 cost is negligible.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            sim_matrix = torch.matmul(z, z.T) / self.temperature # (2B, 2B)

            # Mask out self-similarity (diagonal)
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)

            # Targets: pos pair for index i is (i + batch_size) and for (i + batch_size) is i
            targets = torch.arange(batch_size, device=z.device)
            targets = torch.cat([targets + batch_size, targets], dim=0)

            loss = F.cross_entropy(sim_matrix, targets)

            # Top-1 contrastive accuracy: did the argmax pick the true positive?
            # The diagonal is already masked to -1e9, so argmax never picks "self".
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == targets).float().mean()

        return loss, accuracy
