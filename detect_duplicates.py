import os
import time
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from model import SlideDuplicateModel
from slide_detector import load_film_crop_fast

class SlideInferenceEngine:
    def __init__(self, checkpoint_path: Optional[str] = "slide_duplicate_model.pt", device: Optional[str] = None):
        if device is None: self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = device

        self.model = SlideDuplicateModel(model_name="facebook/dinov2-small", projection_dim=256)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[+] Carregando modelo treinado de: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint: self.model.load_state_dict(checkpoint["model_state_dict"])
            else: self.model.load_state_dict(checkpoint)
        else:
            if checkpoint_path:
                print(f"[!] Checkpoint '{checkpoint_path}' nao encontrado. Usando pesos base DINOv2.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        film_crop = load_film_crop_fast(img_path, target_size=(224, 224))
        tensor = self.transform(film_crop).unsqueeze(0)
        return tensor

    def compute_embedding(self, img_path: str) -> np.ndarray:
        tensor = self.preprocess_image(img_path).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
            return emb.squeeze(0).cpu().numpy()

    def compute_embeddings_batch(self, img_paths: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Extraindo embeddings"):
            batch_files = img_paths[i:i + batch_size]
            tensors = [self.preprocess_image(p) for p in batch_files]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            with torch.no_grad():
                embs = self.model(batch_tensor).cpu().numpy()
                embeddings.append(embs)
        return np.vstack(embeddings)

    @staticmethod
    def similarity_to_probability(sim: float, sharp_factor: float = 12.0, threshold: float = 0.72) -> float:
        # Sigmoid transfer function
        prob = 1.0 / (1.0 + np.exp(-sharp_factor * (sim - threshold)))
        return float(prob)

    def analyze_single_file_against_collection(
        self,
        query_image_path: str,
        collection_path: str,
        top_k: int = 20,
        min_probability: float = 0.10,
        max_images: Optional[int] = None,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        all_files = self._get_image_list(collection_path, recursive=recursive)
        # Avoid comparing image with itself if it's inside the collection
        norm_query = os.path.abspath(query_image_path)
        target_files = [f for f in all_files if os.path.abspath(f) != norm_query]

        if max_images and len(target_files) > max_images: target_files = target_files[:max_images]

        if not target_files: return []

        print(f"[*] Processando imagem de consulta: {query_image_path}")
        q_emb = self.compute_embedding(query_image_path)

        print(f"[*] Extraindo embeddings de {len(target_files)} imagens da colecao...")
        target_embs = self.compute_embeddings_batch(target_files)

        # Cosine similarities (vectors are unit normalized)
        sims = target_embs @ q_emb

        results = []
        for path, sim in zip(target_files, sims):
            prob = self.similarity_to_probability(sim)
            if prob >= min_probability:
                results.append({
                    "query": query_image_path,
                    "candidate": path,
                    "similarity": float(sim),
                    "probability": prob,
                    "percentage_str": f"{prob * 100.0:.2f}%"
                })

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results[:top_k]

    def analyze_folder_for_duplicates(
        self,
        folder_path: str,
        max_images: Optional[int] = None,
        min_probability: float = 0.50,
        top_k: int = 50,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        files = self._get_image_list(folder_path, recursive=recursive)
        if max_images and len(files) > max_images: files = files[:max_images]

        n = len(files)
        if n < 2:
            print("[!] Menos de 2 imagens encontradas para comparacao.")
            return []

        print(f"[*] Analisando pasta: {folder_path}")
        print(f"   Total de imagens: {n}")
        print(f"   Calculando embeddings neurais...")

        start_time = time.time()
        embs = self.compute_embeddings_batch(files)

        # Pairwise cosine similarity matrix
        sim_matrix = embs @ embs.T

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                prob = self.similarity_to_probability(sim)
                if prob >= min_probability:
                    pairs.append({
                        "file_a": files[i],
                        "file_b": files[j],
                        "similarity": sim,
                        "probability": prob,
                        "percentage_str": f"{prob * 100.0:.2f}%"
                    })

        pairs.sort(key=lambda x: x["probability"], reverse=True)
        elapsed = time.time() - start_time
        print(f"   Concluido em {elapsed:.2f}s! Total de pares suspeitos: {len(pairs)}")
        return pairs[:top_k]

    def _get_image_list(self, path: str, recursive: bool = False) -> List[str]:
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        if os.path.isfile(path): return [path]
        files = []
        if recursive:
            for root, _, filenames in os.walk(path):
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(path):
                full_p = os.path.join(path, f)
                if os.path.isfile(full_p) and os.path.splitext(f)[1].lower() in valid_exts:
                    files.append(full_p)
        return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Detector de Fotos Duplicadas em Slides de Filmes")
    parser.add_argument("path", type=str, help="Caminho da pasta ou arquivo para analisar")
    parser.add_argument("--against", type=str, default=None, help="Pasta de referência (se o path principal for um arquivo)")
    parser.add_argument("--model", type=str, default="slide_duplicate_model.pt", help="Caminho do modelo treinado (.pt)")
    parser.add_argument("--min_prob", type=float, default=0.50, help="Probabilidade mínima (0.0 a 1.0) para exibir")
    parser.add_argument("--max_images", type=int, default=None, help="Máximo de imagens para analisar na pasta")
    parser.add_argument("--top_k", type=int, default=30, help="Número máximo de pares retornados")
    parser.add_argument("--recursive", action="store_true", help="Buscar recursivamente em subpastas")
    args = parser.parse_args()

    engine = SlideInferenceEngine(checkpoint_path=args.model)

    if os.path.isfile(args.path):
        ref_folder = args.against if args.against else os.path.dirname(args.path)
        print(f"\n=======================================================")
        print(f"[*] ANALISANDO ARQUIVO UNICO CONTRA A COLECAO")
        print(f"Arquivo: {args.path}")
        print(f"Pasta Alvo: {ref_folder}")
        print(f"=======================================================\n")

        matches = engine.analyze_single_file_against_collection(
            query_image_path=args.path,
            collection_path=ref_folder,
            top_k=args.top_k,
            min_probability=args.min_prob,
            max_images=args.max_images,
            recursive=args.recursive
        )

        if not matches:
            print("Nenhuma duplicata provavel encontrada para este arquivo acima do limiar.")
        else:
            print(f"Rank | Probabilidade | Similaridade | Imagem Candidata")
            print("-" * 75)
            for rank, item in enumerate(matches, 1):
                cand = os.path.basename(item["candidate"])
                print(f"#{rank:<3} | {item['percentage_str']:<13} | {item['similarity']:.4f}       | {cand}")

    elif os.path.isdir(args.path):
        print(f"\n=======================================================")
        print(f"[*] ANALISANDO PASTA COMPLETA EM BUSCA DE DUPLICATAS")
        print(f"Pasta: {args.path}")
        print(f"=======================================================\n")

        pairs = engine.analyze_folder_for_duplicates(
            folder_path=args.path,
            max_images=args.max_images,
            min_probability=args.min_prob,
            top_k=args.top_k,
            recursive=args.recursive
        )

        if not pairs:
            print("Nenhum par duplicado encontrado acima do limiar especificado.")
        else:
            print(f"Rank | Probabilidade | Similaridade | Foto A <-> Foto B")
            print("-" * 85)
            for rank, item in enumerate(pairs, 1):
                fa = os.path.basename(item["file_a"])
                fb = os.path.basename(item["file_b"])
                print(f"#{rank:<3} | {item['percentage_str']:<13} | {item['similarity']:.4f}       | {fa:<20} <-> {fb:<20}")
    else:
        print(f"[!] Caminho invalido: {args.path}")

if __name__ == "__main__":
    main()
