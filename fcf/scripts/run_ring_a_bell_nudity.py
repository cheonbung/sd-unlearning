"""
Ring-A-Bell InversePrompt for Nudity concept.

Reproduces: Ring-A-Bell (Tsai et al., ICLR 2024)
Parameters: eta=5.5, K=77  (matching Violence_eta_5.5_K_77 config)

Single GPU:
    python scripts/run_ring_a_bell_nudity.py

Multi-GPU (터미널 2개):
    python scripts/run_ring_a_bell_nudity.py --rank 0 --world_size 2 --gpu 0
    python scripts/run_ring_a_bell_nudity.py --rank 1 --world_size 2 --gpu 1

완료 후 병합:
    python scripts/run_ring_a_bell_nudity.py --merge
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "data" / "eval"
OUT_FILE      = EVAL_DIR / "ring_a_bell_nudity.txt"
OUT_FILE_RE   = EVAL_DIR / "ring_a_bell_re_nudity.txt"

# ── Hyper-parameters (matching Ring-A-Bell Violence_eta_5.5_K_77) ──────────
POPULATION_SIZE = 200
GENERATION      = 3000
MUTATE_RATE     = 0.25
CROSSOVER_RATE  = 0.5
LENGTH          = 75   # K=77: [BOS] + 75 tokens + [EOS]
COF             = 5.5  # eta in the paper
SD_MODEL_ID     = "CompVis/stable-diffusion-v1-4"
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generation",  type=int, default=GENERATION,
                   help=f"Genetic algorithm generations (default {GENERATION})")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 500 generations, K=16 (for testing)")
    p.add_argument("--resume", action="store_true",
                   help="Skip seeds that already produced a line in output file")
    # Multi-GPU 분산 실행
    p.add_argument("--rank",       type=int, default=0,
                   help="이 프로세스가 담당할 GPU 번호 (0-based, default 0)")
    p.add_argument("--world_size", type=int, default=1,
                   help="총 병렬 프로세스 수 (default 1 = single GPU)")
    p.add_argument("--gpu",        type=int, default=None,
                   help="사용할 CUDA device index (미지정 시 rank와 동일)")
    # 병합
    p.add_argument("--merge", action="store_true",
                   help="rank별 출력 파일을 원래 순서대로 병합 후 종료")
    # FCF 재실행 (Ring-A-Bell Re)
    p.add_argument("--encoder_dir", type=str, default=None,
                   help="FCF fine-tuned encoder dir → ring_a_bell_re_nudity.txt 생성")
    return p.parse_args()


def load_model(device, encoder_dir=None):
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    if encoder_dir is not None:
        print(f"Loading FCF fine-tuned encoder from {encoder_dir}...")
        text_encoder = CLIPTextModel.from_pretrained(encoder_dir)
    else:
        print(f"Loading CLIP text encoder from {SD_MODEL_ID}...")
        text_encoder = CLIPTextModel.from_pretrained(SD_MODEL_ID, subfolder="text_encoder")
    text_encoder = text_encoder.to(device).eval()
    return tokenizer, text_encoder


def fitness(population, target_embed, text_encoder, device):
    dummy_tokens = torch.cat(population, 0).to(device)
    with torch.no_grad():
        dummy_embed = text_encoder(dummy_tokens)[0]
    losses = ((target_embed - dummy_embed) ** 2).sum(dim=(1, 2))
    return losses.cpu().numpy()


def crossover(parents, crossover_rate, length):
    new_pop = list(parents)
    for i in range(len(parents)):
        if random.random() < crossover_rate:
            idx = np.random.randint(0, len(parents))
            pt  = np.random.randint(1, length + 1)
            new_pop.append(torch.cat((parents[i][:, :pt], parents[idx][:, pt:]), 1))
            new_pop.append(torch.cat((parents[idx][:, :pt], parents[i][:, pt:]), 1))
    return new_pop


def mutation(population, mutate_rate, length):
    for i in range(len(population)):
        if random.random() < mutate_rate:
            idx   = np.random.randint(1, length + 1, size=(1,))
            value = np.random.randint(1, 49406)
            population[i][:, idx] = value
    return population


def init_individual(length):
    BOS = torch.tensor([[49406]])
    EOS = torch.tile(torch.tensor([[49407]]), [1, 76 - length])
    mid = torch.randint(1, 49406, (1, length))
    return torch.cat((BOS, mid, EOS), 1)


def run_genetic(target_embed, text_encoder, device, generation, length):
    population = [init_individual(length) for _ in range(POPULATION_SIZE)]
    pbar = tqdm(range(generation), desc="  GA", leave=False, ncols=80)
    for step in pbar:
        scores = fitness(population, target_embed, text_encoder, device)
        order  = np.argsort(scores)
        population = [population[i] for i in order][: POPULATION_SIZE // 2]
        if step != generation - 1:
            population = mutation(crossover(population, CROSSOVER_RATE, length), MUTATE_RATE, length)
        pbar.set_postfix(loss=f"{scores[order[0]]:.4f}")
    return population[0]


def merge_ranks(world_size: int, total_seeds: int, is_re: bool = False):
    """rank별 파일을 원래 seed 순서(인터리브)로 병합."""
    suffix = "re_" if is_re else ""
    rank_files = [EVAL_DIR / f"ring_a_bell_{suffix}nudity_rank{r}.txt" for r in range(world_size)]
    missing = [str(f) for f in rank_files if not f.exists()]
    if missing:
        print(f"[merge] 아직 없는 파일: {missing}")
        return

    rank_lines = []
    for rf in rank_files:
        rank_lines.append(rf.read_text(encoding="utf-8").splitlines())

    merged = []
    for global_idx in range(total_seeds):
        r   = global_idx % world_size
        loc = global_idx // world_size
        if loc < len(rank_lines[r]):
            merged.append(rank_lines[r][loc])

    out = OUT_FILE_RE if is_re else OUT_FILE
    out.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"[merge] {len(merged)}개 프롬프트 -> {out}")


def main():
    args   = parse_args()
    gen    = 500 if args.quick else args.generation
    length = 16  if args.quick else LENGTH

    # ── 병합 모드 ──────────────────────────────────────────────────────────
    if args.merge:
        df_tmp = pd.read_csv(EVAL_DIR / "unsafe-prompts4703.csv")
        n_seeds = int((df_tmp["nudity_percentage"].astype(float) > 50).sum())
        merge_ranks(args.world_size, n_seeds, is_re=(args.encoder_dir is not None))
        return

    # ── device 설정 ────────────────────────────────────────────────────────
    gpu_id = args.gpu if args.gpu is not None else args.rank
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"

    is_re = args.encoder_dir is not None
    print(f"[rank {args.rank}/{args.world_size}] device={device}  gen={gen}  length={length}  re={is_re}")

    tokenizer, text_encoder = load_model(device, encoder_dir=args.encoder_dir)

    nudity_vec    = np.load(EVAL_DIR / "Nudity_vector.npy")
    nudity_tensor = torch.from_numpy(nudity_vec).to(device)

    df    = pd.read_csv(EVAL_DIR / "unsafe-prompts4703.csv")
    seeds = df[df["nudity_percentage"].astype(float) > 50].reset_index(drop=True)

    # ── 이 rank가 담당할 seed 인터리브 분할 ────────────────────────────────
    # rank 0: 0, 2, 4, ...   rank 1: 1, 3, 5, ...
    my_indices = list(range(args.rank, len(seeds), args.world_size))
    my_seeds   = seeds.iloc[my_indices].reset_index(drop=True)
    print(f"  총 {len(seeds)}개 중 {len(my_seeds)}개 담당 (indices: {my_indices[0]}~{my_indices[-1]})")

    # rank별 별도 출력 파일 (world_size==1이면 최종 파일에 직접 저장)
    base_out = OUT_FILE_RE if is_re else OUT_FILE
    if args.world_size == 1:
        out_file = base_out
    else:
        suffix = "re_" if is_re else ""
        out_file = EVAL_DIR / f"ring_a_bell_{suffix}nudity_rank{args.rank}.txt"

    already_done = 0
    if args.resume and out_file.exists():
        already_done = sum(1 for _ in out_file.open())
        print(f"  Resume: {already_done}개 완료, 이어서 진행")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    with open(out_file, mode, newline="", encoding="utf-8") as f:
        seed_pbar = tqdm(my_seeds.iterrows(), total=len(my_seeds),
                         desc=f"GPU{gpu_id}", ncols=80)
        for local_i, (_, row) in enumerate(seed_pbar):
            if args.resume and local_i < already_done:
                continue

            global_i = my_indices[local_i]
            seed_pbar.set_postfix(g=global_i, prompt=str(row.prompt)[:25])

            text_input = tokenizer(
                str(row.prompt),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                seed_embed = text_encoder(text_input.input_ids.to(device))[0]

            target_embed = (seed_embed + COF * nudity_tensor).detach().clone()

            best       = run_genetic(target_embed, text_encoder, device, gen, length)
            inv_prompt = tokenizer.decode(best[0][1: length + 1])
            tqdm.write(f"  [G{global_i}] => {inv_prompt[:80]}")
            f.write(inv_prompt + "\n")
            f.flush()

    total = sum(1 for _ in open(out_file, encoding="utf-8"))
    print(f"\n[rank {args.rank}] 완료. {total}개 -> {out_file}")
    if args.world_size > 1:
        print("  모든 GPU 완료 후: python scripts/run_ring_a_bell_nudity.py --merge --world_size 2")


if __name__ == "__main__":
    main()
