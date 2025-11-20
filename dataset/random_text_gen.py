import os
import csv
import json
import random
import string
import numpy as np
from argparse import ArgumentParser


# -------------------------------
# Random text generation methods
# -------------------------------

def random_uniform(n):
    """Uniform random characters."""
    chars = string.ascii_letters + string.digits + string.punctuation + " "
    return ''.join(random.choice(chars) for _ in range(n))


def random_weighted(n, chars, probs):
    """Random chars with the same character distribution as real text."""
    return ''.join(np.random.choice(chars, size=n, p=probs))


def permuted(text):
    """Random permutation of a real text line."""
    t = list(text)
    random.shuffle(t)
    return ''.join(t)


def random_markov_like(n):
    """Produce pseudo-word sequences that look a bit structured but are gibberish."""
    consonants = "bcdfghjklmnpqrstvwxyz "
    vowels = "aeiou "
    s = ""
    for _ in range(n):
        pattern = random.choice([
            random.choice(consonants),
            random.choice(vowels),
            random.choice(consonants) + random.choice(vowels),
            random.choice(vowels) + random.choice(consonants)
        ])
        s += pattern
    return s[:n]


# -------------------------------
# Dataset Generator
# -------------------------------

def generate_dataset(
    real_lines,
    total_samples=50000,
    split=0.1,
    max_len=256,
    format="csv"
):
    random_samples = []
    real_samples = []

    # Flatten real corpus for weighted random generation
    full_corpus_str = "\n".join(real_lines)

    chars, counts = np.unique(list(full_corpus_str), return_counts=True)
    probs = counts / counts.sum()

    for _ in range(total_samples // 2):  # 50% real, 50% random
        # --- REAL SAMPLE ---
        text = random.choice(real_lines).strip()
        if len(text) > max_len:
            r = random.randint(0, len(text) - max_len)
            text = text[r:r + max_len]
        label = 1
        real_samples.append((text, label))

        # --- RANDOM SAMPLE ---
        random_type = random.choice(["uniform", "weighted", "permuted", "markov"])

        if random_type == "uniform":
            rand_text = random_uniform(max_len)

        elif random_type == "weighted":
            rand_text = random_weighted(max_len, chars, probs)

        elif random_type == "permuted":
            base = random.choice(real_lines).strip()
            base = base[:max_len]
            rand_text = permuted(base)

        elif random_type == "markov":
            rand_text = random_markov_like(max_len)

        random_samples.append((rand_text, 0))

        if _ % 500 == 0:
            print(f"[INFO] Generated {_ * 2} samples...")

    real_samples_train, real_samples_val = real_samples[:int(len(real_samples)*(1-split))], real_samples[int(len(real_samples)*(1-split)):]
    random_samples_train, random_samples_val = random_samples[:int(len(random_samples)*(1-split))], random_samples[int(len(random_samples)*(1-split)):]
    train_dataset = real_samples_train + random_samples_train
    val_dataset = real_samples_val + random_samples_val

    # -------------------------------
    # Save dataset
    # -------------------------------
    train_path = "train_dataset_128"
    val_path = "val_dataset_128"
    if format == "csv":
        with open(train_path+".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text", "label"])
            for text, label in train_dataset:
                writer.writerow([text, label])
        with open(val_path+".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text", "label"])
            for text, label in val_dataset:
                writer.writerow([text, label])
        print(f"[OK] CSV dataset saved")

    elif format == "jsonl":
        with open(train_path+".json", "w", encoding="utf-8") as f:
            for text, label in train_dataset:
                f.write(json.dumps({"text": text, "label": label}) + "\n")
        with open(train_path+".json", "w", encoding="utf-8") as f:
            for text, label in val_dataset:
                f.write(json.dumps({"text": text, "label": label}) + "\n")
        print(f"[OK] JSONL dataset saved")

    else:
        raise ValueError("Format must be 'csv' or 'jsonl'.")


# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to real corpus .txt file")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--split", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--format", type=str, choices=["csv", "jsonl"], default="csv")

    args = parser.parse_args()

    # Load real corpus lines
    with open(args.corpus, "r", encoding="utf-8") as f:
        real_lines = [line.strip() for line in f if len(line.strip()) > 3]

    generate_dataset(
        real_lines=real_lines,
        total_samples=args.samples,
        split=args.split,
        max_len=args.max_len,
        format=args.format
    )
