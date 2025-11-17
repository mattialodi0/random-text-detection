import argparse, math, random, collections, sys

# Simple character n-gram language model to detect likely-real vs random text
# Usage:
#   python test.py --text "Some input string"
#   python test.py --train-file path/to/corpus.txt --text "..." --ngram 3


class NGramCharLM:
    def __init__(self, n=3):
        self.n = max(1, n)
        self.counts = collections.defaultdict(collections.Counter)
        self.context_totals = collections.Counter()
        self.vocab = set()

    def _normalize(self, s):
        return s.replace("\r"," ").replace("\n"," ").lower()

    def train(self, corpus):
        s = self._normalize(corpus)
        self.vocab.update(set(s))
        padded = (" " * (self.n - 1)) + s
        for i in range(len(padded) - self.n + 1):
            ctx = padded[i:i + self.n - 1]
            ch = padded[i + self.n - 1]
            self.counts[ctx][ch] += 1
            self.context_totals[ctx] += 1

    def char_prob(self, context, ch):
        # Laplace (add-one) smoothing
        ctx = context[-(self.n - 1):] if self.n > 1 else ""
        V = len(self.vocab) or 1
        count = self.counts.get(ctx, {}).get(ch, 0)
        total = self.context_totals.get(ctx, 0)
        return (count + 1) / (total + V)

    def score(self, text):
        s = self._normalize(text)
        padded = (" " * (self.n - 1)) + s
        logp = 0.0
        n_chars = 0
        for i in range(len(padded) - self.n + 1):
            ctx = padded[i:i + self.n - 1]
            ch = padded[i + self.n - 1]
            p = self.char_prob(ctx, ch)
            logp += math.log(p)
            n_chars += 1
        return logp / max(1, n_chars)  # average log-prob per char

def generate_random_string(length, chars):
    return "".join(random.choice(list(chars)) for _ in range(length))

def is_likely_real(text, model, samples=200, alpha=2.0):
    score = model.score(text)
    length = max(1, len(text))
    chars = model.vocab or set("abcdefghijklmnopqrstuvwxyz ")
    # sample random strings from same character set
    random_scores = [model.score(generate_random_string(length, chars)) for _ in range(samples)]
    mean = sum(random_scores) / samples
    var = sum((x - mean) ** 2 for x in random_scores) / samples
    std = math.sqrt(var)
    threshold = mean + alpha * std
    is_real = score > threshold
    return {
        "score": score,
        "random_mean": mean,
        "random_std": std,
        "threshold": threshold,
        "is_likely_real": is_real
    }

def default_corpus():
    return (
        "In a village of La Mancha, the name of which I have no desire to call to mind, "
        "there lived not long since one of those gentlemen that keep a lance in the lance-rack, "
        "an old buckler, a lean hack, and a greyhound for coursing."
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", help="path to training text (plain .txt)")
    p.add_argument("--text", help="input text to evaluate; if omitted, read stdin")
    p.add_argument("--ngram", type=int, default=3)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--alpha", type=float, default=2.0, help="sensitivity for decision")
    args = p.parse_args()

    train_text = ""
    if args.train_file:
        try:
            with open(args.train_file, "r", encoding="utf-8") as f:
                train_text = f.read()
        except Exception as e:
            print("Could not read train file:", e, file=sys.stderr)
            sys.exit(1)
    else:
        train_text = default_corpus()

    if args.text:
        text = args.text
    else:
        print("Paste or type the text to evaluate, then EOF (Ctrl+D / Ctrl+Z):")
        text = sys.stdin.read()

    model = NGramCharLM(n=args.ngram)
    model.train(train_text)
    res = is_likely_real(text, model, samples=args.samples, alpha=args.alpha)

    print("avg_log_prob_per_char: {:.4f}".format(res["score"]))
    print("random_mean: {:.4f}, random_std: {:.4f}, threshold: {:.4f}".format(
        res["random_mean"], res["random_std"], res["threshold"]))
    print("Result: {}".format("LIKELY REAL" if res["is_likely_real"] else "LIKELY RANDOM"))

if __name__ == "__main__":
    main()