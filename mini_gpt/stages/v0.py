"""نسخهٔ صفر: شمارش جفت نویسه‌ها، بدون PyTorch و بدون شبکهٔ عصبی."""

from collections import Counter, defaultdict
from pathlib import Path
import random

def transition_counts(ids):
    counts = defaultdict(Counter)
    for current, following in zip(ids, ids[1:]):
        counts[current][following] += 1
    return counts


def generate(counts, prompt, length, vocabulary_size, seed=42):
    if type(length) is not int or length < 0 or type(vocabulary_size) is not int or vocabulary_size < 1:
        raise ValueError("length must be nonnegative and vocabulary_size positive integers")
    rng = random.Random(seed)
    result = list(prompt)
    if not result:
        raise ValueError("provide a nonempty prompt")
    for _ in range(length):
        row = counts.get(result[-1], Counter())
        candidates = list(range(vocabulary_size))
        weights = [row[i] + 1 for i in candidates]
        result.append(rng.choices(candidates, weights=weights, k=1)[0])
    return result


def main():
    text = Path("data/sample.txt").read_text(encoding="utf-8")
    train_text = text[:int(0.9 * len(text))]
    # همان نگاشت سادهٔ جلسهٔ ۲؛ قرارداد نشانهٔ ناشناخته را در بخش چهار اضافه می‌کنیم.
    vocabulary = sorted(set(train_text))
    to_id = {char: index for index, char in enumerate(vocabulary)}
    ids = [to_id[char] for char in train_text]
    counts = transition_counts(ids)
    prompt = [to_id[char] for char in "مدل "]
    output = generate(counts, prompt, 80, len(vocabulary))
    print("".join(vocabulary[index] for index in output))


if __name__ == "__main__":
    from ..console import configure_console
    configure_console()
    main()
