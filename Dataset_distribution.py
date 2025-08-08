from datasets import load_dataset
from transformers import GPT2TokenizerFast
from collections import Counter
from itertools import islice
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def batch_iterator(dataset, batch_size):
    iterator = iter(dataset)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


counter = Counter()
batch_size = 1000
max_batches = 10000  

for i, batch in enumerate(batch_iterator(ds, batch_size)):
    texts = [x["text"] for x in batch]
    encodings = tokenizer(texts, return_attention_mask=False,truncation=True, max_length=1024)["input_ids"]
    for ids in encodings:
        counter.update(ids)
    if i % 100 == 0:
        print(f"Processed {i * batch_size} stories...")
    if i >= max_batches:
        break

freqs = Counter(counter.values())  # frequency of frequencies

x, y = zip(*freqs.items())
df = pd.DataFrame(freqs.items(), columns=["token_count", "num_tokens"])
p = (
    ggplot(df, aes(x="token_count", y="num_tokens")) +
    geom_col(width=1.0) +
    scale_x_log10() +
    scale_y_log10() +
    labs(
        x="Token Count",
        y="Number of Tokens with that Count",
        title="Histogram of Token Frequencies (log-log)"
    ) +
    theme(figure_size=(10, 6))
)