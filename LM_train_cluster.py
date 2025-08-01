import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import math
import time
import os
import tiktoken
import matplotlib.pyplot as plt
import datasets
from itertools import cycle
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ROC_curve_micro(pred_tensor, label_tensor):
    n_class=pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class) 
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive.flatten()
    fp_diff = is_negative.flatten()
    thresh_tensor = -pred_tensor.flatten()
    fn_denom = is_positive.sum()
    fp_denom = is_negative.sum()
    sorted_indices = torch.argsort(thresh_tensor)
    sorted_fp_cum = fp_diff[sorted_indices].cumsum(0) / fp_denom
    sorted_fn_cum = -fn_diff[sorted_indices].flip(0).cumsum(0).flip(0) / fn_denom

    sorted_thresh = thresh_tensor[sorted_indices]
    sorted_is_diff = sorted_thresh.diff() != 0
    sorted_fp_end = torch.cat([sorted_is_diff, torch.tensor([True])])
    sorted_fn_end = torch.cat([torch.tensor([True]), sorted_is_diff])

    uniq_thresh = sorted_thresh[sorted_fp_end]
    uniq_fp_after = sorted_fp_cum[sorted_fp_end]
    uniq_fn_before = sorted_fn_cum[sorted_fn_end]

    FPR = torch.cat([torch.tensor([0.0]), uniq_fp_after])
    FNR = torch.cat([uniq_fn_before, torch.tensor([0.0])])

    return {
        "FPR": FPR,
        "FNR": FNR,
        "TPR": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([torch.tensor([-1]), uniq_thresh]),
        "max_constant": torch.cat([uniq_thresh, torch.tensor([0])])
    }
def ROC_AUC_micro(pred_tensor, label_tensor):
    roc = ROC_curve_micro(pred_tensor, label_tensor)
    FPR_diff = roc["FPR"][1:]-roc["FPR"][:-1]   
    TPR_sum = roc["TPR"][1:]+roc["TPR"][:-1]
    return torch.sum(FPR_diff*TPR_sum/2.0)
#AUM 
def Proposed_AUM_micro(pred_tensor, label_tensor):

    roc = ROC_curve_micro(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1]
    constant_diff = roc["min_constant"][1:].diff()
    return torch.sum(min_FPR_FNR * constant_diff)
def ROC_curve_macro(pred_tensor, label_tensor):
    device = pred_tensor.device  
    n_class = pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class).to(device)
    is_positive = one_hot_labels
    is_negative = 1 - one_hot_labels
    fn_diff = -is_positive
    fp_diff = is_negative
    thresh_tensor = -pred_tensor
    fn_denom = is_positive.sum(dim=0).clamp(min=1)
    fp_denom = is_negative.sum(dim=0).clamp(min=1)
    sorted_indices = torch.argsort(thresh_tensor, dim=0)
    sorted_fp_cum = torch.div(
        torch.gather(fp_diff, dim=0, index=sorted_indices).cumsum(0),
        fp_denom
    )
    sorted_fn_cum = -torch.div(
        torch.gather(fn_diff, dim=0, index=sorted_indices).flip(0).cumsum(0).flip(0),
        fn_denom
    )
    sorted_thresh = torch.gather(thresh_tensor, dim=0, index=sorted_indices)
    zeros_vec = torch.zeros(1, n_class, device=device)
    FPR = torch.cat([zeros_vec, sorted_fp_cum])
    FNR = torch.cat([sorted_fn_cum, zeros_vec])
    return {
        "FPR_all_classes": FPR,
        "FNR_all_classes": FNR,
        "TPR_all_classes": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([-torch.ones(1, n_class, device=device), sorted_thresh]),
        "max_constant": torch.cat([sorted_thresh, zeros_vec])
    }


def ROC_AUC_macro(pred_tensor, label_tensor):
    roc = ROC_curve_macro(pred_tensor, label_tensor)
    FPR_diff = roc["FPR_all_classes"][1:,:]-roc["FPR_all_classes"][:-1,]
    TPR_sum = roc["TPR_all_classes"][1:,:]+roc["TPR_all_classes"][:-1,:]
    sum_FPR_TPR= torch.sum(FPR_diff*TPR_sum/2.0,dim=0)
    count_non_defined=(sum_FPR_TPR == 0).sum()
    if count_non_defined==pred_tensor.size(1):
        return torch.tensor(0.0,device=pred_tensor.device)
    return  sum_FPR_TPR.sum()/(pred_tensor.size(1)-count_non_defined)

def Proposed_AUM_macro(pred_tensor, label_tensor):
    roc = ROC_curve_macro(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1, :]
    constant_diff = roc["min_constant"][1:, :].diff(dim=0)
    sum_min = torch.sum(min_FPR_FNR * constant_diff, dim=0)
    count_non_defined = (sum_min == 0).sum()
    if count_non_defined == pred_tensor.size(1):
        return torch.tensor(0.0, device=pred_tensor.device)

    return sum_min.sum() / (pred_tensor.size(1)-count_non_defined)

loss_dict={
    "AUM_micro": Proposed_AUM_micro,
    "AUM_macro": Proposed_AUM_macro,
    "Cross-entropy": F.cross_entropy
}
loss_fn_str=sys.argv[1]
loss_fn=loss_dict[loss_fn_str]


# A simple configuration container
class GPTConfig:
    def __init__(
        self, 
        vocab_size,  # size of the vocabulary, from tokenizer, for gpt2 tokenizer it is 50257
        n_layer,   # number of transformer blocks
        n_head,    # number of attention heads for each transformer block
        n_embd,  # embedding dimension for each token ie: how many dimensions in each token-vector
        seq_len,  # sequence length for the model - e.g. the "context window" 
    
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.seq_len = seq_len
    
def get_position_encoding(seq_len, d, n=10000):
    
    P = torch.zeros(seq_len, d).to(device)
    for pos in range(seq_len):
        for i in range(0, d // 2):
            P[pos, 2 * i] = math.sin(pos / (n ** ((2 * i) / d)))
            if i + 1 < d:
                P[pos, 2* i + 1] = math.cos(pos / (n ** ((2 * i) / d)))

    return P.unsqueeze(0)
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Query weights - will transform input embeddings into queries
        self.Wk = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Key weights - will transform input embeddings into keys
        self.Wv = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Value weights - will transform input embeddings into values

    def forward(self, x):
        print("Attention input shape:", x.shape)
        print("")
        print("Query weights shape:", self.Wq.shape)
        print("Key weights shape:", self.Wk.shape)
        print("Value weights shape:", self.Wv.shape)
        queries = x @ self.Wq # Matrix multiplication to transform input embeddings into queries
        keys = x @ self.Wk # Matrix multiplication to transform input embeddings into keys
        values = x @ self.Wv # Matrix multiplication to transform input embeddings into values
        print("")
        print("Queries shape:", queries.shape)
        print("Keys shape:", keys.shape)
        print("Values shape:", values.shape)

        qkt = queries @ keys.transpose(-2, -1) # Calculate QK^T
        qkt_scaled = qkt / math.sqrt(queries.size(-1)) # Scale QK^T by the dimension of the keys
        qkt_softmax = F.softmax(qkt_scaled, dim=-1) # Apply softmax row-wise to get attention weights
        print("")
        print("QK^T shape:", qkt.shape)

        attn_output = qkt_softmax @ values # Multiply softmax(QK^T) by values
        print("")
        print("Attention output shape:", attn_output.shape)
        return attn_output 
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Query weights - will transform input embeddings into queries
        self.Wk = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Key weights - will transform input embeddings into keys
        self.Wv = nn.Parameter(torch.randn(config.n_embd, config.n_embd)).to(device) # Value weights - will transform input embeddings into values

    def forward(self, x):
        seq_len = x.shape[1] # Get sequence length (number of tokens / context window length)
        queries = x @ self.Wq # Matrix multiplication to transform input embeddings into queries
        keys = x @ self.Wk    # Matrix multiplication to transform input embeddings into keys
        values = x @ self.Wv  # Matrix multiplication to transform input embeddings into values
        qkt = queries @ keys.transpose(-2, -1)  # Calculate QK^T
        qkt_scaled = qkt / math.sqrt(queries.size(-1))  # Scale QK^T by the dimension of the keys

        # MASKING
        # THIS IS THE ONLY DIFFERENCE, USE -inf FOR UPPER TRIANGLE MASK SO THAT SOFTMAX WILL BE 0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))  # Upper triangle masked with -inf 
        qkt_scaled = qkt_scaled + causal_mask # Add the mask to the scaled QK^T
        # END MASKING

        qkt_softmax = F.softmax(qkt_scaled, dim=-1) # Apply softmax row-wise to get attention weights, the -inf values will become 0 here
        attn_output = qkt_softmax @ values # Multiply softmax(QK^T) by values
        return attn_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_heads = nn.ModuleList([
            CausalSelfAttention(config) for _ in range(config.n_head)
        ])  # Create n_head attention heads
        self.projection = nn.Linear(config.n_embd * config.n_head, config.n_embd).to(device) # Linear layer to project multi-head attention outputs

    def forward(self, x):
        head_outputs = [head(x) for head in self.attn_heads] # Get the output of each attention head
        multihead_output = torch.cat(head_outputs, dim=-1) # Concatenate the outputs
        return self.projection(multihead_output) # Project the concatenated outputs
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        ).to(device)
        self.ln2 = nn.LayerNorm(config.n_embd).to(device)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd).to(device)
        self.position_encoding = get_position_encoding(config.seq_len, config.n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd).to(device)
        self.head = nn.Linear(config.n_embd, config.vocab_size).to(device)
    
    def forward(self, x):
        x = self.token_embedding(x) + self.position_encoding
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)



# Example config:
batch_size = 64
sequence_len = 128
num_steps = 15000
accumulation_steps = 100

# Reload the train and test datasets
train_ds = datasets.load_dataset("parquet", data_files="Tinystories_train.parquet", split="train")
test_ds = datasets.load_dataset("parquet", data_files="Tinystories_test.parquet", split="train")

hf_tokenizer = AutoTokenizer.from_pretrained("gpt2_local")

# Convert dataset to PyTorch format
train_ds.set_format("torch", columns=["input", "target"])
test_ds.set_format("torch", columns=["input", "target"])

# Create DataLoaders for training and testing
train_dataloader = cycle(DataLoader(train_ds, batch_size=batch_size, shuffle=False))
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

config = GPTConfig(
    vocab_size=hf_tokenizer.vocab_size,
    n_layer=4,   # fewer layers for a quick demo
    n_head=4,
    n_embd=64,
    seq_len=sequence_len,
)

# Create the GPT model
model = GPTModel(config)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)


# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.3, patience=10, min_lr=5e-6, threshold=1e-4)


# Training loop
losses = []
test_losses = []
accumulator = 0
accumulator_loss = 0
start_time = time.time()
for i in range(num_steps):
    model.train()
    example = next(train_dataloader)
    train_input = example["input"].to(device)
    train_target = example["target"].to(device)

    logits = model(train_input)
    loss = loss_fn(logits.view(-1, logits.size(-1)), train_target.view(-1))
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    accumulator += 1
    accumulator_loss += loss.item()

    
    if accumulator >= accumulation_steps:
        losses.append(accumulator_loss / accumulation_steps)
        accumulator = 0
        accumulator_loss = 0
        model.eval()
        test_loss = 0
        test_accumulator = 0
        print("Here")
        with torch.no_grad():
            for test_example in test_dataloader:
                test_input = test_example["input"].to(device)
                test_target = test_example["target"].to(device)
                test_logits = model(test_input)
                test_loss += loss_fn(test_logits.view(-1, test_logits.size(-1)), test_target.view(-1)).item()
                test_accumulator += 1
            test_losses.append(test_loss / test_accumulator)
            elapsed_time = time.time() - start_time
            print(f"Step {i+1}/{num_steps}, Loss: {losses[-1]}, Test Loss: {test_losses[-1]}, LR: {optimizer.param_groups[0]['lr']}, Elapsed Time: {elapsed_time:.2f} seconds")
            test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            scheduler.step(test_losses[-1])


    if (i+1) % 5000 == 0:
        # Save the model checkpoint
        print(f"Saving model checkpoint at step {i+1}")
        torch.save(model, f"./model_checkpoint_{i}.pt")
df = pd.DataFrame({
    'epoch': list(range(1, len(losses) + 1)),
    'train_loss': losses,
    'test_loss': test_losses
})
df.to_csv('training_log.csv', index=False)
