from email.headerregistry import SingleAddressHeader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import math

from transformers.data.processors.squad import squad_convert_example_to_features_init

torch.manual_seed(2048)

@dataclass
class GPTConfig:
    block_size: int = 512  #文本的最大长度
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  #hidden_dim, hidden_size;这里emb_size
    hidden_dim: int = n_embd
    #为了可以tie_embedding_weight
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    #vocab_size
    # gpt2 的官方的tokenizer
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size

        # attention_mask 通过 register_buffer 注册
        # 因为不用计算梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            "attention_mask",
            # tril 是下三角的意思
            # block_size 是 512
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)  #@是torch.matmul的简化写法
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )
        # 要注意计算weight的时候， 除以根号下d_k (为什么？）（Attention 公式）
        weight = F.softmax(weight, dim= -1) / math.sqrt(self.head_size)

        #dropout要放到weight后面
        weight = self.dropout(weight)
        output = weight @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 可以有更优雅的写法（矩阵旋转做法， 后面继续学习来补充）
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

#3. feed forward (MLP)
class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim), # swiglu #8/3
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
    def forward(self,x):
        return self.net(x)

#4. Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = SingleHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self,x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

#5. GPT
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 其他大模型的一些常见升级点：
        # position embedding 从 0， 1， xxx embedding升级到 rope
        # norm: layer norm-> rms norm
        # mlp -> swiglu
        # mha -> gqa
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 现在的 SLM 模型， 会用 tie weight 来减少参数 （非常重要）
        # linear(4 -> 8), weight 实际上的shape 是 8 * 4
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 初始化为正态分布
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        # idx 输入是 token ids,
        # targets 是目标 token ids (batch, seq_len)
        # shape 要一样
        batch, seq_len = idx.size() #(batch, seq_len)
        token_emb = self.token_embedding_table(idx) #（batch, seq_len, n_embed)
        pos_emb = self.position_embedding_table(
            # 要确保 位置编码和输入的 idx 在同一个设备上
            torch.arrange(seq_len, device = idx.device)
        )
        # 经典题目： token_embedding 和 position_embedding 可以相加吗？
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape (batch, seq_len)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = -1)
        return idx


class MyDataset(Dataset):
    def __init__(self, path, block_size = 512):
        # 数据在 /root/fs/mobvoi_seq_monkey_general_open_corpus.json1
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self. block_size = block_size #pos最大长度

        self. encoded_data = []
        # 特殊符号分割不同的训练文本
        # gpt 是 <|endoftext|>
        self.eos_token = self.enc.encoder(
            "<|endoftext|>",
            allowed_special = {"<|endoftext|>"}
        )[0]

        self.max_lines = 1000
        import json

        raw_data = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                    raw_data.append(text)
                except Exception as e:
                    continue

        full_encode = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encode.extend(encoded_text + [self.eos_token])

        # 由于 block_size 是 512
        # 因此需要将长切割成短（512）
        for i in range(0, len(full_encode), self.block_size):
            chunk = full_encode[i: i + self.block_size + 1] #512 每一行实际是513
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.Tensor(chunk[:-1], dtype = torch.long)
        y = torch.tensor(chunk[1:], dtype = torch.long)
        return x, y

    def encode(self, text):
        # 将文本编码为 token IDs
        return self.enc.encode(text)

    def decode(self, ids):
        # 将 token IDs解码为文本
        return self.enc.decode(ids)


# 训练
model = GPT(GPTConfig())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#打印模型一共有多少参数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params / 1e6} M")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#设置cosine 学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# train data
train_dataset = MyDataset('/root/LLM/mobvoi_seq_monkey_general_open_corpus.jsonl')

# split traindataset to train and val
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        return total_loss

def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

for epoch in range(2):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
    val_loss = eval(model, val_loader, device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    # 保存模型
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }
    # 保存每个epoch的模型
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')





