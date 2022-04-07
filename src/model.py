import math
import logging
import defect

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Attention(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0

    self.key = nn.Linear(config.n_embd, config.n_embd)
    self.query = nn.Linear(config.n_embd, config.n_embd)
    self.value = nn.Linear(config.n_embd, config.n_embd)

    self.attn_drop = nn.Dropout(config.attn_pdrop)
    self.resid_drop = nn.Dropout(config.resid_pdrop)

    self.proj = nn.Linear(config.n_embd, config.n_embd)

    self.register_buffer(
      "mask", torch.tril(
        torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size))

    self.n_head = config.n_head

  def forward(self, x):
    B, T, C = x.size()

    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    y = self.resid_drop(self.proj(y))
    return y


class Block(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)

    self.attn = Attention(config)
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(),
      nn.Linear(4 * config.n_embd, config.n_embd),
      nn.Dropout(config.resid_pdrop)
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class GPT(nn.Module):
  
  def __init__(self, config):
    super().__init__()

    self.config = config

    self.tok_emb = nn.Embedding(
      config.vocab_size, config.n_embd, padding_idx=config.pad_idx)
    self.pos_emb = nn.Parameter(
      torch.zeros(1, config.block_size, config.n_embd))
    self.drop = nn.Dropout(config.embd_pdrop)
    
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

    self.ln_f = nn.LayerNorm(config.n_embd)
    self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.block_size = config.block_size
    self.apply(self._init_weights)

    logger.info("number of parameters: %e", 
      sum(p.numel() for p in self.parameters()))

  def get_block_size(self):
    return self.block_size

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      torch.nn.init.zeros_(module.bias)
      torch.nn.init.ones_(module.weight)
    elif isinstance(module, GPT):
      torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

  def configure_optimizers(self, train_config):
    
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn

        if pn.endswith('bias'):
          no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
          no_decay.add(fpn)

    no_decay.add('pos_emb')

    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, \
      "parameters %s made it into both decay/no_decay sets!" \
        % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
      "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )
    
    optim_gropus = [
      {"params": [param_dict[pn] for pn in sorted(list(decay))],
        "weight_decay": train_config.weight_decay},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
        "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(
      optim_gropus, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer

  def forward(self, idx, targets=None):
    b, t = idx.size()
    assert t <= self.block_size, \
      "Cannot forward, model block size is exhausted."

    token_embeddings = self.tok_emb(idx)
    position_embeddings = self.pos_emb[:, :t, :]
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), 
          ignore_index=self.config.pad_idx)
    
    return logits, loss

  def predict(self, src, device='cpu', beam=3):
    self.eval()
    self.to(device)

    b, t = src.size()
    assert t <= self.block_size, \
      "Cannot forward, model block size is exhausted."

    src_list = []
    src_list.append(src)

    for i in range(int(self.config.max_length)):
      if len(ans) >= 3:
        break

      src_tmp = []
      for pr, src in src_list:
        b, t = src.size()
        token_embeddings = self.tok_emb(src)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        prob = sorted([[p,i] for i,p in enumerate(logits[0][-1])])[:beam]

        for p, idx in prob:
          src = torch.cat(
            (src, torch.LongTensor(idx).view(1,1)), dim=-1).to(device)

          if idx == self.config.eos_idx:
            ans.append([pr*p, src])
            break
          else:
            src_tmp.append([pr*p, src])

      for p, src in sorted(src_tmp)[:beam]:
        src_list.append([p, src])

    return ans


if __name__ == "__main__":
  config_filename = 'config.yaml'
  config = defect.torch_util.load_config(config_filename)

  gpt = GPT(config)

  




