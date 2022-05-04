from src import GPT, fulContxtDataset, Trainer
import defect
import json
import torch
import soundfile as sf
import pyopenjtalk


# load config
config_filename = "config.yaml"

config = defect.torch_util.load_config(config_filename)

# json
json_filename = './token.json'
with open(json_filename, 'r') as jf:
  tokenizer = json.load(jf)
retokenizer = {v:k for k,v in tokenizer.items()}

# model 
model = GPT(config.model)
#model.load_state_dict(torch.load('./logg/728emb12L12nt/0_state_dict'))
model.load_state_dict(torch.load('./logg/finetune/state_dict'))
print(model.config.eos_idx)

torch.save(model, 'finetuned-GPT')

# predict
src = input('>>')
src = defect.text.sentence2phoneSymbol(src)
inp = src.copy()
print(f"input:",*(src[:-1]))
src[-1] = '|'
src = [tokenizer[p] for p in src] 
src = torch.LongTensor(src).unsqueeze(0)
out = model.predict(src, device='cpu').squeeze(0)
out = [retokenizer[int(i)] for i in out]

with open('out.txt', 'w') as f:
  f.write(' '.join(inp))
  f.write('\n')
  f.write(' '.join(out))

print(f"output:",*out)



