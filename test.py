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

# model 
model = GPT(config.model)
model.load_state_dict(torch.load(config.train.ckpt_path))

# predict
src = input('>>')
src = defect.text.sentence2phoneSymbol(src)
inp = src.copy()
print(f"input:",*(src[:-1]))
src = [tokenizer['p2i'][p] for p in src]
src = torch.LongTensor(src[:-1]).unsqueeze(0)
out = model.predict(src, device='cpu').squeeze(0)
out = [tokenizer['i2p'][str(int(i))] for i in out]

with open('out.txt', 'w') as f:
  f.write(' '.join(inp))
  f.write('\n')
  f.write(' '.join(out))

print(f"output:",*out)

waveform, sr = pyopenjtalk.synthesize(out)

print("create")

sf.write('out.wav', waveform, sr)



