from src import GPT, fulContxtDataset, Trainer
import defect
import json
import torch

epoch = 10

# load config
config_filename = "config.yaml"
json_filename = './token.json'

config = defect.torch_util.load_config(config_filename)

# model 
model = GPT(config.model)

# trainer
trainer = Trainer(
  model, None, None, config.train)

# dataset
train_filename = [f"./data/data/"+f"000{i}.txt"[-7:] for i in range(1, 151)]
test_filename = './data/data/160.txt'

for e in range(epoch):
  for i, train in enumerate(train_filename):
    train_dataset = fulContxtDataset(train, json_filename)
    test_dataset = fulContxtDataset(test_filename, json_filename)

    trainer.train_dataset = train_dataset
    if i == len(train_filename) -1:
      trainer.test_dataset = test_dataset
    else:
      trainer.test_dataset = None

    trainer.train()
    trainer.save_checkpoint(str(e))

