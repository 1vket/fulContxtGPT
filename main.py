from src import GPT, fulContxtDataset, Trainer
import defect
import json
import torch


# load config
config_filename = "config.yaml"

config = defect.torch_util.load_config(config_filename)

# model 
model = GPT(config.model)

# dataset
train_filename = './data/cc100set/001.txt'
test_filename = './data/cc100split/00002.txt'
json_filename = './token.json'

train_dataset = fulContxtDataset(train_filename, json_filename)
test_dataset = fulContxtDataset(test_filename, json_filename)

# trainer
trainer = Trainer(
  model, train_dataset, test_dataset, config.train)

trainer.train()

