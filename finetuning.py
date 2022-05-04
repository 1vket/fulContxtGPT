from src import GPT, fulContxtDataset, Trainer
import defect
import json
import torch
import sys

epoch = 10

# load config
config_filename = "config.yaml"
json_filename = './token.json'

config = defect.torch_util.load_config(config_filename)

# model 
model = GPT(config.model)
model.load_state_dict(torch.load(sys.argv[1]))

# dataset
train_filename = './data/tdata/fcl_train.txt'
test_filename = './data/tdata/fcl_test.txt'

train_dataset = fulContxtDataset(train_filename, json_filename)
test_dataset = fulContxtDataset(test_filename, json_filename)


# trainer
trainer = Trainer(
  model, train_dataset, test_dataset, config.train)

trainer.train()
