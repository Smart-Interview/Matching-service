import argparse
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import Adam
from utils import trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    orig_labels = [item['orig_label'] for item in batch]
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'orig_label': orig_labels
    }

parser = argparse.ArgumentParser(description='Train Bert-NER')
parser.add_argument('-e', type=int, default=5, help='number of epochs')
parser.add_argument('-o', type=str, default='.', help='output path to save model state')

args = parser.parse_args().__dict__

output_path = args['o']

MAX_LEN = 500
EPOCHS = args['e']
MAX_GRAD_NORM = 1.0
MODEL_NAME = 'bert-base-uncased'
TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME)  # Assurez-vous d'utiliser le bon tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = trim_entity_spans(convert_goldparse('data/Resumes.json'))

total = len(data)
train_data, val_data = data[:180], data[180:]

train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8, collate_fn=collate_fn)
val_dl = DataLoader(val_d, batch_size=4, collate_fn=collate_fn)

model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2idx))
model.to(DEVICE)
optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl
)

torch.save(
    {
        "model_state_dict": model.state_dict()
    },
    f'{output_path}/model-state.bin',
)
