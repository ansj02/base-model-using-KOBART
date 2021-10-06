from torch import nn
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from nltk.tokenize import sent_tokenize
import os.path


def sentence_tokenize(text, bos_token, eos_token):
    new_text = ''
    for sent in sent_tokenize(text):
        sent_token = bos_token+sent+eos_token
        new_text += sent_token
    return new_text

def encode_sentences(tokenizer, source_sentences, target_sentences):
    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        sentence = sentence_tokenize(sentence, '<s>', '</s>')
        encoded_dict = tokenizer.encode_plus(sentence, padding='max_length', max_length=400, truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    for sentence in target_sentences:
        encoded_dict = tokenizer.encode_plus(sentence, padding='max_length', max_length=400, truncation=True)
        target_ids.append(encoded_dict['input_ids'])
    target_ids = torch.tensor(target_ids)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }
    return batch

class DataModule():
    def __init__(self, tokenizer, data_path, batch_size=4, num_examples=20000):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.data = pd.read_csv(data_path)[:self.num_examples]
        self.batch_size = batch_size
        #self.train, self.validate = np.split(self.data.sample(frac=1), [int(0.7*len(self.data)), int(0.3*len(self.data))])
        self.train = self.data[int(0.6*len(self.data)):int(0.9*len(self.data))]
        self.validate = self.data[int(0.9*len(self.data)):]
    def setup(self):
        self.train = encode_sentences(self.tokenizer, self.train['text'], self.train['summary'])
        self.validate = encode_sentences(self.tokenizer, self.validate['text'], self.validate['summary'])
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data
    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)
        return val_data

class KoBartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def train(model, tokenizer, optimizer, device, save_path, train_data_path, save_cycle = 20):
    if os.path.isfile(save_path) : model.load_state_dict(torch.load(save_path))
    data = DataModule(tokenizer, train_data_path)
    data.setup()
    train_data = data.train_dataloader()
    model = model.to(device)
    for idx, (input_ids, attention_mask, labels) in enumerate(train_data):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        loss = model(input_ids, attention_mask, labels)['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        if idx % save_cycle == 0 :
            torch.save(model.state_dict(), save_path)
            print('save ', idx)
    torch.save(model.state_dict(), save_path)

def test(model, tokenizer, save_path, test_data_path):
    model.load_state_dict(torch.load(save_path))
    test_data = pd.read_csv(test_data_path)

    test_sentences = test_data['text']
    summary = []

    for i, sentence in enumerate(test_sentences):
        sentence = sentence_tokenize(sentence, '<s>', '</s>')
        inputs = tokenizer([sentence], return_tensors='pt')
        if len(model.model.generate(inputs['input_ids'])) > 0:
            summary.append(tokenizer.decode(model.model.generate(inputs['input_ids'])[0]))
        else:
            summary.append('')
        summary_db = pd.DataFrame(summary)
        if i % 100 == 0:
            summary_db = pd.DataFrame(summary)
            summary_db.to_csv('result.csv')
    summary_db.to_csv('result.csv')
