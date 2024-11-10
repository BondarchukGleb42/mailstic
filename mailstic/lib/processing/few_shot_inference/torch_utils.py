from torch.utils.data import Dataset, DataLoader
import torch


class EvalDataset(Dataset):
    def __init__(self, data, target_column="target"):

        data = data.reset_index(drop=True)

        self.texts = data['full_text']
        self.labels = data[target_column]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
        
        
def tokenize(text, tokenizer, max_length, return_tensors=None):
    return tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors=return_tensors)['input_ids']

def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)
    
    
class DataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def padding(self, encodings):
        max_length = max(len(encoding) for encoding in encodings)
        attention_masks = torch.stack([torch.tensor([1] * len(enc) + [0] * (max_length - len(enc))) for enc in encodings])
        encodings = torch.stack([torch.tensor(enc + [0] * (max_length - len(enc))) for enc in encodings])

        return {'input_ids': encodings, 'attention_mask': attention_masks}

    def __call__(self, batch_unprocessed):
        
        batch = {'encodings': [], 'labels': []}

        for text, label in batch_unprocessed:
            batch['encodings'].append(tokenize(text, tokenizer=self.tokenizer, max_length=self.max_length))
            batch['labels'].append(label)

        batch['encodings'] = self.padding(batch['encodings'])

        
        return batch