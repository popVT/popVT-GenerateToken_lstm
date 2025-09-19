from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=20):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embeddings = []

        # токенизация текста
        for text in texts:
            tokens = tokenizer(text, truncation=True, max_length=max_length, padding=False)

            if len(tokens['input_ids']) > 1:
                self.embeddings.append(tokens['input_ids'])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        input_ids = self.embeddings[idx]

        x = input_ids[:-1]
        y = input_ids[1:]

        return {
            'input_ids': x,
            'labels': y
            }