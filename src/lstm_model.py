import torch
from torch import nn

class LSTMGenerateWord(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, hidden=None):
        x = self.embedding(x)

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        return out, hidden

    def generate(self, tokenizer, prompt, max_length=20, device='cpu'):
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(prompt.lower(), return_tensors='pt').to(device)
            generated = tokens.clone()

            for _ in range(max_length - tokens.size(1)):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :]  # последний токен
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            return tokenizer.decode(generated[0], skip_special_tokens=True)



