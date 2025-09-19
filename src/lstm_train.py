import torch
from torch import nn
from torch.optim import Adam
from src.lstm_model import LSTMGenerateWord

def model_train(train_dataloader, tokenizer, device):
    model = LSTMGenerateWord(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    num_epochs = 5
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(input_ids)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("Модель сохранена")
    
    return model