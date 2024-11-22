import torch
import torch.optim as optim
from model import POSModel
from data_preprocessing import prepare_data, POSDataset, pad_collate
from torch.utils.data import DataLoader

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
LEARNING_RATE = 0.01

# Prepare data
train_data, test_data, vocab, tagset = prepare_data()

train_loader = DataLoader(POSDataset(train_data, vocab, tagset), batch_size=32, collate_fn=pad_collate)
test_loader = DataLoader(POSDataset(test_data, vocab, tagset), batch_size=32, collate_fn=pad_collate)

# Initialize the model
model = POSModel(len(vocab), len(tagset), EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore the padding in the loss calculation

# Train the model
def train_model(model, train_loader, optimizer, loss_function, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for words, tags, lengths in train_loader:
            model.zero_grad()
            tag_scores = model(words, lengths)
            loss = loss_function(tag_scores.view(-1, len(tagset)), tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss}")

train_model(model, train_loader, optimizer, loss_function, epochs=EPOCHS)

# Save the model
torch.save(model.state_dict(), 'moedl/pos_model.pth')
