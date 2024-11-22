import torch
from model import POSModel
from data_preprocessing import prepare_data

import nltk
from nltk.corpus import treebank
# Load the model
def load_model(model_path, vocab_size, tagset_size, embedding_dim, hidden_dim):
    model = POSModel(vocab_size, tagset_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Test the model on a sentence
def test_model(model, sentence, vocab, tagset):
    model.eval()
    with torch.no_grad():
        inputs = [vocab.get(word.lower(), vocab['<unk>']) for word in sentence.split()]
        inputs_tensor = torch.tensor(inputs).unsqueeze(0)
        tag_scores = model(inputs_tensor, torch.tensor([len(inputs)]))
        tag_ids = torch.argmax(tag_scores, dim=2)
        tags = [list(tagset.keys())[tag_id] for tag_id in tag_ids[0]]
        return list(zip(sentence.split(), tags))

nltk.download('treebank')
nltk.download('universal_tagset')

# Load the data
sentences = treebank.tagged_sents(tagset='universal')
train_data = sentences[:3000]

vocab = {word for sent in train_data for word, tag in sent}
vocab = {word: i+2 for i, word in enumerate(sorted(vocab))}
vocab['<pad>'] = 0  # Padding
vocab['<unk>'] = 1  # Unknown words

tagset = {tag for sent in train_data for word, tag in sent}
tagset = {tag: i for i, tag in enumerate(tagset)}

# Load the model
model_loaded = load_model('pos_model.pth', len(vocab), len(tagset), 100, 128)

# Test the model on a new sentence
sentence = "A very good man"
tagged_sentence = test_model(model_loaded, sentence, vocab, tagset)
print(tagged_sentence)
