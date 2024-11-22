import torch
import nltk
from torch.utils.data import Dataset
from nltk.corpus import treebank

# nltk.download('treebank')
# nltk.download('universal_tagset')

def prepare_data():
    # Load the data
    sentences = treebank.tagged_sents(tagset='universal')
    train_data = sentences[:3000]
    test_data = sentences[3000:4000]

    # Create vocabulary
    vocab = {word for sent in train_data for word, tag in sent}
    vocab = {word: i+2 for i, word in enumerate(sorted(vocab))}
    vocab['<pad>'] = 0  # Padding
    vocab['<unk>'] = 1  # Unknown words

    # Create tagset
    tagset = {tag for sent in train_data for word, tag in sent}
    tagset = {tag: i for i, tag in enumerate(tagset)}

    return train_data, test_data, vocab, tagset


class POSDataset(Dataset):
    def __init__(self, data, vocab, tagset):
        self.data = data
        self.vocab = vocab
        self.tagset = tagset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = zip(*self.data[idx])
        word_ids = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        tag_ids = [self.tagset[tag] for tag in tags]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=-1)  # -1 for ignore_index in loss calculation
    return xx_pad, yy_pad, torch.tensor(x_lens, dtype=torch.long)
