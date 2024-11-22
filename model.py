import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gates parameters
        self.W_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_c = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_dim).to(x.device), 
                        torch.zeros(batch_size, self.hidden_dim).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class POSModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(POSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = CustomLSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words, word_lengths):
        embeds = self.embedding(words)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])
        tag_space = self.fc(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores.view(words.size(0), words.size(1), -1)
