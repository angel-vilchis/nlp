from torch import nn 
from configs.model.word2vec import MAX_WORD_LEN
from configs.model.rnn import RNN_HIDDEN_SIZE, NUM_LAYERS, RNN_TYPE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNModule(nn.Module):
    def __init__(self, input_size):
        super(RNNModule, self).__init__()
        self.hidden_size = RNN_HIDDEN_SIZE
        if RNN_TYPE == "gru":
            model = nn.GRU
        elif RNN_TYPE == "lstm":
            model = nn.LSTM
        else:
            model = nn.RNN
        self.rnn = model(bidirectional=True, input_size=input_size, hidden_size=RNN_HIDDEN_SIZE // 2, num_layers=NUM_LAYERS, batch_first=True)
        
    def forward(self, final_embeddings, word_mask):
        packed_input = pack_padded_sequence(final_embeddings, word_mask.sum(axis=1).tolist(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed_input)
        outs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=MAX_WORD_LEN)
        return outs