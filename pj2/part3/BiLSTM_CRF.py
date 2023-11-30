import torch
import torch.nn as nn
from utils import argmax, log_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CRF():
    def __init__(self, tags_dic):
        self.tags_dic = tags_dic
        self.tags_sz = len(self.tags_dic)
        self.BEGIN_TAG = "<s>" 
        self.END_TAG = "</s>" 

        self.trans = nn.Parameter(torch.randn(self.tags_sz, self.tags_sz)).to(device)
        self.trans.data[self.tags_dic[self.BEGIN_TAG], :] = -99999
        self.trans.data[:, self.tags_dic[self.END_TAG]] = -99999

    def _forward_(self, feats, seq_len):
        init_alphas = torch.full((self.tags_sz,), -99999.).to(device)
        init_alphas[self.tags_dic[self.BEGIN_TAG]] = 0.

        # shape：(batch_size, seq_len + 1, tagset_size)
        log_prob = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32).to(device)
        log_prob[:, 0, :] = init_alphas # reset start of each sentence to init_alphas

        # shape：(tagset_size, tagset_size) -> (batch_size, tagset_size, tagset_size)
        trans = self.trans.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        for seq_i in range(feats.shape[1]):
            emit = feats[:, seq_i, :]
            raw_prob = (log_prob[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  # (batch_size, tagset_size, tagset_size)
                       + trans + emit.unsqueeze(2).repeat(1, 1, feats.shape[2]))
            # must use clone, or getting error
            cloned = log_prob.clone()
            cloned[:, seq_i + 1, :] = log_sum(raw_prob)
            log_prob = cloned

        log_prob = log_prob[range(feats.shape[0]), seq_len, :]
        log_prob += self.trans[self.tags_dic[self.END_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        return log_sum(log_prob)

    def _cur_score_(self, feats, tags, seq_len):
        score = torch.zeros(feats.shape[0]).to(device)
        start_tag = torch.tensor([self.tags_dic[self.BEGIN_TAG]]).unsqueeze(0).repeat(feats.shape[0], 1).to(device)
        tags = torch.cat([start_tag, tags], dim=1)
        for batch_i in range(feats.shape[0]):
            score[batch_i] = torch.sum(self.trans[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]]) \
                + self.trans[self.tags_dic[self.END_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    def _loss_neg_log_(self, feats, tags, seq_len):
        forward_score = self._forward_(feats, seq_len)
        gt_score = self._cur_score_(feats, tags, seq_len)
        return torch.mean(forward_score - gt_score)

    def _viterbi_decode_(self, feats):
        states = []
        log_prob = torch.full((1, self.tags_sz), -99999.).to(device)
        log_prob[0][self.tags_dic[self.BEGIN_TAG]] = 0

        for feat in feats:
            state = []  # holds the backpointers for this step
            state_prob = []  # holds the viterbi variables for this step

            for next_tag in range(self.tags_sz):
                next_prob = log_prob + self.trans[next_tag]
                best_tag_id = argmax(next_prob)
                state.append(best_tag_id)
                state_prob.append(next_prob[0][best_tag_id].view(1))
            log_prob = (torch.cat(state_prob) + feat).view(1, -1)
            states.append(state)

        log_prob += self.trans[self.tags_dic[self.END_TAG]]
        best_tag_id = argmax(log_prob)
        path_score = log_prob[0][best_tag_id]

        best_path = [best_tag_id]
        for state in reversed(states):
            best_tag_id = state[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tags_dic[self.BEGIN_TAG]  
        best_path.reverse()
        return path_score, best_path


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, words_dict, tags_dict):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  
        self.hidden_dim = hidden_dim  
        self.words_size = len(words_dict)  
        self.tags_size = len(tags_dict)  
        self.state = 'train'  # 'train'、'eval'、'test'

        self.word_embeds = nn.Embedding(self.words_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tags_size, bias=True)
        self.crf = CRF(tags_dict)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def _get_lstm_features(self, sent, seq_len):
        embeds = self.word_embeds(sent)
        self.dropout(embeds)

        seq_len_cpu = seq_len.to("cpu")
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len_cpu, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        seqence_output = self.layer_norm(seq_unpacked)
        lstm_feats = self.hidden2tag(seqence_output)
        return lstm_feats

    def forward(self, sent, seq_len, tags=''):
        feats = self._get_lstm_features(sent, seq_len)
        if self.state == 'train':
            loss = self.crf._loss_neg_log_(feats, tags, seq_len)
            return loss
        elif self.state == 'eval':
            all_tag = []
            for i, feat in enumerate(feats):
                all_tag.append(self.crf._viterbi_decode_(feat[:seq_len[i]])[1])
            return all_tag
        else:
            return self.crf._viterbi_decode_(feats[0])[1]


 