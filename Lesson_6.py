import torch
import torch.nn as nn
import torch.nn.functional as func


## ------------------------------------------------------##
class MaskedSelfAttention(nn.Module):

    def __init__(self, d_model = 2, row_dim = 0, col_dim = 1):

        super().__init__()

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodeings, mask = None):
        q = self.W_q(token_encodeings)
        k = self.W_k(token_encodeings)
        v = self.W_v(token_encodeings)

        sims = torch.matmul(q, k.transpose(dim0 = self.row_dim, dim1 = self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask = mask, value = -1e9)

        attention_percents = func.softmax(scaled_sims, dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

## ------------------------------------------------------##
encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])

torch.manual_seed(42)

maskedSelfAttention = MaskedSelfAttention(d_model = 2, row_dim = 0, col_dim = 1)

mask = torch.tril(torch.ones(3, 3))
mask = mask == 0
mask     # print()

## ------------------------------------------------------##
maskedSelfAttention(encodings_matrix, mask)

## ------------------------------------------------------##
maskedSelfAttention.W_q.weight.transpose(0, 1)

## ------------------------------------------------------##
maskedSelfAttention.W_k.weight.transpose(0, 1)

## ------------------------------------------------------##
maskedSelfAttention.W_v.weight.transpose(0, 1)

## ------------------------------------------------------##
maskedSelfAttention.W_q(encodings_matrix)

## ------------------------------------------------------##
maskedSelfAttention.W_k(encodings_matrix)

## ------------------------------------------------------##
maskedSelfAttention.W_v(encodings_matrix)

## ------------------------------------------------------##
q = maskedSelfAttention.W_q(encodings_matrix)
q   # print()

## ------------------------------------------------------##
k = maskedSelfAttention.W_k(encodings_matrix)
k   # print()

## ------------------------------------------------------##
sims = torch.matmul(q, k.transpose(dim0 = 0, dim1 = 1))
sims   # print()

## ------------------------------------------------------##
scaled_sims = sims / (torch.tensor(2) ** 0.5)
scaled_sims   # print()

## ------------------------------------------------------##
masked_scaled_sims = scaled_sims.masked_fill(mask = mask, value = -1e9)
masked_scaled_sims   # print()

## ------------------------------------------------------##
attention_percents = func.sofmax(masked_scaled_sims, dim = 1)
attention_percents   # print()

## ------------------------------------------------------##
torch.matmul(attention_percents, maskedSelfAttention.W_v(encodings_matrix))

