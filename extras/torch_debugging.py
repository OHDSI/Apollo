import torch

loaded = torch.load("d:/GPM_MDCD/model/token_predictions_25.pt", map_location=torch.device('cpu'))

token_ids = loaded["token_ids"]
token_ids.shape

token_predictions = loaded["token_predictions"]
token_predictions.shape

inputs = loaded["inputs"]

masked_token_ids = inputs["masked_token_ids"]
for i in range(32):
    print(masked_token_ids[i].tolist())

outputs = loaded["outputs"]

import torch
for j in range(32):
    print(j)
    test_e = []
    test_m = []
    for i in range(32):
        if i != j:
            test_e.append(embedded[i].unsqueeze(0))
            test_m.append(padding_mask[i].unsqueeze(0))
        else:
            test_e.append(good_embedded[i].unsqueeze(0))
            test_m.append(good_padding_mask[i].unsqueeze(0))

    test_e = torch.cat(test_e, out=torch.Tensor(32, 512, 768))
    test_m = torch.cat(test_m, out=torch.Tensor(32, 512))

    encoded = self.transformer_encoder(src=test_e, src_key_padding_mask=test_m)
    print(encoded.shape)



torch.save({"embedded": embedded, "padding_mask": padding_mask}, "d:/GPM_MDCD/model/embedded.pt")

loaded = torch.load("d:/GPM_MDCD/model/embedded.pt")
good_embedded = loaded["embedded"]
good_padding_mask = loaded["padding_mask"]

embedded[7]
good_embedded[7]

padding_mask[7]
good_padding_mask[7]




from torch import nn

d_model = 32
nhead = 4
dim_feedforward = 48
num_layers = 2
batch_size = 3
seq_length = 5

layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dim_feedforward=dim_feedforward)
xencoder = nn.TransformerEncoder(layer, num_layers=num_layers)

x = torch.rand(batch_size, seq_length, d_model)
mask = torch.zeros(batch_size, seq_length, dtype=bool)

xencoder.train()
print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.eval()
print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.train()
with torch.no_grad():
    print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.eval()
with torch.no_grad():
    print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

# Set last mask to True for all batches:
for i in range(batch_size):
    mask[i, -1] = True

xencoder.train()
print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.eval()
print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.train()
with torch.no_grad():
    print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 5, 32])

xencoder.eval()
with torch.no_grad():
    print(xencoder(x, src_key_padding_mask=mask).shape)
# torch.Size([3, 4, 32]) # Last token is truncated