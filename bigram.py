import torch
import torch.nn as nn
import torch.nn.functional as F

# Global Variables
vDebug = 1
vBatchSize = 32
vBlockSize = 8
vTrainingIterations = 10000
vDevice = 'cuda'

# Reading the input file
with open('input/shakespeare.txt', mode= 'r', encoding= 'utf-8') as f:
    input_text = f.read()

if vDebug:
    print("----------------------INPUT----------------------")
    print(input_text[:100])

# Vocab Size --> Unique characters present in the input text
vVocab = sorted(list(set(input_text)))
vVocabSize = len(vVocab)

if vDebug:
    print("----------------------VOCAB----------------------")
    print(''.join(vVocab))

# Vocab dictionary
itos = {i:s for i,s in enumerate(vVocab)}
stoi = {s:i for i,s in enumerate(vVocab)}

# Basic encoder decoder
encode = lambda s: [stoi[l] for l in s]
decode = lambda i: ''.join([itos[c] for c in i])

input_encoded = torch.tensor(encode(input_text), dtype = torch.long)

if vDebug:
    print("----------------------ENCODE----------------------")
    print(encode("Hello"))
    print("----------------------DECODE----------------------")
    print(decode([20, 43, 50, 50, 53]))

# Splitting the input dataset into test and train
n = int(0.9*len(input_encoded))
ds_train = input_encoded[:n]
ds_test = input_encoded[n:]

# Sampling dataset, returns B * T tensor of input and output where B = Batch size and T = Block size
def udfGetBatch(split: str):
    if split.lower() == 'train':
        ds = ds_train
    else:
        ds = ds_test

    ix = torch.randint(len(ds) - vBlockSize, (vBatchSize,))

    # why not torch.tensor? -> ds is a trensor, so we use stack to combine all tensors into one .tensor accepts lists input
    x = torch.stack([ds[i:i+vBlockSize] for i in ix])
    y = torch.stack([ds[i+1:i+vBlockSize+1] for i in ix])
    return x,y

xb, yb = udfGetBatch('train')

class BigramModel(nn.Module):

    def __init__(self, vVocabSize):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vVocabSize, vVocabSize)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]         # Only last character matters in bigram
            prob = F.softmax(logits)
            pred = torch.multinomial(prob, num_samples = 1)
            idx = torch.cat((idx, pred), dim= 1)
        return idx

bm = BigramModel(vVocabSize)

# # Test code
# logits, loss = bm(xb, yb)
# print(logits)
# print(loss)
# print(decode(bm.generate(idx = torch.zeros((1,1), dtype = torch.long), max_tokens = 100)[0].tolist()))

# Training the bigram model

optimizer = torch.optim.AdamW(bm.parameters(),lr=1e-3)

for steps in range(vTrainingIterations+1):
    x, y = udfGetBatch('train')
    logits, loss = bm(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps%1000 == 0:
        print(f'{steps} : {loss.item()}')

# Post training output
print(decode(bm.generate(idx = torch.zeros((1,1), dtype = torch.long), max_tokens = 500)[0].tolist()))
