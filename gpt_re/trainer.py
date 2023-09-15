import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import math

from model import GPT, GPTPretrain


seq_len = 512
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20


def make_data(datas):
    train_datas =[]
    for data in datas:
        data=data.strip()
        train_data = ['<bos>']+[i if i!='\t' else "<sep>" for i in data]+['<eos>']
        train_datas.append(train_data)

    return train_datas


class MyDataset(Dataset):
    def __init__(self,datas, q_len=256, pad_id=0):
        self.datas = datas
        self.q_len = q_len  # 输入长度
        self.pad_id = pad_id
    

    def __getitem__(self, index):
        datas = self.datas
        data = datas[index]
        data_len = len(data)
        return {"data":data, "data_len": data_len}
    

    def __len__(self):
        return len(datas)


    def padding_batch(self, batch):
        for d in batch:
            if len(d["data"])>=seq_len:
                d["data"] = d["data"][:512]
            else:
                d["data"].extend([self.pad_id] * (self.q_len-d["data_len"]))
        
        inputs = torch.tensor([d["data"] for d in batch], dtype=torch.long)
        return inputs


def train_step(model, data_loader, criterion, optimizer, scheduler, epoch):
    losses = 0
    model.train()
    total_loss = 0

    for batch, inputs in enumerate(data_loader):
        # |input| : (batch_size, seq_len)
        X = inputs[:, :-1]
        y = inputs[:, 1:]
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X)
        # |output| : (batch_size, seq_len, vocab_size)
        loss = criterion(output.contiguous().view(-1, model.vocab_size), y.contiguous().view(-1))
        losses += loss.item()
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if batch % 100 == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = losses / 200
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(data_loader):5d} batches | '
                  f'lr {lr:05.5f} |'
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            losses = 0
    
    return total_loss / len(data_loader)
    

def train(model, data_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.95)

    best_loss = float('inf')
    for epoch in range(epochs):
        step_loss = train_step(model, data_loader, criterion, optimizer, scheduler, epoch)
        scheduler.step()
        if step_loss < best_loss:
            best_loss = step_loss
            torch.save(model.state_dict(), "./GPT2.pt")
        


if __name__ == "__main__":
    with open("./datas/dataset.txt", "r", encoding="utf-8") as f:
        datas = f.readlines()


    train_datas = make_data(datas)
    dict_datas = json.load(open('./datas/dict_datas.json', 'r'))
    word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
    train_datas_nums = [[word2id[word] for word in line] for line in train_datas]  # 字符转数字
    dataset = MyDataset(train_datas_nums, q_len=seq_len)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.padding_batch)

    gpt = GPT(vocab_size=len(dict_datas["word2id"]), seq_len=seq_len)
    
    # 输入：(batch_size, seq_len), 输出：(batch_size, seq_len, vocab_size)
    gpt_pretrain = GPTPretrain(gpt)
    model = gpt_pretrain.to(device)

    train(model, data_loader, epochs)

