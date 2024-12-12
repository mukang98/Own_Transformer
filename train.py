"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters') 
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience) #学习率调整

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):

    '''
    每个批次中的训练：
    1.优化器初始化
    2.加载数据，正向传播
    3.计算loss
    4.反向传播，计算梯度
    5.更新模型参数
    '''
    model.train()  # 启用训练模式，确保在训练过程中激活 Dropout 和 Batch Normalization
    epoch_loss = 0  # 初始化当前 epoch 的总损失

    for i, batch in enumerate(iterator):  # 遍历数据集的每一个批次
        src = batch.src  # 获取输入序列 (源序列)
        trg = batch.trg  # 获取目标序列 (目标序列)

        optimizer.zero_grad()  # 清零梯度，以便在每次反向传播时计算新的梯度
        output = model(src, trg[:, :-1])  # 将源序列和去掉最后一个时间步的目标序列传入模型，用于前向传播 (teacher forcing 技术)

        output_reshape = output.contiguous().view(-1, output.shape[-1])  # 将模型输出重塑为二维张量，以便计算每个时间步的预测损失
        trg = trg[:, 1:].contiguous().view(-1)  # 去掉目标序列的第一个时间步并重塑为一维张量，以确保与输出形状一致

        loss = criterion(output_reshape, trg)  # 计算预测输出与目标序列之间的损失
        loss.backward()  # 反向传播，计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 限制梯度的范数，防止梯度爆炸
        optimizer.step()  # 更新模型参数

        epoch_loss += loss.item()  # 累加当前批次的损失
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())  # 打印当前批次的训练进度（百分比）和损失

    return epoch_loss / len(iterator)  # 返回当前 epoch 的平均损失

def evaluate(model, iterator, criterion):
    model.eval()  # 启用评估模式，确保评估过程中不激活 Dropout 和 Batch Normalization
    epoch_loss = 0  # 初始化当前 epoch 的总损失
    batch_bleu = []  # 用于存储每个批次的 BLEU 分数

    with torch.no_grad():  # 在评估过程中不计算梯度
        for i, batch in enumerate(iterator):  # 遍历数据集的每一个批次
            src = batch.src  # 获取输入序列 (源序列)
            trg = batch.trg  # 获取目标序列 (目标序列)
            output = model(src, trg[:, :-1])  # 将源序列和去掉最后一个时间步的目标序列传入模型，进行前向传播

            output_reshape = output.contiguous().view(-1, output.shape[-1])  # 将模型输出重塑为二维张量，以便计算每个时间步的预测损失
            trg = trg[:, 1:].contiguous().view(-1)  # 去掉目标序列的第一个时间步并重塑为一维张量，以确保与输出形状一致

            loss = criterion(output_reshape, trg)  # 计算预测输出与目标序列之间的损失
            epoch_loss += loss.item()  # 累加当前批次的损失

            total_bleu = []  # 用于存储当前批次的每个样本的 BLEU 分数
            for j in range(batch_size):  # 遍历当前批次中的每个样本
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)  # 将目标序列的索引转换为单词
                    output_words = output[j].max(dim=1)[1]  # 获取预测输出中每个时间步的最大概率的单词索引
                    output_words = idx_to_word(output_words, loader.target.vocab)  # 将预测输出的索引转换为单词

                    # 计算预测序列与目标序列之间的 BLEU 分数
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())  
                    total_bleu.append(bleu)  # 将 BLEU 分数添加到列表中
                except:  # 如果某个样本计算 BLEU 分数失败，则跳过该样本
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)  # 计算当前批次的平均 BLEU 分数
            batch_bleu.append(total_bleu)  # 将当前批次的 BLEU 分数添加到列表中

    batch_bleu = sum(batch_bleu) / len(batch_bleu)  # 计算整个数据集上的平均 BLEU 分数
    return epoch_loss / len(iterator), batch_bleu  # 返回当前 epoch 的平均损失和平均 BLEU 分数

def run(total_epoch, best_loss):
    # 初始化用于存储每个 epoch 的训练损失、测试损失和 BLEU 分数的列表
    train_losses, test_losses, bleus = [], [], []

    # 遍历每个 epoch，进行训练和验证
    for step in range(total_epoch):
        start_time = time.time()  # 记录当前时间，计算本 epoch 的训练时间

        # 训练模型，返回训练损失
        train_loss = train(model, train_iter, optimizer, criterion, clip)

        # 评估模型，返回验证集的损失和 BLEU 分数
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()  # 记录结束时间

        # 如果当前 epoch 超过 warmup 步数，则调整学习率
        if step > warmup:
            scheduler.step(valid_loss)

        # 将当前 epoch 的训练损失、验证损失和 BLEU 分数添加到列表中
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        # 计算本 epoch 的训练时间（分钟和秒）
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 如果当前验证集损失小于历史最佳损失，更新最佳损失并保存模型
        if valid_loss < best_loss:
            best_loss = valid_loss  # 更新最佳损失
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))  # 保存模型

        # 将训练损失写入文件
        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))

        # 将 BLEU 分数写入文件
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))

        # 将测试损失写入文件
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        # 打印本 epoch 的训练时间、训练损失、训练困惑度、验证损失、验证困惑度和 BLEU 分数
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')



if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
