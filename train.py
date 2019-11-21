from util import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__ == '__main__':
    # 导入参数
    config = Config()

    # 导入训练机和测试集，默认使用ag_news
    train_file = './data/ag_news_train.csv'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = './data/ag_news_test.csv'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]

    # 导入预训练的word embedding
    w2v_file = '../data/glove.840B.300d.txt'

    # 装载数据集
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)

    # 指定网络模型
    model = CRAN(len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # 确定优化方法
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.add_optimizer(optimizer)

    # 确定损失函数
    NLLLoss = nn.NLLLoss()
    model.add_loss_op(NLLLoss)

    # 开始训练
    train_losses = []
    val_accuracies = []
    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    # 训练完毕
    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)
    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))