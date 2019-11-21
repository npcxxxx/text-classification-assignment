class Config(object):
    embed_size = 300  # embedding后的维度
    word_gru_hidden = 100  # gru维度
    output_size = 4  # 分类数量
    max_epochs = 10  # 迭代次数
    max_sen_len = 300  # 输入文本最长长度
    kernel_num = 256  # CNN卷积核数量
    kernel_size_list = (3, 4, 5)  # CNN卷积核大小
    lr = 0.25   # 学习率
    batch_size = 256  # 最小批处理数量
    dropout = 0.3  # 失活率
