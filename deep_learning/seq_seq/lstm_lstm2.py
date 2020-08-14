"""
博客连接:https://blog.csdn.net/weiwei9363/article/details/79464789
keras例子：https://github.com/keras-team/keras/tree/master/examples

将句子转换为3个numpy arrays, encoder_input_data, decoder_input_data, decoder_target_data:
encoder_input_data 是一个 3D 数组，大小为 (num_pairs, max_english_sentence_length, num_english_characters)，包含英语句子的one-hot向量
decoder_input_data 是一个 3D 数组，大小为 (num_pairs, max_fench_sentence_length, num_french_characters) 包含法语句子的one-hot向量
decoder_target_data 与 decoder_input_data 相同，但是有一个时间的偏差。 decoder_target_data[:, t, :] 与decoder_input_data[:, t+1, :]相同
训练一个基于LSTM的Seq2Seq模型，在给定 encoder_input_data和decoder_input_data时，预测 decoder_target_data，我们的模型利用了teacher forcing
解码一些语言用来验证模型事有效的
"""

from corpus import chinese_to_english_path


def load_seq():
    x, y = [], []
    with open(chinese_to_english_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            print(line)


if __name__ == '__main__':
    load_seq()
