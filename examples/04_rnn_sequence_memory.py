# -*- coding: utf-8 -*-
"""
示例 04: RNN 简单序列记忆
"""

from deep_learning.architectures import SimpleRNN


def main():
    rnn = SimpleRNN(input_size=2, hidden_size=3, output_size=1, learning_rate=0.05)

    sequence = [[1, 0], [0, 1], [1, 1], [0, 0]]
    outputs = rnn.forward(sequence)
    print("RNN outputs:", outputs)


if __name__ == "__main__":
    main()
