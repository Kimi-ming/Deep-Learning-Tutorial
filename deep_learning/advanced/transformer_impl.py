# -*- coding: utf-8 -*-
"""
Transformer 简化 Encoder-Decoder 演示
"""


def encoder_decoder_demo():
    """
    简化的 Encoder-Decoder 演示，输出流程说明。
    """
    print("=== Transformer Encoder-Decoder 简化演示 ===")
    steps = [
        "编码器：嵌入 + 位置编码 -> 多头自注意力 -> 前馈网络",
        "解码器：嵌入 + 位置编码 -> Masked 自注意力 -> 编码器-解码器注意力 -> 前馈网络",
        "输出：线性层 + Softmax 得到目标序列概率",
    ]
    for s in steps:
        print("•", s)


__all__ = ["encoder_decoder_demo"]
