import torch
import torch.nn as nn
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self, input_dimension, output_dimension, nhead, nhid, nlayers, dropout=0.5
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.input_attention_layers = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(), input_dimension, 12),
                    input_dimension,
                    12,
                    activation="gelu",
                )
            ],
            norm_layer=torch.nn.LayerNorm(input_dimension),
        )
        self.upscaling_layer = nn.Linear(input_dimension, output_dimension)
        self.attention_layers = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        LinearAttention(query_dimensions=128), output_dimension, 12
                    ),
                    output_dimension,
                    12,
                    activation="gelu",
                )
            ],
            norm_layer=torch.nn.LayerNorm(output_dimension),
        )

    def forward(self, src, length_mask=None):
        src = self.input_attention_layers(src, length_mask=length_mask)
        src = self.upscaling_layer(src)
        output = self.attention_layers(src, length_mask=length_mask)
        cls_output = output[:, 0, :]  # batch, token/character, dimension

        return cls_output
