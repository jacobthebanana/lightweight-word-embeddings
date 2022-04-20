import torch
import torch.nn as nn
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        output_dimension,
        nhead,
        nlayers,
        nlayers_input=None,
        dropout=0.5,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.nlayers_input = nlayers_input
        self.num_padding_dimensions = hidden_dimension - input_dimension
        assert output_dimension >= input_dimension
        self.attention_layers = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(FullAttention(), hidden_dimension, nhead),
                    hidden_dimension,
                    nhead,
                    activation="gelu",
                    dropout=dropout,
                )
                for _ in range(nlayers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_dimension),
        )
        self.output_layer = torch.nn.Linear(hidden_dimension, output_dimension)

    def forward(self, src, length_mask=None):
        batch_size: int = src.shape[0]
        batch_width: int = src.shape[1]
        padding = torch.zeros(
            (batch_size, batch_width, self.num_padding_dimensions)
        ).to(src.device)

        src = torch.cat((src, padding), axis=2)
        output = self.attention_layers(src, length_mask=length_mask)

        cls_output = output[:, 0, :]  # batch, token/character, dimension
        cls_output = self.output_layer(cls_output)

        return cls_output


class TranslatorModel(nn.Module):
    """
    Translation layer for translating generic embeddings to GloVe embeddings.
    """

    def __init__(self, dimension, hidden_dimension):
        super(TranslatorModel, self).__init__()
        self.input_layer = torch.nn.Linear(dimension, hidden_dimension)
        self.ReLU = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(hidden_dimension, dimension)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.ReLU(x)
        return self.output_layer(x)
