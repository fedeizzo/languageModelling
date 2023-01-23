import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EmbeddingDropout(torch.nn.Embedding):
    """
    Modified version of Embedding Dropout proposed in
    https://github.com/carpedm20/ENAS-pytorch/blob/0468b8c4ddcf540c9ed6f80c27289792ff9118c9/models/shared_rnn.py#L51

    Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.
    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).
    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=0,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        dropout=0.1,
        scale=None,
    ):
        """Embedding constructor.
        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.
        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(
            self,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), (
            "Dropout must be >= 0.0 " "and < 1.0"
        )
        self.scale = scale

    def forward(self, inputs):  # pylint:disable=arguments-differ
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(
            inputs,
            masked_weight,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
def _weight_drop(module, weights, dropout):
    """
    Modified version of the WeightDrop implementation from torchnlp.
    Weights must be converted back to nn.Parameter before assignment.

    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + "_raw", nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + "_raw")
            # FIX: convert weights back to nn.Paramter before assignment
            w = nn.Parameter(
                nn.functional.dropout(raw_w, p=dropout, training=module.training)
            )
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, "forward", forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    `__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward
