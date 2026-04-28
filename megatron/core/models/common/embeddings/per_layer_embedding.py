# megatron/core/models/common/embeddings/per_layer_embedding.py
import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class PerLayerEmbedding(MegatronModule):
    """Gemma4 Per-Layer Embeddings (PLE)"""

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        num_layers: int,
        ple_dim: int,                    # hidden_size_per_layer_input
        embed_scale: float = 1.0,
    ):
        super().__init__(config=config)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.embed_scale = embed_scale

        # Token-identity embedding: [num_layers, vocab_size, ple_dim]
        self.embed_tokens_per_layer = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=ple_dim,
            padding_idx=None,
        )
        # Context-aware projection: main_hidden -> ple_dim
        self.per_layer_model_projection = torch.nn.Linear(
            config.hidden_size, ple_dim, bias=False
        )

        self.embed_scale = torch.tensor(embed_scale, dtype=config.params_dtype)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor):
        """
        input_ids: [batch, seq_len]
        hidden_states: [batch, seq_len, hidden_size]  # current main residual
        Returns: [batch, seq_len, num_layers, ple_dim]
        """
        batch, seq_len = input_ids.shape

        # 1. Token-identity part
        token_emb = self.embed_tokens_per_layer(input_ids)  # [b, s, ple_dim]
        token_emb = token_emb.unsqueeze(2).expand(-1, -1, self.num_layers, -1)  # [b, s, L, ple_dim]

        # 2. Context-aware part
        ctx = self.per_layer_model_projection(hidden_states)          # [b, s, ple_dim]
        ctx = ctx.unsqueeze(2).expand(-1, -1, self.num_layers, -1)   # [b, s, L, ple_dim]

        # 3. Combine + scale (HF 原始做法)
        ple = (token_emb + ctx) * (0.5 ** 0.5)                       # / sqrt(2)
        ple = ple * self.embed_scale

        return ple
