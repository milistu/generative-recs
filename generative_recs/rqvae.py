import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class RQVAE(nn.Module):
    """Residual-Quantized Variational AutoEncoder for Semantic ID generation."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] = [512, 256, 128],
        latent_dim: int = 32,
        num_quantizers: int = 3,
        codebook_size: int = 256,
        use_cosine_sim: bool = False,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        commitment_weight: float = 0.25,
    ):
        super().__init__()

        # Build encoder: input_dim -> hidden_dims -> latent_dim
        encoder_layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Residual quantizer
        self.quantizer = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            use_cosine_sim=use_cosine_sim,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            commitment_weight=commitment_weight,
        )

        # Build decoder: latent_dim -> reversed hidden_dims -> input_dim
        decoder_layers: list[nn.Module] = []
        dims = [latent_dim] + hidden_dims[::-1]
        for i in range(len(dims) - 1):
            decoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Cosine similarity for reconstruction loss
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latent representation.

        Returns:
            quantized: Quantized latent vectors (batch, latent_dim)
            indices: Codebook indices per level (batch, num_quantizers)
            commit_loss: Commitment loss per quantizer level (num_quantizers,)
        """
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation back to input space."""
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> quantize -> decode.

        Args:
            x: Input embeddings (batch, input_dim)

        Returns:
            x_recon: Reconstructed embeddings (batch, input_dim)
            indices: Semantic ID codewords (batch, num_quantizers)
            recon_loss: MSE reconstruction loss (scalar)
            commit_loss: Commitment loss (scalar)
        """
        z = self.encode(x)
        z_q, indices, commit_loss = self.quantize(z)
        x_recon = self.decode(z_q)

        recon_loss = 1 - self.cosine_similarity(x, x_recon).mean()

        return x_recon, indices, recon_loss, commit_loss.sum()

    @torch.no_grad()
    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Get Semantic IDs (codebook indices) for input embeddings.

        Args:
            x: Input embeddings (batch, input_dim)

        Returns:
            indices: Semantic ID codewords (batch, num_quantizers)
        """
        z = self.encode(x)
        _, indices, _ = self.quantize(z)
        return indices


class UsageTracker:
    """Tracks codebook utilization during training via normalized entropy.

    Accumulates how often each codebook entry is selected across training steps.
    Measures usage as normalized Shannon entropy per quantizer level:
    - 1.0 = perfectly uniform usage (all entries used equally)
    - 0.0 = complete collapse (everything maps to one entry)

    The paper trains until usage >= 0.8 across all levels.
    """

    def __init__(self, num_quantizers: int, codebook_size: int, device: torch.device):
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.frequencies = torch.zeros((num_quantizers, codebook_size), device=device)
        self.max_entropy = torch.log2(torch.tensor(codebook_size, dtype=torch.float32))

    def update(self, indices: torch.Tensor) -> None:
        """Update usage frequencies from a batch of quantizer indices.

        Args:
            indices: Codebook indices (batch, num_quantizers)
        """
        for level in range(self.num_quantizers):
            counts = torch.bincount(
                indices[:, level].flatten(),
                minlength=self.codebook_size,
            )
            self.frequencies[level] += counts

    def get_usage(self) -> list[float]:
        """Get normalized entropy per quantizer level.

        Returns:
            List of floats in [0, 1], one per quantizer level.
        """
        usages = []
        for level_freq in self.frequencies:
            probs = level_freq / level_freq.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs))
            usages.append((entropy / self.max_entropy).item())
        return usages

    def reset(self) -> None:
        """Reset all accumulated frequencies."""
        self.frequencies.zero_()
