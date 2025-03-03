import torch
import torch.nn as nn
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    ViTImageProcessorFast,
)


class VITEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        output_activation=None,
        model_name="google/vit-base-patch16-224",
        pool_mode="mean",
    ):
        """
        Args:
            input_dim: Tuple of (channels, height, width)
            latent_dim: Output dimension of the encoder
            output_activation: Optional activation to apply to the output
            model_name: Name of the pretrained ViT model to use
            pool_mode: How to pool features from the transformer. One of ['mean', 'max', 'attention', 'linear']
        """
        super().__init__()

        # Initialize image processor and ViT model
        self.image_processor = ViTImageProcessorFast.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.vit_model = ViTForImageClassification.from_pretrained(
            model_name, attn_implementation="sdpa", torch_dtype=torch.float16
        )
        self.vit_model.eval()  # Set to eval mode since we're using pretrained weights

        # Feature pooling setup
        self.pool_mode = pool_mode
        hidden_size = 768  # Base ViT hidden dimension

        if pool_mode == "mean":
            self.feat_pool = torch.mean
        elif pool_mode == "max":
            self.feat_pool = torch.max
        elif pool_mode == "attention":
            self.feat_pool = nn.Linear(hidden_size, 1)
        elif pool_mode == "linear":
            self.feat_pool = nn.Linear(hidden_size * 16, hidden_size)
        else:
            raise ValueError(f"Invalid feature pool mode: {pool_mode}")

        # Output projection if needed
        self.output_proj = None
        if (
            latent_dim != hidden_size * 2
        ):  # *2 because we concatenate cls token and pooled features
            self.output_proj = nn.Linear(hidden_size * 2, latent_dim)

        # Output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            latent: Encoded features of shape (batch_size, latent_dim)
        """

        # Process image through ViT preprocessor
        image_processed = self.image_processor(x, do_rescale=False)

        # Get hidden states from ViT
        outputs = self.vit_model(**image_processed, output_hidden_states=True)
        cls_features = outputs.hidden_states[-1][:, 0, :].float()  # CLS token features
        local_features = outputs.hidden_states[-1][
            :, 1:, :
        ].float()  # Local patch features

        # Pool local features based on specified mode
        if self.pool_mode == "attention":
            weights = torch.softmax(self.feat_pool(local_features), dim=1)
            local_features_pooled = torch.sum(weights * local_features, dim=1)
        elif self.pool_mode == "linear":
            local_features_pooled = self.feat_pool(
                local_features.reshape(local_features.shape[0], -1)
            )
        elif self.pool_mode == "mean":
            local_features_pooled = torch.mean(local_features, dim=1)
        elif self.pool_mode == "max":
            local_features_pooled = torch.max(local_features, dim=1)[0]

        # Combine CLS token and pooled features
        latent = torch.cat([cls_features, local_features_pooled], dim=1)

        # Project to desired dimension if needed
        if self.output_proj is not None:
            latent = self.output_proj(latent)

        # Apply output activation
        latent = self.output_activation(latent)

        return latent
