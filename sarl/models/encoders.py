"""Encoder architectures for representation learning."""
import torch
import torch.nn as nn
from typing import List, Optional


class MLPEncoder(nn.Module):
    """MLP encoder for tabular data."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], output_dim: int = 64,
                 activation: str = 'relu', dropout: float = 0.0, batch_norm: bool = True):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        # Create activation function (don't create all activations, just the one we need)
        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(0.2),
            'elu': nn.ELU,
            'gelu': nn.GELU
        }
        activation_fn = activation_map.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvEncoder(nn.Module):
    """CNN encoder for small images."""
    def __init__(self, in_channels: int = 3, output_dim: int = 128, base_channels: int = 32, num_layers: int = 4):
        super().__init__()
        self.output_dim = output_dim
        layers = []
        prev_channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([nn.Conv2d(prev_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()])
            prev_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate feature dimension based on architecture
        # After convolutions and pooling, we have base_channels * 2^(num_layers-1) channels
        conv_output_channels = base_channels * (2 ** (num_layers - 1))
        self.fc = nn.Linear(conv_output_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional encoder.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)

        Returns:
            Encoded representation of shape (batch, output_dim)
        """
        features = self.pool(self.conv(x)).view(x.size(0), -1)
        return self.fc(features)


class ResNetEncoder(nn.Module):
    """ResNet encoder for images."""
    def __init__(self, output_dim: int = 128, pretrained: bool = True, backbone: str = 'resnet18', freeze_backbone: bool = False):
        super().__init__()
        self.output_dim = output_dim
        try:
            import torchvision.models as models
            weights = 'IMAGENET1K_V1' if pretrained else None
            if backbone == 'resnet18':
                self.backbone = models.resnet18(weights=weights)
                backbone_dim = 512
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(weights=weights)
                backbone_dim = 2048
            else:
                self.backbone = models.resnet18(weights=weights)
                backbone_dim = 512
            self.backbone.fc = nn.Identity()
        except ImportError:
            self.backbone = ConvEncoder(in_channels=3, output_dim=512)
            backbone_dim = 512
        self.projector = nn.Sequential(nn.Linear(backbone_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


def create_encoder(input_type: str, input_dim: Optional[int] = None, output_dim: int = 64, **kwargs) -> nn.Module:
    if input_type == 'tabular':
        return MLPEncoder(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif input_type == 'image_small':
        return ConvEncoder(output_dim=output_dim, **kwargs)
    elif input_type == 'image_large':
        return ResNetEncoder(output_dim=output_dim, **kwargs)
    raise ValueError(f"Unknown input_type: {input_type}")
