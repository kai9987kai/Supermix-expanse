"""Extra-lite native-image v38 student distilled from the v37 lite checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from model_variants import GatedExpertClassifierHead
from run import ChampionNet


class NativeImageExtraLiteDecoder(nn.Module):
    """Aggressively smaller 64x64 image decoder for the extra-lite student."""

    def __init__(self, cond_dim: int = 256, image_size: int = 64, seed_hw: int = 8, seed_channels: int = 24) -> None:
        super().__init__()
        if int(image_size) != 64:
            raise ValueError("This decoder currently supports image_size=64 only.")
        self.image_size = int(image_size)
        self.seed_hw = int(seed_hw)
        self.seed_channels = int(seed_channels)

        self.seed_proj = nn.Sequential(
            nn.Linear(cond_dim, 384),
            nn.SiLU(),
            nn.Linear(384, self.seed_hw * self.seed_hw * self.seed_channels),
        )
        self.film1 = nn.Linear(cond_dim, 24 * 2)
        self.film2 = nn.Linear(cond_dim, 12 * 2)
        self.film3 = nn.Linear(cond_dim, 8 * 2)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.seed_channels, 24, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 12),
            nn.SiLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(12, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 3, kernel_size=1),
        )

    @staticmethod
    def _apply_film(h: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        scale, bias = params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + 0.14 * scale) + 0.14 * bias

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        batch = cond.shape[0]
        h = self.seed_proj(cond).view(batch, self.seed_channels, self.seed_hw, self.seed_hw)
        h = self._apply_film(self.up1(h), self.film1(cond))
        h = self._apply_film(self.up2(h), self.film2(cond))
        h = self._apply_film(self.up3(h), self.film3(cond))
        return torch.sigmoid(self.to_rgb(h))


class ChampionNetUltraExpertNativeImageExtraLite(nn.Module):
    """Extra-lite single-checkpoint text+image model with a compressed image branch."""

    def __init__(self, dropout: float = 0.1, image_size: int = 64, n_experts: int = 6, top_k: int = 2) -> None:
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(GatedExpertClassifierHead(256, 10, n_experts=n_experts, top_k=top_k, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

        self.image_size = int(image_size)
        self.image_cond_norm = nn.LayerNorm(512)
        self.image_cond_proj = nn.Sequential(
            nn.Linear(512, 320),
            nn.SiLU(),
            nn.Linear(320, 256),
        )
        self.image_mix = nn.Linear(256, 256, bias=True)
        self.image_decoder = NativeImageExtraLiteDecoder(cond_dim=256, image_size=self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def encode_image_condition(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:10]:
            h = layer(h)
        pooled_mean = h.mean(dim=1)
        pooled_max = h.amax(dim=1)
        cond = torch.cat([pooled_mean, pooled_max], dim=-1)
        cond = self.image_cond_norm(cond)
        cond = self.image_cond_proj(cond)
        cond = cond + 0.22 * torch.tanh(self.image_mix(cond))
        return cond

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_decoder(self.encode_image_condition(x))

    @torch.no_grad()
    def render_prompt(self, prompt: str, feature_mode: str = "context_mix_v4", device: torch.device | None = None) -> torch.Tensor:
        from chat_pipeline import text_to_model_input

        if device is None:
            device = next(self.parameters()).device
        inputs = text_to_model_input(prompt, feature_mode=feature_mode).to(device)
        image = self.forward_image(inputs)[0].detach().cpu().clamp(0.0, 1.0)
        return image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    data = image_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(data, mode="RGB")


def save_prompt_image(
    model: ChampionNetUltraExpertNativeImageExtraLite,
    prompt: str,
    output_path: str | Path,
    *,
    feature_mode: str = "context_mix_v4",
    device: torch.device | None = None,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = model.render_prompt(prompt, feature_mode=feature_mode, device=device)
    tensor_to_pil(image).save(out_path)
    return out_path
