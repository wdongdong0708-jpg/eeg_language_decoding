"""EEG encoders that do not receive shortcut metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Type

import torch
from torch import nn


class DilatedConvBlock(nn.Module):
    """One same-length 1D dilated convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        activation: Type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,
        dropout: float = 0.0,
        dropout_input: float = 0.0,
        residual: bool = False,
        activate: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size and dilation must be positive.")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= dropout_input < 1.0:
            raise ValueError("dropout probabilities must be in [0, 1).")

        padding = kernel_size // 2 * dilation
        layers: list[nn.Module] = []
        if dropout_input:
            layers.append(nn.Dropout(dropout_input))
        layers.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        if activate:
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.use_residual = residual and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.use_residual:
            y = y + x
        return y


class DilatedSimpleConv(nn.Module):
    """Same-length dilated Conv1d EEG encoder.

    Input shape: ``[batch, input_channels, time]``.
    Output shape: ``[batch, output_channels, time]``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        hidden_channels: int = 320,
        depth: int = 4,
        kernel_size: int = 5,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: Optional[int] = None,
        dropout: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = False,
        residual: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
        activation_on_last: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if input_channels < 1 or output_channels < 1 or hidden_channels < 1:
            raise ValueError("channel counts must be positive.")
        if growth <= 0 or dilation_growth < 1:
            raise ValueError("growth and dilation_growth must be positive.")
        if dilation_period is not None and dilation_period < 1:
            raise ValueError("dilation_period must be positive when provided.")

        channels = self._make_channels(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            depth=depth,
            growth=growth,
        )

        blocks: list[nn.Module] = []
        dilation = 1
        for layer_idx, (chin, chout) in enumerate(
            zip(channels[:-1], channels[1:], strict=True)
        ):
            if dilation_period and layer_idx % dilation_period == 0:
                dilation = 1

            is_last = layer_idx == depth - 1
            blocks.append(
                DilatedConvBlock(
                    chin,
                    chout,
                    kernel_size,
                    dilation,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    dropout_input=dropout_input if layer_idx == 0 else 0.0,
                    residual=residual,
                    activate=activation_on_last or not is_last,
                )
            )
            dilation *= dilation_growth

        self.net = nn.Sequential(*blocks)

    @staticmethod
    def _make_channels(
        *,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        depth: int,
        growth: float,
    ) -> Sequence[int]:
        channels = [input_channels]
        channels += [
            int(round(hidden_channels * growth**index)) for index in range(depth)
        ]
        channels[-1] = output_channels
        return channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
