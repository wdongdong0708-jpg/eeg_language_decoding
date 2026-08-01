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


class MetaAlignedConvNoSubject(nn.Module):
    """BrainMagick-style SimpleConv without subject or spatial-merger layers.

    The default topology is the Meta configuration used for speech decoding:
    an initial 1x1 projection, ten dilated Conv1d blocks, a residual connection
    whenever channel shapes match, a contextual GLU every two blocks, and the
    two-layer ``complex_out`` projection.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        initial_channels: int = 270,
        hidden_channels: int = 320,
        depth: int = 10,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        glu_every: int = 2,
        glu_context: int = 1,
        batch_norm: bool = True,
        skip: bool = True,
        n_subjects: int | None = None,
        subject_layer_init_identity: bool = False,
    ) -> None:
        super().__init__()
        if min(
            input_channels,
            output_channels,
            initial_channels,
            hidden_channels,
            depth,
            kernel_size,
            dilation_growth,
            dilation_period,
            glu_every,
        ) < 1:
            raise ValueError("Meta-aligned encoder dimensions must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        if glu_context < 0:
            raise ValueError("glu_context must be non-negative")

        self.initial = nn.Conv1d(input_channels, initial_channels, 1)
        self.subject_layer = (
            BrainMagickSubjectLayer(
                initial_channels,
                initial_channels,
                n_subjects,
                init_identity=subject_layer_init_identity,
            )
            if n_subjects is not None
            else None
        )
        self.layers = nn.ModuleList()
        self.glus = nn.ModuleList()
        self.dilations: list[int] = []
        in_channels = initial_channels
        dilation = 1
        for layer_index in range(depth):
            if layer_index % dilation_period == 0:
                dilation = 1
            padding = kernel_size // 2 * dilation
            block: list[nn.Module] = [
                nn.Conv1d(
                    in_channels,
                    hidden_channels,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                )
            ]
            if batch_norm:
                block.append(nn.BatchNorm1d(hidden_channels))
            block.append(nn.GELU())
            self.layers.append(nn.Sequential(*block))
            self.dilations.append(dilation)

            if (layer_index + 1) % glu_every == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(
                            hidden_channels,
                            2 * hidden_channels,
                            1 + 2 * glu_context,
                            padding=glu_context,
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(nn.Identity())
            in_channels = hidden_channels
            dilation *= dilation_growth

        self.skip = skip
        self.final = nn.Sequential(
            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(2 * hidden_channels, output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        return self._forward_conv_stack(x)

    def _forward_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        for layer, glu in zip(self.layers, self.glus, strict=True):
            previous = x
            x = layer(x)
            if self.skip and x.shape == previous.shape:
                x = x + previous
            x = glu(x)
        return self.final(x)


class BrainMagickSubjectLayer(nn.Module):
    """Per-subject channel mixing used by BrainMagick's ``SubjectLayers``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        *,
        init_identity: bool = False,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, n_subjects) < 1:
            raise ValueError("Subject-layer dimensions must be positive")
        self.weights = nn.Parameter(
            torch.randn(n_subjects, in_channels, out_channels)
        )
        if init_identity:
            if in_channels != out_channels:
                raise ValueError("Identity init requires equal channel dimensions")
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1.0 / in_channels**0.5

    def forward(
        self,
        x: torch.Tensor,
        subject_indices: torch.Tensor,
    ) -> torch.Tensor:
        if subject_indices.ndim != 1 or subject_indices.shape[0] != x.shape[0]:
            raise ValueError("subject_indices must have shape [batch]")
        if subject_indices.dtype != torch.long:
            raise ValueError("subject_indices must use torch.long")
        if torch.any(subject_indices < 0) or torch.any(
            subject_indices >= self.weights.shape[0]
        ):
            raise ValueError("subject index is out of range")
        _, in_channels, out_channels = self.weights.shape
        weights = self.weights.gather(
            0,
            subject_indices.view(-1, 1, 1).expand(
                -1, in_channels, out_channels
            ),
        )
        return torch.einsum("bct,bcd->bdt", x, weights)


class MetaAlignedConvWithSubject(MetaAlignedConvNoSubject):
    """Meta-aligned convolutional encoder with a per-subject 1x1 layer."""

    requires_subject_indices = True

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        n_subjects: int,
        subject_layer_init_identity: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            n_subjects=n_subjects,
            subject_layer_init_identity=subject_layer_init_identity,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.initial(x)
        assert self.subject_layer is not None
        x = self.subject_layer(x, subject_indices)
        return self._forward_conv_stack(x)


def build_eeg_encoder(
    config: dict[str, object],
    *,
    input_channels: int,
    output_channels: int,
    n_subjects: int | None = None,
) -> nn.Module:
    """Build a configured EEG encoder while retaining old checkpoint support."""

    name = str(config.get("name", "dilated_simple_conv"))
    if name == "dilated_simple_conv":
        return DilatedSimpleConv(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=int(config["hidden_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            growth=float(config["growth"]),
            dilation_growth=int(config["dilation_growth"]),
            dilation_period=config.get("dilation_period"),
            dropout=float(config["dropout"]),
            dropout_input=float(config["dropout_input"]),
            batch_norm=bool(config["batch_norm"]),
            residual=bool(config["residual"]),
            activation_on_last=bool(config["activation_on_last"]),
        )
    if name == "meta_aligned_conv_no_subject":
        return MetaAlignedConvNoSubject(
            input_channels=input_channels,
            output_channels=output_channels,
            initial_channels=int(config["initial_channels"]),
            hidden_channels=int(config["hidden_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            dilation_growth=int(config["dilation_growth"]),
            dilation_period=int(config["dilation_period"]),
            glu_every=int(config["glu_every"]),
            glu_context=int(config["glu_context"]),
            batch_norm=bool(config["batch_norm"]),
            skip=bool(config["skip"]),
        )
    if name == "meta_aligned_conv_subject":
        if n_subjects is None:
            raise ValueError("Subject-layer encoder requires n_subjects")
        return MetaAlignedConvWithSubject(
            input_channels=input_channels,
            output_channels=output_channels,
            n_subjects=n_subjects,
            subject_layer_init_identity=bool(
                config.get("subject_layer_init_identity", False)
            ),
            initial_channels=int(config["initial_channels"]),
            hidden_channels=int(config["hidden_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            dilation_growth=int(config["dilation_growth"]),
            dilation_period=int(config["dilation_period"]),
            glu_every=int(config["glu_every"]),
            glu_context=int(config["glu_context"]),
            batch_norm=bool(config["batch_norm"]),
            skip=bool(config["skip"]),
        )
    raise ValueError(f"Unknown EEG encoder: {name}")
