import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks used throughout U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Initialize the double convolution block.
        """

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation.
        """

        return self.conv(x)


class UNET(nn.Module):
    """
    Standard U-Net architecture for binary image segmentation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ) -> None:
        """
        Initialize the U-Net model.
        """

        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # -------------------------
        # Encoder
        # -------------------------
        for feature in features:
            self.downs.append(
                DoubleConv(
                    in_channels,
                    feature,
                )
            )
            in_channels = feature

        # -------------------------
        # Decoder
        # -------------------------
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )

            self.ups.append(
                DoubleConv(
                    feature * 2,
                    feature,
                )
            )

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
        )

        # -------------------------
        # Output layer
        # -------------------------
        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation.
        """

        skip_connections = []

        # -------------------------
        # Encoder
        # -------------------------
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # -------------------------
        # Bottleneck
        # -------------------------
        x = self.bottleneck(x)

        skip_connections.reverse()

        # -------------------------
        # Decoder
        # -------------------------
        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)

            skip_connection = skip_connections[idx // 2]

            # Resize if dimensions do not match.
            if x.shape != skip_connection.shape:
                x = TF.resize(
                    x,
                    size=skip_connection.shape[2:],
                )

            # Concatenate encoder and decoder features.
            x = torch.cat(
                (skip_connection, x),
                dim=1,
            )

            x = self.ups[idx + 1](x)

        # -------------------------
        # Final prediction
        # -------------------------
        return self.final_conv(x)


def test() -> None:
    """
    Verify that the network preserves
    the input spatial dimensions.
    """

    images = torch.randn(
        (
            3,
            1,
            161,
            161,
        )
    )

    model = UNET(
        in_channels=1,
        out_channels=1,
    )

    predictions = model(images)

    assert predictions.shape == images.shape

    print("U-Net test passed.")


if __name__ == "__main__":
    test()