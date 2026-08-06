import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Two consecutive convolutional blocks used throughout the U-Net architecture.

    Each block consists of a 2D convolution, batch normalization,
    and ReLU activation. The two blocks preserve the spatial
    dimensions of the input feature map while increasing the
    representational capacity of the network.

    Attributes
    ----------
    conv : nn.Sequential
        Sequential container consisting of two convolutional
        blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Initialize the double convolution block.

        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
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
        Perform forward propagation through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map after two convolutional blocks.
        """

        return self.conv(x)


class UNET(nn.Module):
    """"
    Standard U-Net architecture for binary image segmentation.

    The network consists of an encoder-decoder structure with skip
    connections. The encoder progressively extracts high-level
    features, while the decoder reconstructs the segmentation mask
    using the encoder feature maps through skip connections.

    Attributes
    ----------
    downs : nn.ModuleList
        Encoder blocks.
    ups : nn.ModuleList
        Decoder blocks, consisting of transposed convolutions
        followed by DoubleConv blocks.
    pool : nn.MaxPool2d
        Max-pooling layer used for downsampling.
    bottleneck : DoubleConv
        Double convolution block at the network bottleneck.
    final_conv : nn.Conv2d
        Final 1×1 convolution producing the segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ) -> None:
        """
        Initialize the U-Net model.

        Parameters
        ----------
        in_channels : int, default=1
            Number of input image channels.
        out_channels : int, default=1
            Number of output segmentation channels.
            For binary segmentation, this should be 1.
        features : tuple[int, ...], default=(64, 128, 256, 512)
            Number of feature channels used at each encoder level.
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
        Perform forward propagation through the U-Net.

        The input image is processed by the encoder, bottleneck,
        and decoder. Skip connections are used to concatenate
        encoder feature maps with decoder feature maps at the
        corresponding resolution.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Segmentation logits of shape
            (N, out_channels, H, W).
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
    Verify that the U-Net preserves the input spatial dimensions.

    A random input tensor is passed through the network, and an
    assertion is performed to ensure that the output tensor has
    the same shape as the input tensor.
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