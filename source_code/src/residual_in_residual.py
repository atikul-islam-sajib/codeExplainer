import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("/src/")

from dense_block import DenseBlock


class ResidualInResidual(nn.Module):
    def __init__(self, in_channels=64, res_scale=0.2):
        super(ResidualInResidual, self).__init__()

        self.in_channels = in_channels
        self.res_scale = res_scale

        self.denseblock1 = DenseBlock(
            in_channels=self.in_channels, out_channels=self.in_channels
        )
        self.denseblock2 = DenseBlock(
            in_channels=self.in_channels, out_channels=self.in_channels
        )
        self.denseblock3 = DenseBlock(
            in_channels=self.in_channels, out_channels=self.in_channels
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            output1 = self.denseblock1(x)
            input2 = output1 + x

            output2 = self.denseblock2(input2)
            input3 = output2 + input2

            output = self.denseblock3(input3)
            output = torch.mul(output, self.res_scale) + input3

            return output + x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual In Residual".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Define in_channels for ResidualInResidual".capitalize(),
    )

    args = parser.parse_args()

    residual_in_residual = ResidualInResidual(in_channels=args.in_channels)

    print(residual_in_residual)
