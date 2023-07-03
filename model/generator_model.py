import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, out_channels, use_dropout=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(input_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x) if self.use_dropout else x


class ConvTBlock(nn.Module):
    def __init__(self, input_channels, out_channels, use_dropout=False):
        super(ConvTBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(input_channels, out_channels, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.convT(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(Generator, self).__init__()

		# Encoder
		
		self.encoder1_conv = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 4, 2, 1, padding_mode="reflect"),
			nn.LeakyReLU(0.2),
		)

		self.encoder2_conv = ConvBlock(output_channels, output_channels * 2, use_dropout=False)
		self.encoder3_conv = ConvBlock(output_channels * 2, output_channels * 4, use_dropout=False)
		self.encoder4_conv = ConvBlock(output_channels * 4, output_channels * 8, use_dropout=False)
		self.encoder5_conv = ConvBlock(output_channels * 8, output_channels * 8, use_dropout=False)
		self.encoder6_conv = ConvBlock(output_channels * 8, output_channels * 8, use_dropout=False)
		self.encoder7_conv = ConvBlock(output_channels * 8, output_channels * 8, use_dropout=False)
		
		self.bottleneck = nn.Sequential(
			nn.Conv2d(output_channels * 8, output_channels * 8, 4, 2, 1), nn.ReLU()
			)
		
		self.decoder1_convt = ConvTBlock(output_channels * 8, output_channels * 8, use_dropout=True)
		self.decoder2_convt = ConvTBlock(output_channels * 8 * 2, output_channels * 8, use_dropout=True)
		self.decoder3_convt = ConvTBlock(output_channels * 8 * 2, output_channels * 8, use_dropout=True)
		self.decoder4_convt = ConvTBlock(output_channels * 8 * 2, output_channels * 8, use_dropout=False)
		self.decoder5_convt = ConvTBlock(output_channels * 8 * 2, output_channels * 4, use_dropout=False)
		self.decoder6_convt = ConvTBlock(output_channels * 4 * 2, output_channels * 2, use_dropout=False)
		self.decoder7_convt = ConvTBlock(output_channels * 2 * 2, output_channels , use_dropout=False)
		self.decoder_out = nn.Sequential(
			nn.ConvTranspose2d(output_channels * 2, input_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh(),
			)


	def forward(self, x):
		e1 = self.encoder1_conv(x)
		e2 = self.encoder2_conv(e1)
		e3 = self.encoder3_conv(e2)
		e4 = self.encoder4_conv(e3)
		e5 = self.encoder5_conv(e4)
		e6 = self.encoder6_conv(e5)
		e7 = self.encoder7_conv(e6)
		
		bottleneck = self.bottleneck(e7)

		d1 = self.decoder1_convt(bottleneck)
		d2 = self.decoder2_convt(torch.cat([d1, e7], 1))
		d3 = self.decoder3_convt(torch.cat([d2, e6], 1))
		d4 = self.decoder4_convt(torch.cat([d3, e5], 1))
		d5 = self.decoder5_convt(torch.cat([d4, e4], 1))
		d6 = self.decoder6_convt(torch.cat([d5, e3], 1))
		d7 = self.decoder7_convt(torch.cat([d6, e2], 1))
		decoded = self.decoder_out(torch.cat([d7, e1], 1))

		return decoded

		return e1

def test():
	x = torch.randn((1, 3, 256, 256))
	model = Generator(3, 64)
	out = model(x)
	print(out.shape)

if __name__ == "__main__":
	test()