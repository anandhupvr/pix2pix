import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self, input_channels, output_channels, stride, batch_norm=True):
		super(ConvBlock, self).__init__()

		# self.conv = nn.Sequential(
		# 	nn.Conv2d(input_channels, output_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
		# 	nn.BatchNorm2d(output_channels),
		# 	nn.LeakyReLU(0.2)
		# )
		self.batch_norm = batch_norm

		self.conv = nn.Conv2d(input_channels, output_channels, 4, stride, 1, bias=False, padding_mode="reflect")
		self.bn = nn.LeakyReLU(0.2)

	def forward(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, input_channels):
		super(Discriminator, self).__init__()
		self.conv1 = ConvBlock(input_channels * 2, 64, 2, False)
		self.conv2 = ConvBlock(64, 128, 2)
		self.conv3 = ConvBlock(128, 256, 2)
		self.conv4 = ConvBlock(256, 512, 1)
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect")

	def forward(self, x, y):

		x = torch.cat([x, y], dim=1)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		return x

def test():
	x = torch.randn((1, 3, 256, 256))
	y = torch.randn((1, 3, 256, 256))
	model = Discriminator(input_channels=3)
	preds = model(x, y)
	print(model)
	print(preds.shape)


if __name__ == "__main__":
	test()
