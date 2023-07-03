import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize
from data_loader import CustomImageDataset
from generator_model import Generator
from discriminator_model import Discriminator
import torchvision
import torch.nn as nn

input_dir = ""
output_dir = ""


transform = transforms.Compose([
    Resize((256, 256)),  # Resize the image to the desired size
    ToTensor(),  # Convert the image to a tensor
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

def train():

	# Create the dataset and data loader


	# Create the dataset and data loader
	dataset = CustomImageDataset(input_dir, output_dir, transform=transform)

	# Split dataset into train and test subsets
	train_ratio = 0.8
	dataset_size = dataset.__len__()
	train_size = int(train_ratio * dataset_size)
	test_size = dataset_size - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


	batch_size = 1
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	input_channels = 3
	output_channels = 64
	discriminator = Discriminator(input_channels).to(device)
	generator = Generator(input_channels, output_channels).to(device)

	opt_disc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999),)
	opt_gen = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

	BCE = nn.BCEWithLogitsLoss()
	L1_LOSS = nn.L1Loss()


	# Training loop
	num_epochs = 100

	generator.train()
	discriminator.train()


	for epoch in range(num_epochs):
		running_loss = 0.0
		for batch_idx, (input_img, label_img) in enumerate(train_loader):
			input_img = input_img.to(device)
			label_img = label_img.to(device)


			opt_disc.zero_grad()
			opt_gen.zero_grad()

			# Forward pass
			y_fake = generator(input_img)
			# train discriminator with real data
			d_real = discriminator(input_img, label_img)
			d_real_loss = BCE(d_real, torch.ones_like(d_real))



			# Train discriminator with fake data
			d_fake = discriminator(input_img, y_fake.detach())
			d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))

			d_loss = (d_real_loss + d_fake_loss) / 2

			d_loss.backward()
			opt_disc.step()


			# Train generator 
			d_fake = discriminator(input_img, y_fake)
			g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
			l1 = L1_LOSS(y_fake, label_img) * 100 # lambda
			g_loss = g_fake_loss + l1

			g_loss.backward()
			opt_gen.step()


			if batch_idx % 1000 == 0:  # Print training progress every 1000 batches
	
				print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
					f'disc Loss: {d_loss:.4f}, '
					f'gen Loss: {g_loss:.4f}',)

		if epoch+1 in [25, 50, 75, 100]:
			gen_checkpoint = {
						'epoch': epoch + 1,
						'model_state_dic' : generator.state_dict(),
						'optimizer_state_dict' : opt_gen.state_dict(),
						'loss' : d_loss
			}
			torch.save(gen_checkpoint, f"gen_checkpoint_{epoch+1}.pth")

			disc_checkpoint = {
								'epoch' : epoch + 1,
								'model_state_dic' : discriminator.state_dict(),
								'optimizer_state_dict' : opt_disc.state_dict(),
								'loss' : g_loss
			}
			torch.save(disc_checkpoint, f"disk_checkpoint_{epoch+1}.pth")
	print("Training Finished")




if __name__ == "__main__":
	train()