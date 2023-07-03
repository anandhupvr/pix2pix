import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.img_files = os.listdir(self.input_dir)
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.img_files[idx])
        out_img_name = self.img_files[idx].replace('_sketch', '')
        output_img_path = os.path.join(self.output_dir, out_img_name)
        # image_in = read_image(input_img_path)
        # image_out = read_image(output_img_path)
        image_in = Image.open(input_img_path).convert("RGB")
        image_out = Image.open(output_img_path).convert("RGB")

        if self.transform:
            input_image = self.transform(image_in)
            output_image = self.transform(image_out)
        return input_image, output_image
