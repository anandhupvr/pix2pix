import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from generator_model import Generator
from discriminator_model import Discriminator
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize


def main(img_path, model_path):
    # Define the transformation pipeline
    transform = transforms.Compose([
        Resize((256, 256)),  # Resize the image to the desired size
        ToTensor(),  # Convert the image to a tensor
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])

    # Load the input image
    image_in = Image.open(img_path).convert("RGB")

    # Apply transformations to the input image
    input_data = transform(image_in)
    input_data = input_data.unsqueeze(0)  # Add an extra dimension

    # Load the generator model and its checkpoint
    checkpoint = torch.load(model_path)
    model = Generator(3, 64)
    model.load_state_dict(checkpoint['model_state_dic'])

    device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Generate the output image
        output_img = model(input_data)

    # Rearrange dimensions for displaying the image
    out = output_img.permute(0, 2, 3, 1)

    # Display the output image
    plt.imshow(out[0])
    plt.show()


if __name__ == "__main__":
    # Check if two arguments (image path and model path) are provided
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide two arguments: image path and model path.")
