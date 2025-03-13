import torch 
import torch.nn as nn 
import numpy as np 
from torchvision import transforms
from utils import lightpos2angle
from PIL import Image
import openexr_numpy
import yaml 
import os 
import json 
import re


class BlobbyDataset(torch.utils.data.Dataset): 
    def __init__(self, image_dir, normal_dir, latent_vectors_dir ,info_json_path, transform = None): 
        """
        Args: 
            image_dir (path-like): path to the directory consisting of the rendered images
            normal_dir (path-like): path to the directory consisting of the groud truth normal maps 
            normal_dir (path-like): path to the directory consisting of latent vectors of brdfs 
            info_json_path (path-like): path to the json of the information to each image 
        """
        self.image_names = sorted(os.listdir(image_dir))  
        self.normals = sorted(os.listdir(normal_dir)) 
        with open(info_json_path, "r") as f:
            self.image_info  = json.load(f)
            
        self.image_dir = image_dir 
        self.normal_dir = normal_dir 
        self.latent_vectors_dir = latent_vectors_dir
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
        

        self.image_groups = self.group_images()

    def group_images(self) -> dict[int, list[str]]: 
        """
        Groups images based on the 'n' prefix in the filename.
        Returns a dictionary of filenames 
        """
            
        groups = {}
            
        for filename in self.image_names:
            if filename.endswith(".png"):  # Check if it's a PNG file
                match = re.match(r"(\d+)_(\d+)\.png", filename)  # Use regex
                if match:
                    n = int(match.group(1))  
                    m = int(match.group(2)) 
                    if n not in groups:
                        groups[n] = []  # Initialize a list for this 'n'
                    groups[n].append(filename)  # Add the filename to the list
                else:
                    print(f"Warning: Skipping file {filename} - invalid format.")
        return groups

    def __len__(self):
        return len(self.image_groups) 
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            images (torch.Tensor): [num_lights_per_sample, 3, H, W]
            bsdf_latent (torch.Tensor): A latent vector of the bsdf of the corrsponding sample, a [14] shape tensor
            angles (torch.Tensor): A tensor of shape [num_lights_per_sample, 3] representing the unit vector from light to (0,0,0)
            normal (torch.Tensor): [H, W, 3] 
        """

        n_values = list(self.image_groups.keys())
        n = n_values[index]

        # Get the list of filenames for this 'n'
        filenames = self.image_groups[n]
        normal_file = os.path.join(self.normal_dir, f'{str(n)}_1.exr')

        # Load the images.
        images = []
        angles = []
        bsdf_name = self.image_info[os.path.splitext(filenames[0])[0]]['bsdf_name']
        for filename in filenames:
            img_path = os.path.join(self.image_dir, filename)
            image = Image.open(img_path).convert("RGB") # Ensure consistent format
            if self.transform:
                image = self.transform(image)
            images.append(image)
            angles.append(lightpos2angle(self.image_info[os.path.splitext(filename)[0]]['light_position']))


        images = torch.stack(images) 
        normal = torch.Tensor(openexr_numpy.imread(normal_file)) 
        bsdf_latent_name = f'{os.path.splitext(bsdf_name)[0]}_latentVector.npy'
        bsdf_latent = np.load(os.path.join(self.latent_vectors_dir, bsdf_latent_name)).reshape(-1) 

        return images, torch.Tensor(bsdf_latent), torch.stack(angles), normal


#Testing
if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    dataset = BlobbyDataset(image_dir = config['paths']['image_dir'],
                             normal_dir = config['paths']['normal_dir'],
                             latent_vectors_dir = config['paths']['latent_vectors_dir'], 
                             info_json_path = config['paths']['info_json_path']
                             )

