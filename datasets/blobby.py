import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch 
import torch.nn as nn 
import numpy as np 
from torchvision import transforms
from .custom_transforms import RandomCropPair, PixelNormalize
from utils import lightpos2angle
from PIL import Image
import openexr_numpy
from tqdm import tqdm
import yaml 
import json 
import re


class BlobbyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, normal_dir, latent_vectors_dir, info_json_path, config, transform=None):
        """
        Args:
            image_dir (path-like): path to the directory consisting of the rendered images
            normal_dir (path-like): path to the directory consisting of the groud truth normal maps
            normal_dir (path-like): path to the directory consisting of latent vectors of brdfs
            info_json_path (path-like): path to the json of the information to each image
            config: The config file loaded
        """
        self.image_dir = image_dir
        self.normal_dir = normal_dir
        self.latent_vectors_dir = latent_vectors_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.ToTensor()  # Convert to tensor
        ])
        self.config = config 
        with open(info_json_path, "r") as f:
            self.image_info = json.load(f)

        self.image_groups = self.group_images()
        self.preloaded_data = self.preload_data() 

    def group_images(self) -> dict[int, list[str]]:
        """
        Groups images based on the 'n' prefix in the filename.
        Returns a dictionary of filenames
        """

        groups = {}

        for filename in sorted(os.listdir(self.image_dir)):  # Iterate through image directory
            if filename.endswith(".png"):
                match = re.match(r"(\d+)_(\d+)\.png", filename)
                if match:
                    n = int(match.group(1))
                    if n not in groups:
                        groups[n] = []
                    groups[n].append(filename)
                else:
                    print(f"Warning: Skipping file {filename} - invalid format.")
        return groups
    
    def preload_data(self):
        """
        Preloads normal maps, BSDF latent vectors, and image file paths.
        """
        preloaded = {}
        for n, filenames in tqdm(self.image_groups.items(), desc = 'Loading BlobbyDataset'):
            # Load normal map
            normal_file = os.path.join(self.normal_dir, f'{n}_1.exr')
            normal = openexr_numpy.imread(normal_file)
            normal = self.normal_transform(normal)  # Convert to tensor

            # Load BSDF latent vector
            first_filename_key = os.path.splitext(filenames[0])[0]
            bsdf_name = self.image_info[first_filename_key]['bsdf_name']
            bsdf_latent_name = f'{os.path.splitext(bsdf_name)[0]}_latentVector.npy'
            bsdf_latent = np.load(os.path.join(self.latent_vectors_dir, bsdf_latent_name)).reshape(-1)
            bsdf_latent = torch.Tensor(bsdf_latent)  # convert to tensor

            # Store preloaded data and image paths + light angles
            image_paths = []
            angles = []
            for filename in filenames:
                img_path = os.path.join(self.image_dir, filename)
                filename_key = os.path.splitext(filename)[0]
                angle = lightpos2angle(self.image_info[filename_key]['light_position'])
                image_paths.append(img_path)
                angles.append(angle)

            preloaded[n] = {
                'normal': normal,
                'bsdf_latent': bsdf_latent,
                'image_paths': image_paths,
                'angles': torch.stack(angles)  # Store angles as a tensor
            }
        return preloaded

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            images (torch.Tensor): [num_lights_per_sample * 3, H, W]
            bsdf_latent (torch.Tensor): A latent vector of the bsdf of the corrsponding sample, a [14] shape tensor
            angles (torch.Tensor): A tensor of shape [num_lights_per_sample, 3] representing the unit vector from light to (0,0,0)
            normal (torch.Tensor): [3, H, W]
        """
        n_values = list(self.image_groups.keys())
        n = n_values[index]

        # Retrieve preloaded data
        data = self.preloaded_data[n]
        normal = data['normal']
        bsdf_latent = data['bsdf_latent']
        image_paths = data['image_paths']
        angles = data['angles']

        # Load images
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.concat(images, dim=0)
        #Transforms
        if self.config['dataset']['random_crop_size'] is not None: 
            images, normal = RandomCropPair(images, normal, config['dataset']['random_crop_size']) 

        if self.config['dataset']['normalize'] is not None: 
            images = PixelNormalize(images) 
        
        

        return images, bsdf_latent, angles, normal

# Test
if __name__ == '__main__':
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)
    dataset = BlobbyDataset(image_dir=config['paths']['image_dir'],
                             normal_dir=config['paths']['normal_dir'],
                             latent_vectors_dir=config['paths']['latent_vectors_dir'],
                             info_json_path=config['paths']['info_json_path'],
                             config=config)

    # Example usage (check the shapes)
    images, bsdf_latent, angles, normal = dataset[0]
    print("Images shape:", images.shape)  # Expected: [num_lights * 3, H, W]
    print("BSDF latent shape:", bsdf_latent.shape)  # Expected: [14]
    print("Angles shape:", angles.shape)  # Expected: [num_lights, 3]
    print("Normal shape:", normal.shape)  # Expected: [3, H, W]

    # Example with DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_idx, (images, bsdf_latents, angles, normals) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Images shape:", images.shape)  # [batch_size, num_lights * 3, H, W]
        print("  BSDF latents shape:", bsdf_latents.shape) # [batch_size, 14]
        print("  Angles shape:", angles.shape)  # [batch_size, num_lights, 3]
        print("  Normals shape:", normals.shape) # [batch_size, 3, H, W]
        if batch_idx == 2: # Stop after a few batches
            break