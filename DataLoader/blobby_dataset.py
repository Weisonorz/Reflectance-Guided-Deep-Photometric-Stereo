import torch 
import torch.nn as nn 
from torchvision import transforms
from PIL import Image
import openexr_numpy
import os 
import json 
import re


class BlobbyDataset(torch.utils.data.Dataset): 
    def __init__(self, image_dir, normal_dir, info_json_path, transform = None): 
        """
        Args: 
            image_dir (path-like): path to the directory consisting of the rendered images
            normal_dir (path-like): path to the directory consisting of the groud truth normal maps 
            info_json_path (path-like): path to the json of the information to each image 
        """
        self.image_names = sorted(os.listdir(image_dir))  
        self.normals = sorted(os.listdir(normal_dir)) 
        self.image_info = json.loads(info_json_path) 
        self.image_dir = image_dir 
        self.normal_dir = normal_dir 
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
    
    def __getitem__(self, index):

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
            angles.append(self.image_info[os.path.splitext(filename)[0]]['light_position'])


        images = torch.stack(images) 
        normal = torch.Tensor(openexr_numpy.imread(normal_file)) 
        bsdf_latent = np.load()


        return images, normal, 


#Testing
if __name__ == '__main__':
    dataset = BlobbyDataset()