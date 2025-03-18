import torch 
import torch.nn as nn 
import yaml
from tqdm import tqdm
from datasets.dataloader import get_loader 
from models import PS_FCN
from utils import meanAngularError, broadcast_angles, get_mask
from datasets.blobby import BlobbyDataset


@torch.no_grad
def eval(model, val_loader, device): 
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    mae = meanAngularError()
    num_samples = 0
    total_error = 0.
    for i, (images, bsdfs, angles, normals) in enumerate(val_loader):
        images = images.to(device) 
        bsdfs = bsdfs.to(device)
        angles = angles.to(device)
        normals = normals.to(device) 
        broadcasted_angles = broadcast_angles(angles, images.shape[-2], images.shape[-1]) 
        input = [images, broadcasted_angles]
        pred_normals = model(input)
        # breakpoint()

        mask = get_mask(normals)
        breakpoint()
        angular_error = mae.update(normals, pred_normals, mask)
        batch_bar.set_postfix(
            MAE=f"{angular_error:.4f}",
            )
        batch_bar.update()

    batch_bar.close()
    print(f"Validation MAE: {mae.error:.4f}")

        
    


if __name__ == '__main__': 
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    blobby_dataset = BlobbyDataset(image_dir=config['paths']['image_dir'],
                                 normal_dir=config['paths']['normal_dir'], 
                                 latent_vectors_dir=config['paths']['latent_vectors_dir'],
                                 info_json_path=config['paths']['info_json_path'],
                                 config=config)
    _, val_dataset, _, val_loader = get_loader(blobby_dataset, batch_size=config['train']['batch_size'], val_split=config['dataset']['val_split'], random_seed=config['dataset']['random_seed']) 
    model = PS_FCN() 
    checkpoint = torch.load('/data2/datasets/ruoguli/idl_project_datas/PS-FCN_B_S_32.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(config['device'])
    eval(model, val_loader, config['device'])
