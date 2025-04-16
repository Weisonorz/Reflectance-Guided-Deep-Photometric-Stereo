from models import model_utils
from utils  import time_utils 
from tqdm import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast

def train(args, loader, model, criterion, optimizer, epoch) -> float:
    """
    Train the model for one epoch.
    Returns the average loss over the epoch.
    """
    model.train()
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync)

    batch_bar = tqdm(total=len(loader),  dynamic_ncols=True, leave=False, position=0, desc='Train')
    total_loss = 0. 
    scalar = GradScaler(enabled=args.mixed_precision)
    for i, sample in enumerate(loader):
        data  = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)


        optimizer.zero_grad()
        with autocast(enabled=args.mixed_precision):
            out_var = model(input)
            loss_dict = criterion.forward(out_var, data['tar'])
            loss = loss_dict['N_loss_tensor']

        scalar.scale(loss).backward()

        total_loss += loss_dict['N_loss']
        batch_bar.set_postfix(loss="{:.04f}".format(total_loss/(i+1)),
                              lr="{:.06f}".format(optimizer.param_groups[0]['lr']))
        batch_bar.update()
        scalar.step(optimizer)
        scalar.update()

        iters = i + 1
    
    return total_loss / len(loader)