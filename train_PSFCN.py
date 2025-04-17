import torch
from options  import train_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils 
import wandb
import train_utils
import test_utils

args = train_opts.TrainOpts().parse()

def load_wandb(args, model): 
    wandb.login(key=args.wandb_key)
    run_name = args.wandb_name if args.wandb_name is not None else args.model
    run = wandb.init(
        name    = run_name,
        reinit  = True, 
        project = "idl_project", 
    )
    model_arch = str(model) 
    arch_file   = open("model_arch.txt", "w")
    file_write  = arch_file.write(model_arch)
    arch_file.close()
    wandb.save("model_arch.txt")

def main(args):
    train_loader, val_loader = custom_data_loader.customDataloader(args)
    best_val_n_err = 4.89
    model = custom_model.buildModel(args)
    if args.wandb_key is not None:
        load_wandb(args, model)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Criterion(args)


    for epoch in range(args.start_epoch, args.epochs+1):

        train_loss = train_utils.train(args, train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        val_acc = test_utils.test(args, 'val', val_loader, model, epoch)
        if val_acc < best_val_n_err:
            best_val_n_err = val_acc
            print('Best model saved at epoch %d' % (epoch))
            model_utils.save_checkpoint(args, model, optimizer, scheduler, val_acc, epoch, f'{args.save_root}/{args.wandb_name}/best_model.pth')
        print("Save epoch model at %d" % (epoch))
        model_utils.save_checkpoint(args, model, optimizer, scheduler, val_acc, epoch, f'{args.save_root}/{args.wandb_name}/epoch_model.pth')
        
        wandb.log({
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "val_n_err": val_acc})
        print("Epoch %d: train_loss %.4f, val_n_err %.4f" % (epoch, train_loss, val_acc))

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)