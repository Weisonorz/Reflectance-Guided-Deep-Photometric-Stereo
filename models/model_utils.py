import os
import torch
import torch.nn as nn
import re 
import time 

def getInput(args, data):
    input_list = [data['input']]
    if args.in_light: input_list.append(data['l'])
    if data['brdf'] is not None: input_list.append(data['brdf'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    if args.dataset == 'PS_Synth_Dataset':
        input, target, mask = sample['img'], sample['N'], sample['mask']
        if timer: timer.updateTime('ToCPU')
        if args.cuda:
            input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); 

        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        mask_var   = torch.autograd.Variable(mask, requires_grad=False);

        if timer: timer.updateTime('ToGPU')
        data = {'input': input_var, 'tar': target_var, 'm': mask_var}

        if args.in_light:
            light = sample['light'].expand_as(input)
            if args.cuda: light = light.cuda()
            light_var = torch.autograd.Variable(light);
            data['l'] = light_var
        return data 
    elif args.dataset == 'PS_Blobby_BRDF_Dataset': 
        input, target, mask, brdf = sample['img'], sample['N'], sample['mask'], sample['latent_vector']
        if timer: timer.updateTime('ToCPU')
        if args.cuda:
            input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); brdf = brdf.cuda();

        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        brdf_var   = torch.autograd.Variable(brdf)
        mask_var   = torch.autograd.Variable(mask, requires_grad=False);

        if timer: timer.updateTime('ToGPU')
        data = {'input': input_var, 'tar': target_var, 'm': mask_var, 'brdf': brdf}

        if args.in_light:
            light = sample['light'].expand_as(input)
            if args.cuda: light = light.cuda()
            light_var = torch.autograd.Variable(light);
            data['l'] = light_var
        return data 

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in += 3
    print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def loadCheckpoint_to_PSFCN_CBN_debug(path, model, cuda=True, output_log_file=None):
    """
    Loads weights from a pretrained PS_FCN checkpoint into a PS_FCN_CBN model.
    Logs detailed information about loaded, skipped, and uninitialized weights
    to the specified output_log_file.

    Handles mapping for weights AND biases for extractor.conv1-5.
    Assumes original model used model_utils.conv(batchNorm=False).
    """
    if output_log_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_log_file = f"logs/loading_log_{timestamp}.txt"

    log_lines = []
    log_lines.append(f"--- Loading Checkpoint: {path} ---")
    log_lines.append(f"--- Target Model: {type(model).__name__} ---")
    log_lines.append("-" * 50)

    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    log_lines.append(f"Checkpoint loaded successfully from: {path}")
    pretrained_state_dict = checkpoint.get('state_dict', checkpoint) # Handle cases where checkpoint might just be the state_dict
    model_dict = model.state_dict()
    new_model_keys = set(model_dict.keys())

    new_state_dict = {}
    loaded_keys_list = []
    skipped_mismatch_list = []
    skipped_nonexistent_list = []
    # removed skipped_bias_list as biases are now loaded
    original_keys_processed = set()

    log_lines.append("\n--- Processing Pretrained Parameters ---")
    for k_pretrained, v_pretrained in pretrained_state_dict.items():
        original_keys_processed.add(k_pretrained)
        processed = False

        # --- Special Mapping for extractor.conv1 to conv5 Weights ---
        match_weight = re.match(r'extractor\.conv(\d+)\.0\.weight', k_pretrained)
        if match_weight and int(match_weight.group(1)) <= 5:
            layer_num = match_weight.group(1)
            k_new = f'extractor.conv{layer_num}.regular_conv.weight'
            if k_new in model_dict:
                if model_dict[k_new].shape == v_pretrained.shape:
                    new_state_dict[k_new] = v_pretrained
                    loaded_keys_list.append(f"{k_pretrained} -> {k_new} (Shape: {tuple(v_pretrained.shape)})")
                    processed = True
                else:
                    skipped_mismatch_list.append(f"{k_pretrained} (Shape: {tuple(v_pretrained.shape)}) vs {k_new} (Shape: {tuple(model_dict[k_new].shape)})")
                    processed = True
            else:
                 skipped_nonexistent_list.append(f"{k_pretrained} (Target key {k_new} not found in new model)")
                 processed = True
            # Continue to next key in outer loop IF processed, otherwise might be handled by bias rule or direct match
            if processed: continue

        # --- Special Mapping for extractor.conv1 to conv5 Biases ---
        match_bias = re.match(r'extractor\.conv(\d+)\.0\.bias', k_pretrained)
        if match_bias and int(match_bias.group(1)) <= 5:
            layer_num = match_bias.group(1)
            k_new = f'extractor.conv{layer_num}.regular_conv.bias' # Target is now the bias of regular_conv
            if k_new in model_dict:
                if model_dict[k_new].shape == v_pretrained.shape:
                    new_state_dict[k_new] = v_pretrained
                    loaded_keys_list.append(f"{k_pretrained} -> {k_new} (Shape: {tuple(v_pretrained.shape)})")
                    processed = True
                else:
                    skipped_mismatch_list.append(f"{k_pretrained} (Shape: {tuple(v_pretrained.shape)}) vs {k_new} (Shape: {tuple(model_dict[k_new].shape)})")
                    processed = True
            else:
                 skipped_nonexistent_list.append(f"{k_pretrained} (Target key {k_new} not found in new model)")
                 processed = True
            # Continue to next key in outer loop IF processed
            if processed: continue

        # --- Try Direct Mapping for other keys (Regressor, extractor.conv6/7 etc.) ---
        # This rule will now handle keys like extractor.conv7.0.bias directly too.
        if not processed:
            if k_pretrained in model_dict:
                if model_dict[k_pretrained].shape == v_pretrained.shape:
                    new_state_dict[k_pretrained] = v_pretrained
                    loaded_keys_list.append(f"{k_pretrained} -> {k_pretrained} (Direct Match, Shape: {tuple(v_pretrained.shape)})")
                    processed = True
                else:
                    skipped_mismatch_list.append(f"{k_pretrained} (Shape: {tuple(v_pretrained.shape)}) vs {k_pretrained} (Shape: {tuple(model_dict[k_pretrained].shape)})")
                    processed = True
                # Continue to next key in outer loop IF processed
                if processed: continue

        # --- If not processed by any rule above, it's skipped as non-existent in target ---
        if not processed:
             skipped_nonexistent_list.append(f"{k_pretrained} (No matching key or rule in new model)")


    log_lines.append("\n--- Loading into Model ---")
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True) # strict=True ensures all model_dict keys are consumed
    log_lines.append("model.load_state_dict(model_dict) executed.")


    # --- Report Summary ---
    log_lines.append("\n" + "=" * 50)
    log_lines.append("      State Dict Loading Summary")
    log_lines.append("=" * 50)

    # Loaded Keys
    log_lines.append(f"\n--- ({len(loaded_keys_list)}) Keys Loaded Successfully ---")
    loaded_keys_list.sort()
    log_lines.extend(loaded_keys_list)

    # Skipped Keys (Shape Mismatch)
    log_lines.append(f"\n--- ({len(skipped_mismatch_list)}) Pretrained Keys Skipped (Shape Mismatch) ---")
    skipped_mismatch_list.sort()
    log_lines.extend(skipped_mismatch_list)

    # Skipped Keys (Non-Existent Target)
    log_lines.append(f"\n--- ({len(skipped_nonexistent_list)}) Pretrained Keys Skipped (No Target Layer/Param) ---")
    skipped_nonexistent_list.sort()
    log_lines.extend(skipped_nonexistent_list)

    # Keys in New Model NOT Loaded
    final_loaded_keys = set(new_state_dict.keys())
    not_loaded_new_keys = new_model_keys - final_loaded_keys
    not_loaded_list = sorted(list(not_loaded_new_keys))
    log_lines.append(f"\n--- ({len(not_loaded_list)}) Keys in New Model NOT Loaded from Checkpoint (Initialized Instead) ---")
    log_lines.extend(not_loaded_list)
    log_lines.append("=" * 50)


    # --- Write log to file ---
    try:
        with open(output_log_file, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        print(f"Detailed loading log saved to: {output_log_file}")
    except IOError as e:
        print(f"Error writing log file {output_log_file}: {e}")
        print("\n--- Log Output (Console) ---")
        for line in log_lines:
            print(line)

def loadCheckpoint_to_PSFCN_CBN(path, model, cuda=True):
    """
    Loads weights from a pretrained PS_FCN checkpoint into a PS_FCN_CBN model.
    """
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    print(f"Loading pretrained weights for PS_FCN_CBN from: {path} (using standard loader)")
    pretrained_state_dict = checkpoint.get('state_dict', checkpoint)
    model_dict = model.state_dict()
    new_state_dict = {}
    loaded_count = 0
    skipped_count = 0

    for k_pretrained, v_pretrained in pretrained_state_dict.items():
        mapped = False
        # Special Mapping for extractor.conv1 to conv5 Weights
        match_weight = re.match(r'extractor\.conv(\d+)\.0\.weight', k_pretrained)
        if match_weight and int(match_weight.group(1)) <= 5:
            layer_num = match_weight.group(1)
            k_new = f'extractor.conv{layer_num}.regular_conv.weight'
            if k_new in model_dict and model_dict[k_new].shape == v_pretrained.shape:
                new_state_dict[k_new] = v_pretrained
                mapped = True
            if mapped: continue

        # Special Mapping for extractor.conv1 to conv5 Biases
        match_bias = re.match(r'extractor\.conv(\d+)\.0\.bias', k_pretrained)
        if match_bias and int(match_bias.group(1)) <= 5:
            layer_num = match_bias.group(1)
            k_new = f'extractor.conv{layer_num}.regular_conv.bias'
            if k_new in model_dict and model_dict[k_new].shape == v_pretrained.shape:
                new_state_dict[k_new] = v_pretrained
                mapped = True
            if mapped: continue

        # Direct mapping for others
        if not mapped:
            if k_pretrained in model_dict and model_dict[k_pretrained].shape == v_pretrained.shape:
                new_state_dict[k_pretrained] = v_pretrained
                mapped = True

        if mapped:
            loaded_count += 1
        else:
            skipped_count += 1


    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True) # Use strict=True

def save_checkpoint(args, model, optimizer, scheduler, val_acc, epoch, path):
    torch.save(
        {'state_dict'               : model.state_dict(),
         'model'                    : args.model,     
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         'best_val_acc'             : val_acc,
         'epoch'                    : epoch},
         path
    )

# def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
#     state   = {'state_dict': model.state_dict(), 'model': args.model}
    
#     torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
#     torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )