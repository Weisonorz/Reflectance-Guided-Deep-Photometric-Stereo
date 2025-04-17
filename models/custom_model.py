from . import model_utils
import torch 

def buildModel(args):
    print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {'img_num': args.in_img_num, 'in_light': args.in_light}
    if args.model == 'PS_FCN': 
        from models.PS_FCN import PS_FCN
        model = PS_FCN(args.fuse_type, args.use_BN, in_c, other)
    # elif args.model == 'PS_FCN_run':
    #     from models.PS_FCN_run import PS_FCN
    #     model = PS_FCN(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'PS_FCN_CBN':
        from models.PS_FCN_CBN import PS_FCN_CBN
        model = PS_FCN_CBN(fuse_type=args.fuse_type, batchNorm= args.use_BN, 
                           c_in=in_c, other=other)
    elif args.model == 'PS_FCN_FiLM':
        from models.PS_FCN_FiLM import PS_FCN_FiLM
        model = PS_FCN_FiLM(args.fuse_type, args.use_BN, in_c)
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))
    
    if args.cuda: 
        model = model.cuda()
    
    if args.retrain: 
        if args.model == 'PS_FCN_CBN':
            print("=> using pre-trained model %s" % (args.retrain))
            model_utils.loadCheckpoint_to_PSFCN_CBN_debug(args.retrain, model, cuda=args.cuda)
        elif args.model == 'PS_FCN_FiLM':
            print("=> using pre-trained model %s" % (args.retrain))
            model_utils.loadCheckpoint_to_PSFCN_FiLM_debug(args.retrain, model, cuda=args.cuda)
        else:
            print("=> using pre-trained model %s" % (args.retrain))
            model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume:
        print("=> Resume loading checkpoint %s" % (args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)

    # if args.compile:
    #     print("=> Compiling model")
    #     model = torch.compile(model)
    #     print("=> Compiled model")

    print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model