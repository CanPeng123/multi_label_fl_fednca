from configs.model_configs import model_configs
from configs.task_configs_fed import task_configs
from modeling import create_model_map

def prepare_model(args, logger):
    print("main | prepare_model | encoder_name:", args.encoder_name)
    print("main | prepare_model | pretrained_model_name:", args.pretrained_model_name)
    print("main | prepare_model | client_specific_head:", args.client_specific_head)
    if args.pretrained_model_name == "imagenet":
        imagenet_pretrain_flag = True
    else:
        imagenet_pretrain_flag = False
    print("main | prepare_model | imagenet_pretrain:", imagenet_pretrain_flag)

    create_model_method = create_model_map[args.encoder_name]
    model_config = model_configs[args.encoder_name]

    if 'adapter' in args.optimizer_mode:
        adapter_config = {}
        adapter_config['names'] = ['adapter']
        adapter_config['device'] = 'cuda'
        model_config['adapter_config'] = adapter_config
    
    model = create_model_method(
        logger=logger, 
        model_name_or_path=args.pretrained_model_name,
        client_specific_head=args.client_specific_head,
        client_list=args.client_list,
        model_config=model_configs[args.encoder_name],
        task_configs=task_configs,
        device='cuda',
        dataset_path=args.data_dir,
        imagenet_pretrain=imagenet_pretrain_flag,
        obtain_class_wise_feature=args.obtain_class_wise_feature,
        projection_type = args.projection_type,
        seperate_background_class=args.seperate_background_class,
        norm_type=args.norm_type,
        dataset_name=args.dataset_name
        )
    
    model.comm_state_dict_names = []

    if args.optimizer_mode=='full':
        for n, p in model.named_parameters():
            p.requires_grad = True
        for n in model.state_dict().keys():
            model.comm_state_dict_names.append(n)
    elif args.optimizer_mode=='frozen':
        for n, p in model.named_parameters():
            p.requires_grad = False
    elif args.optimizer_mode == "adapter":
        for n, p in model.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
        for n in model.state_dict().keys():
            if 'adapter' in n:
                model.comm_state_dict_names.append(n)
    elif args.optimizer_mode=='none':
        pass
    else:
        raise ValueError("Something wrong with the optimizer_mode.")

    return model
