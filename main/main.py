import torch
import os
import time
import sys
from accelerate import Accelerator, DistributedDataParallelKwargs
from tensorboardX import SummaryWriter
Newlimit = 5000  # New limit 
sys.setrecursionlimit(Newlimit)  # Using sys.setrecursionlimit() method  
sys.path.insert(0, ".")

from utils.arg_parser import get_parser
from utils.logger_utils import root_logger
from configs.model_configs import model_configs
from configs.task_configs_fed import task_configs
from modeling.prepare_model import prepare_model
from train.eval_script.test import test

from train.train_script.local_train import local_train
from train.train_script.fl_train import fl_train


def main():
    # --------------------- Set up experiment info ---------------------
    parser = get_parser()  # Get the parser and parse arguments
    args = parser.parse_args()

    args.visual_input_type = model_configs[args.encoder_name]["visual_input_type"]
    
    args.ordered_fcl_tasks = []
    for inc_step in range(args.num_cl_tasks):
        ordered_task_buff = []
        for client_id in range(args.num_fl_tasks):
            ordered_task_buff.append("{0}_step_{1}_client_{2}".format(args.dataset_name, inc_step, client_id+1))
        args.ordered_fcl_tasks.append(ordered_task_buff)
    
    args.client_list = []
    for client_id in range(args.num_fl_tasks):
        client_name = "{0}_client_{1}".format(args.dataset_name, client_id + 1)
        args.client_list.append(client_name)
    
    if "medmnist" in args.dataset_name:
        args.task_config_key = "medmnist_train"
        task_configs[args.task_config_key]["images_source"] = args.dataset_name
    else:
        args.task_config_key = "{0}_train".format(args.dataset_name)

    task_configs[args.task_config_key]['classifier_type'] = args.classifier_type

    # create output directories
    if args.do_train:
        setting_name = (f"{args.batch_size}_batch_{torch.cuda.device_count()}_GPU")
        timestr = time.strftime("%m%d-%H%M%S")
        if args.exp_name == None:
            final_exp_name = "{0}_{1}".format(setting_name, timestr)
        else:
            final_exp_name = "{0}_{1}_{2}".format(args.exp_name, setting_name, timestr)
        args.output_dir = os.path.join(args.output_dir, final_exp_name)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
    elif args.do_test:
        if args.exp_name == None:
            raise ValueError("For testing, exp_name cannot be none!")
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
        if not os.path.isdir(args.output_dir):
            raise ValueError("For testing, output_dir must exist!")
        
    if args.optimizer_mode == "adapter":
        assert args.adapter_reduction_factor > 0

    if args.do_train:
        logger = root_logger(os.path.join(args.output_dir, f"log_{args.encoder_name}_{args.num_cl_tasks}-steps_{args.num_fl_tasks}-clients_{args.seed}.txt"))
    elif args.do_test:
        logger = root_logger(os.path.join(args.output_dir, f"log_{args.encoder_name}_{args.num_cl_tasks}-steps_{args.num_fl_tasks}-clients_{args.seed}_test.txt"))
    logger.info("-" * 100)
    logger.info("Arguments: %s", args)

    if args.do_train:
        tensorboard_path = '{0}/tensorboard'.format(args.output_dir)
    elif args.do_test:
        tensorboard_path = '{0}/tensorboard_test'.format(args.output_dir)
    if not os.path.isdir(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_path)  # Specify the log directory
    
    # --------------------- Create the ContinualLearner model based on encoder_name argument  ---------------------
    model_config = model_configs[args.encoder_name]

    # --- create the model --- 
    model = prepare_model(args, logger)
    if ('dat' in args.optimizer_mode) or ("Dual_Classifier" in args.classifier_type) or (args.client_specific_head) or ("ETF_Classifier" in args.classifier_type):
        find_unused_parameters = True
    else:
        find_unused_parameters = False
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters, broadcast_buffers=False)
    if args.do_wandb_logging:
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    if args.do_train:
        # --- Print some model info ---
        logger.info("-" * 100)
        logger.info("Succesfully initialized {}-based Learner".format(model_config["encoder_name"]))
        logger.info("Number of CL step: {}, Number of FL clients: {}".format(args.num_cl_tasks, args.num_fl_tasks))
        logger.info("CL Algorithm: {}".format(args.cl_method))
        logger.info("FL Algorithm: {}".format(args.fl_method))
        logger.info("{} client heads: {}".format(len(args.client_list), ",".join(args.client_list)))
        logger.info("Total communication round: {}, Local epochs per round: {}".format(args.comm_rounds, args.local_epochs))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total Parameters: {:.2f}M".format(total_params * 10 ** -6))
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
        logger.info("Trainable Parameters: {:.2f}M ({:.2f}%)".format(trainable_params * 10 ** -6, (trainable_params / total_params * 100)))
        logger.info("Model training starts from pretrained model name: {}".format(args.pretrained_model_name))
        logger.info("Model checkpoints saved to {}".format(args.output_dir))
        logger.info("-" * 100)
        # --- Start training ---
        for inc_step in range(args.num_cl_tasks):
            if args.cl_method == "direct_ft" and args.fl_method == "centralized":
                eval_content = local_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator)
            elif args.cl_method == "direct_ft" and args.fl_method == "standard_aggregation":
                eval_content = fl_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator)
            else:
                eval_content = fl_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator)
            # --- store training log ---
            file_name = "inc_step_{0}_result.txt".format(inc_step)
            file_path = os.path.join(args.output_dir, file_name)
            with open(file_path, "w") as file:
                file.writelines(item + "\n" for item in eval_content)
    elif args.do_test:
        # --- Print some model info ---
        logger.info("-" * 100)
        logger.info("Succesfully initialized {}-based Learner".format(model_config["encoder_name"]))
        logger.info("Number of CL step: {}, Number of FL clients: {}".format(args.num_cl_tasks, args.num_fl_tasks))
        logger.info("{} client heads: {}".format(len(args.client_list), ",".join(args.client_list)))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total Parameters: {:.2f}M".format(total_params * 10 ** -6))
        logger.info("-" * 100)
        # --- Start training ---
        for inc_step in range(args.num_cl_tasks):
            eval_content = test(inc_step, args, logger, tensorboard_writer, model, device, accelerator)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()

