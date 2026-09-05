import torch
import os
import copy
import sys
import random
Newlimit = 5000  # New limit 
sys.setrecursionlimit(Newlimit)  # Using sys.setrecursionlimit() method  
sys.path.insert(0, ".")

from configs.task_configs_fed import task_configs
from method_fl.aggregation import get_average_net

from train.eval_script.evaluation_fl import perform_eval_for_federated_train


def compute_global_pos_weight_from_label_distributions(label_dis_list, num_samples_list, device, max_clip=20.0, eps=1e-6):
    """
    label_dis_list: list of arrays/tensors, each shape [C], containing POSITIVE counts per class on that client.
    num_samples_list: list of ints, number of training samples per client.
    """
    pos = None
    total = 0

    for ld, n in zip(label_dis_list, num_samples_list):
        ld = torch.as_tensor(ld, dtype=torch.float32, device=device)
        pos = ld if pos is None else (pos + ld)
        total += int(n)

    neg = total - pos
    pos_weight = neg / (pos + eps)
    pos_weight = torch.clamp(pos_weight, 1.0, max_clip)
    return pos_weight

def accelerate_and_free_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()   

def generate_random_integers(n, m):
    """
    Generate n random integers in the range [0, m].

    Parameters:
        n (int): Number of integers to generate.
        m (int): Upper bound (inclusive) of the range.

    Returns:
        List[int]: A list of n random integers between 0 and m.
    """
    return [random.randint(0, m) for _ in range(n)]

def record_training_information(args, logger, tensorboard_writer, loss_dict, task_key, inc_step, client_index, comm_round):
    loss_ce = loss_dict["loss_ce"]
    etf_reg = loss_dict["etf_reg"]
    center_loss_reg = loss_dict["center_loss_reg"]
    hnm_loss = loss_dict["hnm_loss_reg"]
    loss_total = loss_dict["loss_total"]
    if args.CenterLoss_regularization:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f}, center_loss_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg, center_loss_reg))
    elif args.HNM_regularization:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f}, hnm_loss_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg, hnm_loss))
    else:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg))
    tensorboard_writer.add_scalar('loss_total/step_{0}_client_{1}'.format(inc_step, client_index+1), loss_total, comm_round)
    tensorboard_writer.add_scalar('loss_ce/step_{0}_client_{1}'.format(inc_step, client_index+1), loss_ce, comm_round)
    tensorboard_writer.add_scalar('loss_etf/step_{0}_client_{1}'.format(inc_step, client_index+1), etf_reg, comm_round)



def fl_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator):
    #  --------- start federated training and communication  ---------
    best_f1 = 0
    task_output_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.isdir(task_output_dir):
        os.makedirs(task_output_dir, exist_ok=True)

    print("fl_train | number of clients: {0}".format(len(args.ordered_fcl_tasks[inc_step])))
    print("fl_train | partial_client_join: {0}".format(args.partial_client_join))
    print("fl_train | num_of_client_per_round: {0}".format(args.num_of_client_per_round))

    # ---- compute global pos_weight once (optional) ----
    global_pos_weight = None
    if args.bce_w_pos_weight and getattr(args, "pos_weight_mode", "global") == "global":
        label_dis_list = []
        num_samples_list = []

        for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):
            task_trainer_class = task_configs[args.task_config_key]["task_trainer"]
            tmp_trainer = task_trainer_class(
                logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator
            )
            label_dis_list.append(tmp_trainer.get_training_data_label_distribution())
            num_samples_list.append(tmp_trainer.get_num_of_training_data())
            del tmp_trainer
            accelerate_and_free_cache(accelerator)

        global_pos_weight = compute_global_pos_weight_from_label_distributions(
            label_dis_list, num_samples_list, device=device, max_clip=20.0
        )

        # Store in args so all trainers can pick it up
        args.global_pos_weight = global_pos_weight
        logger.info(f"[Global pos_weight] min={global_pos_weight.min().item():.3f}, max={global_pos_weight.max().item():.3f}")
    else:
        args.global_pos_weight = None

    # ------- For each round -------
    for comm_round in range(args.comm_rounds):  
        c_models_list = []

        if args.partial_client_join:
            chosen_client_index = generate_random_integers(args.num_of_client_per_round, len(args.client_list))
            client_weight = [1 for _ in range(args.num_of_client_per_round)]
        else:
            client_weight = [1 for _ in range(len(args.client_list))]

        client_output_dir_list = []

        for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):  # ------- For each client -------
            client_output_dir = os.path.join(args.output_dir, "checkpoints", "client{}_{}".format(client_index + 1, task_key))
            if not os.path.isdir(client_output_dir):
                os.makedirs(client_output_dir, exist_ok=True)
            client_output_dir_list.append(client_output_dir)

            if args.partial_client_join and not (client_index in chosen_client_index):
                continue

            # Local train the model
            temp_model = copy.deepcopy(model)
            task_trainer_class = task_configs[args.task_config_key]["task_trainer"]
            task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)
            loss_dict, c_model = task_trainer.train(temp_model, task_key)

            accelerate_and_free_cache(accelerator)

            # record training information, e.g. loss
            record_training_information(args, logger, tensorboard_writer, loss_dict, task_key, inc_step, client_index, comm_round)
            
            # Store the model parameters for later weight averaging          
            c_model_dict = {}
            for n in c_model.state_dict().keys():
                if n in model.comm_state_dict_names:
                    c_model_dict[n] = c_model.state_dict()[n].data.to(device)
            c_models_list.append(c_model_dict)
            del task_trainer, c_model, temp_model

        accelerate_and_free_cache(accelerator)

        # --------- Average client models to get the global model ---------
        model = get_average_net(model, c_models_list, client_weight, device)

        del c_models_list
        model.to(device)
        accelerate_and_free_cache(accelerator)

        # --------- perform evaluation in each communication round ---------
        eval_content, global_cls_avg_f1, result_dict, _, _, _, _ = perform_eval_for_federated_train(inc_step, comm_round, args, logger, tensorboard_writer, model, accelerator, device)
    
        if global_cls_avg_f1 > best_f1:
            best_f1 = global_cls_avg_f1
            model_save_path = os.path.join(task_output_dir, "fedavg_best-global-model_step-{}.pth".format(inc_step))
            torch.save(model.state_dict(), model_save_path)
            logger.info("--- best class-average global F1 achived: {:.2f}% at round {} ---".format(best_f1 * 100, comm_round))

        if comm_round == (args.comm_rounds - 1):
            model_save_path = os.path.join(task_output_dir, "fedavg_final-model-{}-round_step-{}.pth".format(args.comm_rounds, inc_step))
            torch.save(model.state_dict(), model_save_path)

    return eval_content
