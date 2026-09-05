import torch
import os
import copy
import sys
Newlimit = 5000  # New limit 
sys.setrecursionlimit(Newlimit)  # Using sys.setrecursionlimit() method  
sys.path.insert(0, ".")

from configs.task_configs_fed import task_configs
from train.eval_script.evaluation_local import perform_eval_for_local_train


def accelerate_and_free_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()   

def local_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator):
    final_eval_content = []

    for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):  # For each client
        task_output_dir = os.path.join(args.output_dir, "checkpoints", "client{}_{}".format(client_index + 1, task_key))
        client_model = copy.deepcopy(model)  # Deep copy ensures independent parameters

        if inc_step > 0:
            previous_task_key = args.ordered_fcl_tasks[inc_step - 1][client_index]
            previous_task_output_dir = args.output_dir + "/checkpoints/" + "client{}_{}".format(client_index + 1, previous_task_key)
            print("load last step model from previous_task_output_dir: {0}".format(previous_task_output_dir))
            last_inc_step_model_path = os.path.join(previous_task_output_dir, "local-train_best-model_client-{}_step-{}.pth".format(client_index + 1, inc_step - 1))
            state_dict = torch.load(last_inc_step_model_path)
            client_model.load_state_dict(state_dict)

        best_f1 = 0
        for comm_round in range(args.comm_rounds):  # For each round: centralized train one model
            logger.info("-" * 80)
            task_name = task_configs[args.task_config_key]["task_name"]

            task_trainer_class = task_configs[args.task_config_key]["task_trainer"]
            task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)
            loss_dict, _ = task_trainer.train(client_model, task_key)
            accelerate_and_free_cache(accelerator)

            loss_total = loss_dict["loss_total"]
            
            logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {}".format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total))
            tensorboard_writer.add_scalar('loss/step_{0}_client_{1}'.format(inc_step, client_index+1), loss_total, comm_round)

            eval_content, cls_avg_f1, result_dict = perform_eval_for_local_train(inc_step, comm_round, client_index, args, logger, tensorboard_writer, client_model, task_trainer, accelerator, device)
            final_eval_content += eval_content
            final_eval_content += ["-" * 100]

            if cls_avg_f1 > best_f1:
                best_f1 = cls_avg_f1
                model_save_path = os.path.join(task_output_dir, "local-train_best-model_client-{}_step-{}.pth".format(client_index + 1, inc_step))
                torch.save(client_model.state_dict(), model_save_path)
                logger.info("--- best class-average F1 achived: {:.2f}% at round {} ---".format(best_f1 * 100, comm_round))

            if comm_round == (args.comm_rounds - 1):
                model_save_path = os.path.join(task_output_dir, "local-train_final-model-{}-round_client-{}_step-{}.pth".format(
                    args.comm_rounds, client_index + 1, inc_step))
                torch.save(client_model.state_dict(), model_save_path)

        del client_model
    return final_eval_content