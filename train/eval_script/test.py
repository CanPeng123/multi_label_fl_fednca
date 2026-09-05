import torch
import sys
import os
import copy
Newlimit = 5000  # New limit 
sys.setrecursionlimit(Newlimit)  # Using sys.setrecursionlimit() method  
sys.path.insert(0, ".")

from train.eval_script.evaluation_fl import perform_eval_for_federated_train
from neural_collapse.feature_covariance_info import compute_classwise_average, compute_global_mean_feature
from neural_collapse.feature_covariance_info import compute_within_class_covariance, compute_between_class_covariance, compute_total_covariance
from neural_collapse.feature_covariance_info import compute_distance_alignment_matrix
from neural_collapse.feature_covariance_info import compute_l2_norms_and_normalized_directions, compute_direction_cosine_similarity_matrix


def accelerate_and_free_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()   

def test(inc_step, args, logger, tensorboard_writer, model, device, accelerator):
    best_f1 = 0
    final_eval_content = []
    task_output_dir = os.path.join(args.output_dir, "test")
    if not os.path.isdir(task_output_dir):
        os.makedirs(task_output_dir, exist_ok=True)

    current_model = copy.deepcopy(model)  # Deep copy ensures independent parameters
    inc_step_model_path = args.output_dir + "/checkpoints/" + "{0}_step-{1}.pth".format(args.model_name, inc_step)
    print("load current step model from inc_step_model_path: {0}".format(inc_step_model_path))
    
    state_dict = torch.load(inc_step_model_path)
    current_model.load_state_dict(state_dict)

    current_model.to(device)
    accelerate_and_free_cache(accelerator)

    # --------- perform evaluation in each communication round ---------
    comm_round = -1
    eval_content, _, result_dict, global_result_dict, all_feature, all_soft_pred, all_label = perform_eval_for_federated_train(
        inc_step, comm_round, args, logger, tensorboard_writer, current_model, accelerator, device, return_feature=True, test_trainset=args.test_train_set)
    final_eval_content += eval_content
    final_eval_content += ["-" * 100]

    number_of_class = len(all_label[0][0])

    client_output_dir_list = []
    for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):  # For each client
        client_output_dir = os.path.join(args.output_dir, "test", "client{}_{}".format(client_index + 1, task_key))
        if not os.path.isdir(client_output_dir):
            os.makedirs(client_output_dir, exist_ok=True)
        client_output_dir_list.append(client_output_dir)

    for client_index, client_result_dict in enumerate(result_dict):
        if args.obtain_class_wise_feature:
            filtered_label = []
            filtered_feature = []
            for index in range(len(all_label[client_index])):
                # Step 1: Get indices of active classes (where label == 1)
                active_class_indices = (all_label[client_index][index] == 1).nonzero(as_tuple=False).squeeze()  # shape: [num_active]
                # Step 2: Gather corresponding feature vectors
                if active_class_indices.ndim == 0:  # If only one active class, ensure it's still treated as 1D tensor
                    active_class_indices = active_class_indices.unsqueeze(0)
                selected_features = all_feature[client_index][index][active_class_indices]  # shape: [num_active, 512]
                one_hot_labels = torch.eye(number_of_class)[active_class_indices]  # shape: [num_active, number_of_class]
                filtered_feature.extend(selected_features)
                filtered_label.extend(one_hot_labels)

        if args.multiplicity_1:
            filtered_label = []
            filtered_feature = []
            for index in range(len(all_label[client_index])):
                label = all_label[client_index][index]  # shape: [10]
                # Check for exactly one active label
                active_class_indices = (label == 1).nonzero(as_tuple=False).squeeze()  # shape: [num_active]
                # Step 2: Gather corresponding feature vectors
                if active_class_indices.ndim != 0:
                    continue
                filtered_feature.append(all_feature[client_index][index])  
                filtered_label.append(all_label[client_index][index])

        # --- calculate class-wise average features and global feature ---
        if args.obtain_class_wise_feature:
            classwise_averages = compute_classwise_average(filtered_feature, filtered_label, dataset_name=args.dataset_name, multiplicity_1=args.multiplicity_1, 
                                                           seperate_background_class=args.seperate_background_class) 
        else:
            classwise_averages = compute_classwise_average(all_feature[client_index], all_label[client_index], dataset_name=args.dataset_name, multiplicity_1=args.multiplicity_1, 
                                                           seperate_background_class=args.seperate_background_class) 
        global_average = compute_global_mean_feature(classwise_averages)

        # --- calculate covariance information --- 
        if args.test_train_set:
            within_class_cov_file_path = "{0}_within_class_cov_trainset.png".format(client_output_dir_list[client_index])
        else:
            within_class_cov_file_path = "{0}_within_class_cov_testset.png".format(client_output_dir_list[client_index])
        if args.obtain_class_wise_feature:
            within_class_cov = compute_within_class_covariance(filtered_feature, filtered_label, classwise_averages, plot_path=within_class_cov_file_path)
        else:
            within_class_cov = compute_within_class_covariance(all_feature[client_index], all_label[client_index], classwise_averages, plot_path=within_class_cov_file_path)
        if args.test_train_set:
            between_class_cov_file_path = "{0}_between_class_cov_trainset.png".format(client_output_dir_list[client_index])
        else:
            between_class_cov_file_path = "{0}_between_class_cov_testset.png".format(client_output_dir_list[client_index])
        between_class_cov = compute_between_class_covariance(classwise_averages, global_average, plot_path=between_class_cov_file_path)
        if args.test_train_set:
            total_cov_file_path = "{0}_total_cov_trainset.png".format(client_output_dir_list[client_index])
        else:
            total_cov_file_path = "{0}_total_cov_testset.png".format(client_output_dir_list[client_index])
        if args.obtain_class_wise_feature:
            total_cov = compute_total_covariance(filtered_feature, filtered_label, global_average, plot_path=total_cov_file_path)
        else:
            total_cov = compute_total_covariance(all_feature[client_index], all_label[client_index], global_average, plot_path=total_cov_file_path)
        
        # --- calculate class-wise prototype alignment information --- 
        if args.test_train_set:
            distance_alignment_matrix_path = "{0}_distance_alignment_matrix_trainset.png".format(client_output_dir_list[client_index])
        else:
            distance_alignment_matrix_path = "{0}_distance_alignment_matrix_testset.png".format(client_output_dir_list[client_index])
        if args.multiplicity_1:
            buff = distance_alignment_matrix_path.split('.')
            distance_alignment_matrix_path = "{0}.{1}_multiplicity_1.{2}".format(buff[0], buff[1], buff[2])
            print("distance_alignment_matrix_path: {0}".format(distance_alignment_matrix_path))
        distance_alignment_matrix = compute_distance_alignment_matrix(classwise_averages, global_average, plot_path=distance_alignment_matrix_path)

        _, class_wise_directions = compute_l2_norms_and_normalized_directions(classwise_averages, global_average)
        if args.test_train_set:
            similarity_plot_path = "{0}_cosine_similarity_matrix_trainset.png".format(client_output_dir_list[client_index])
            compared_similarity_plot_path = "{0}_cosine_similarity_matrix_compared_w_ETF_trainset.png".format(client_output_dir_list[client_index])
        else:
            similarity_plot_path = "{0}_cosine_similarity_matrix_testset.png".format(client_output_dir_list[client_index])
            compared_similarity_plot_path = "{0}_cosine_similarity_matrix_compared_w_ETF_testset.png".format(client_output_dir_list[client_index])
        if args.multiplicity_1:
            buff = similarity_plot_path.split('.')
            similarity_plot_path = "{0}.{1}_multiplicity_1.{2}".format(buff[0], buff[1], buff[2])
            buff = compared_similarity_plot_path.split('.')
            compared_similarity_plot_path = "{0}.{1}_multiplicity_1.{2}".format(buff[0], buff[1], buff[2])
            print("similarity_plot_path: {0}".format(similarity_plot_path))
            print("compared_similarity_plot_path: {0}".format(compared_similarity_plot_path))
        similarity_matrix, similarity_matrix_compared_w_ETF = compute_direction_cosine_similarity_matrix(class_wise_directions, similarity_plot_path, compared_similarity_plot_path)
        
    return final_eval_content
