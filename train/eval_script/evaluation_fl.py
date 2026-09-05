import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

from configs.model_configs import model_configs
from configs.task_configs_fed import task_configs
from data.dataset_info import CIFAR10_LABEL_LIST
from data.dataset_info import VOC2012_LABEL_CLASSES, XRAY14_LABEL_CLASSES, COCO2017_LABEL_CLASSES
from data.dataset_info import CHESTMNIST_LABEL_LIST, CHESTMNIST_LABEL_LIST_W_BACKGROUND, SKIN_LABEL_LIST, CANCER_LABEL_LIST
from train.eval_script.evaluation_metric import accelerate_and_free_cache
from train.eval_script.evaluation_metric import compute_mcc, compute_acc, compute_precision, compute_recall, compute_specificity, compute_f1, compute_auc


def convert_probability_to_hard_prediction(args, label_list, soft_pred_list):
    if args.pred_threshold_type in ['fix_05', 'PR_curve', 'class_mean']:
        hard_pred_list = []
        pred_threshold_list = []
        for cls_index, _ in enumerate(label_list):          
            cls_soft_pred = soft_pred_list[cls_index]
            cls_label_list = label_list[cls_index]
            if args.pred_threshold_type == 'fix_05':
                cls_hard_pred = (cls_soft_pred > 0.5).astype(float)
                pred_threshold_list.append(0.5)
            elif args.pred_threshold_type == 'PR_curve':
                precision, recall, thresholds = precision_recall_curve(cls_label_list, cls_soft_pred)

                # align with thresholds
                precision = precision[:-1]
                recall = recall[:-1]

                f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

                best_threshold = thresholds[np.argmax(f1_scores)]

                # hard clamp (your requirement)
                best_threshold = float(np.clip(best_threshold, 0.05, 0.5))

                cls_hard_pred = (cls_soft_pred > best_threshold).astype(float)
                pred_threshold_list.append(best_threshold)

            elif args.pred_threshold_type == 'class_mean':
                mean_value = np.mean(cls_soft_pred, axis=0)
                cls_hard_pred = (cls_soft_pred > mean_value).astype(float)
                pred_threshold_list.append(mean_value)
            else:
                raise ValueError("The prediction threshold type is set wrong!")
            hard_pred_list.append(cls_hard_pred)
    elif args.pred_threshold_type in ['max_w_tolerance']:
        soft_pred_tensor = torch.tensor(soft_pred_list)
        # Find the maximum value for each batch along the class dimension (dim=0)
        max_values, _ = torch.max(soft_pred_tensor, dim=0, keepdim=True)  # Shape: [1, batch_size]
        # Find all indices where the value is within the tolerance of the maximum value
        peak_mask = (soft_pred_tensor >= max_values - args.max_w_tolerance_value) & (soft_pred_tensor <= max_values + args.max_w_tolerance_value)
        # Initialize a multi-hot label tensor with zeros
        multi_hot_labels = torch.zeros_like(soft_pred_tensor, dtype=torch.int)
        # Set the positions corresponding to peak indices to 1
        multi_hot_labels[peak_mask] = 1
        # Convert the multi-hot label tensor to a 2-dimensional list
        multi_hot_labels_list = multi_hot_labels.tolist()
        hard_pred_list = multi_hot_labels_list
        pred_threshold_list = [-1 for cls_index, _ in enumerate(label_list)]
    return hard_pred_list, pred_threshold_list


def compute_class_wise_confusion_matrix(hard_pred_list, label_list):
    cm_matrix = []
    for cls_index, cls_hard_pred in enumerate(hard_pred_list):
        cm_class = confusion_matrix(label_list[cls_index], hard_pred_list[cls_index]).ravel()  # TN, FP, FN, TP
        if cm_class.shape == (1,):
            cm_class = np.array([cm_class[0], 0, 0, 0])
        cm_matrix.append(cm_class)
    return cm_matrix


def compute_class_wise_performance(cm_matrix, soft_pred_list, label_list):
    cls_wise_perform_dict = {}
    cls_wise_precision, cls_wise_recall, cls_wise_specificity = [], [], []
    cls_wise_f1, cls_wise_auc, cls_wise_acc, cls_wise_mcc = [], [], [], []
    cls_wise_tn, cls_wise_fp, cls_wise_fn, cls_wise_tp = [], [], [], []
    for cls_index, cls_cm in enumerate(cm_matrix):  # cls_cm = [TN, FP, FN, TP]
        tn, fp, fn, tp = cls_cm
        precision = compute_precision(tp, tn, fp, fn)
        recall = compute_recall(tp, tn, fp, fn)
        specificity = compute_specificity(tp, tn, fp, fn)
        f1 = compute_f1(precision, recall)
        auc = compute_auc(label_list[cls_index], soft_pred_list[cls_index])
        mcc = compute_mcc(tp, tn, fp, fn)
        acc = compute_acc(tp, tn, fp, fn)

        cls_wise_precision.append(precision)
        cls_wise_recall.append(recall)
        cls_wise_specificity.append(specificity)
        cls_wise_f1.append(f1)
        cls_wise_auc.append(auc)
        cls_wise_acc.append(acc)
        cls_wise_mcc.append(mcc)
        cls_wise_tn.append(tn)
        cls_wise_fp.append(fp)
        cls_wise_fn.append(fn)
        cls_wise_tp.append(tp)

    cls_wise_perform_dict["cls_wise_precision"] = cls_wise_precision
    cls_wise_perform_dict["cls_wise_recall"] = cls_wise_recall
    cls_wise_perform_dict["cls_wise_specificity"] = cls_wise_specificity
    cls_wise_perform_dict["cls_wise_f1"] = cls_wise_f1
    cls_wise_perform_dict["cls_wise_auc"] = cls_wise_auc
    cls_wise_perform_dict["cls_wise_acc"] = cls_wise_acc
    cls_wise_perform_dict["cls_wise_mcc"] = cls_wise_mcc
    cls_wise_perform_dict["cls_wise_tn"] = cls_wise_tn
    cls_wise_perform_dict["cls_wise_fp"] = cls_wise_fp
    cls_wise_perform_dict["cls_wise_fn"] = cls_wise_fn
    cls_wise_perform_dict["cls_wise_tp"] = cls_wise_tp
    return cls_wise_perform_dict


def compute_current_step_cls_wise_performance(args, LABEL_LIST, inc_step, cls_wise_perform_dict, label_list, soft_pred_list):
    current_step_cls_wise_perform_dict = {}
    customer_label_index_list = []

    cls_wise_precision = cls_wise_perform_dict["cls_wise_precision"]
    cls_wise_recall = cls_wise_perform_dict["cls_wise_recall"] 
    cls_wise_specificity = cls_wise_perform_dict["cls_wise_specificity"] 
    cls_wise_f1 = cls_wise_perform_dict["cls_wise_f1"] 
    cls_wise_auc = cls_wise_perform_dict["cls_wise_auc"]
    cls_wise_acc = cls_wise_perform_dict["cls_wise_acc"]
    cls_wise_mcc = cls_wise_perform_dict["cls_wise_mcc"] 
    cls_wise_tn = cls_wise_perform_dict["cls_wise_tn"]
    cls_wise_fp = cls_wise_perform_dict["cls_wise_fp"] 
    cls_wise_fn = cls_wise_perform_dict["cls_wise_fn"]
    cls_wise_tp = cls_wise_perform_dict["cls_wise_tp"]
    
    for index, label in enumerate(LABEL_LIST):
        if label in args.customer_label_list:
            customer_label_index_list.append(index)
    current_step_cls_wise_precision = [cls_wise_precision[i] for i in range(len(cls_wise_precision)) if i in customer_label_index_list]
    current_step_cls_wise_recall = [cls_wise_recall[i] for i in range(len(cls_wise_recall)) if i in customer_label_index_list]
    current_step_cls_wise_f1 = [cls_wise_f1[i] for i in range(len(cls_wise_f1)) if i in customer_label_index_list]
    current_step_cls_wise_specificity = [cls_wise_specificity[i] for i in range(len(cls_wise_specificity)) if i in customer_label_index_list]
    current_step_cls_wise_auc = [cls_wise_auc[i] for i in range(len(cls_wise_auc)) if i in customer_label_index_list]
    current_step_cls_wise_acc = [cls_wise_acc[i] for i in range(len(cls_wise_acc)) if i in customer_label_index_list]
    current_step_cls_wise_mcc = [cls_wise_mcc[i] for i in range(len(cls_wise_mcc)) if i in customer_label_index_list]
    current_step_cls_wise_tn = [cls_wise_tn[i] for i in range(len(cls_wise_tn)) if i in customer_label_index_list]
    current_step_cls_wise_fp = [cls_wise_fp[i] for i in range(len(cls_wise_fp)) if i in customer_label_index_list]
    current_step_cls_wise_fn = [cls_wise_fn[i] for i in range(len(cls_wise_fn)) if i in customer_label_index_list]
    current_step_cls_wise_tp = [cls_wise_tp[i] for i in range(len(cls_wise_tp)) if i in customer_label_index_list]
    current_step_label_list = [label_list[i] for i in range(len(label_list)) if i in customer_label_index_list]
    current_step_soft_label_list = [soft_pred_list[i] for i in range(len(soft_pred_list)) if i in customer_label_index_list]

    current_step_cls_wise_perform_dict['current_step_cls_wise_precision'] = current_step_cls_wise_precision
    current_step_cls_wise_perform_dict['current_step_cls_wise_recall'] = current_step_cls_wise_recall
    current_step_cls_wise_perform_dict['current_step_cls_wise_f1'] = current_step_cls_wise_f1
    current_step_cls_wise_perform_dict['current_step_cls_wise_specificity'] = current_step_cls_wise_specificity
    current_step_cls_wise_perform_dict['current_step_cls_wise_auc'] = current_step_cls_wise_auc
    current_step_cls_wise_perform_dict['current_step_cls_wise_acc'] = current_step_cls_wise_acc
    current_step_cls_wise_perform_dict['current_step_cls_wise_mcc'] = current_step_cls_wise_mcc
    current_step_cls_wise_perform_dict['current_step_cls_wise_tn'] = current_step_cls_wise_tn
    current_step_cls_wise_perform_dict['current_step_cls_wise_fp'] = current_step_cls_wise_fp
    current_step_cls_wise_perform_dict['current_step_cls_wise_fn'] = current_step_cls_wise_fn
    current_step_cls_wise_perform_dict['current_step_cls_wise_tp'] = current_step_cls_wise_tp
    current_step_cls_wise_perform_dict['current_step_label_list'] = current_step_label_list
    current_step_cls_wise_perform_dict['current_step_soft_label_list'] = current_step_soft_label_list
    return current_step_cls_wise_perform_dict, customer_label_index_list


def summarize_per_client_performance(
        args, cls_wise_perform_dict, current_step_cls_wise_perform_dict, pred_threshold_list, eval_result, inc_step, LABEL_LIST, customer_label_index_list):
 
    current_step_cls_wise_f1_dist = {}
    current_step_cls_wise_auc_dist = {}
    current_step_cls_wise_acc_dist = {}
    current_step_cls_wise_mcc_dist = {}
    current_step_cls_wise_precision_dist = {}
    current_step_cls_wise_recall_dist = {}
    current_step_cls_wise_specificity_dist = {}
    current_step_pred_threshold_dist = {}
    current_step_label_name_list = []
    result_content = []

    cls_wise_precision = cls_wise_perform_dict["cls_wise_precision"]
    cls_wise_recall = cls_wise_perform_dict["cls_wise_recall"] 
    cls_wise_specificity = cls_wise_perform_dict["cls_wise_specificity"] 
    cls_wise_f1 = cls_wise_perform_dict["cls_wise_f1"] 
    cls_wise_auc = cls_wise_perform_dict["cls_wise_auc"]
    cls_wise_acc = cls_wise_perform_dict["cls_wise_acc"]
    cls_wise_mcc = cls_wise_perform_dict["cls_wise_mcc"] 
    cls_wise_tn = cls_wise_perform_dict["cls_wise_tn"]
    cls_wise_fp = cls_wise_perform_dict["cls_wise_fp"] 
    cls_wise_fn = cls_wise_perform_dict["cls_wise_fn"]
    cls_wise_tp = cls_wise_perform_dict["cls_wise_tp"]

    current_step_cls_wise_precision = current_step_cls_wise_perform_dict['current_step_cls_wise_precision']
    current_step_cls_wise_recall = current_step_cls_wise_perform_dict['current_step_cls_wise_recall'] 
    current_step_cls_wise_f1 = current_step_cls_wise_perform_dict['current_step_cls_wise_f1']
    current_step_cls_wise_specificity = current_step_cls_wise_perform_dict['current_step_cls_wise_specificity'] 
    current_step_cls_wise_auc = current_step_cls_wise_perform_dict['current_step_cls_wise_auc'] 
    current_step_cls_wise_acc = current_step_cls_wise_perform_dict['current_step_cls_wise_acc'] 
    current_step_cls_wise_mcc = current_step_cls_wise_perform_dict['current_step_cls_wise_mcc']
    current_step_cls_wise_tn = current_step_cls_wise_perform_dict['current_step_cls_wise_tn']
    current_step_cls_wise_fp = current_step_cls_wise_perform_dict['current_step_cls_wise_fp'] 
    current_step_cls_wise_fn = current_step_cls_wise_perform_dict['current_step_cls_wise_fn'] 
    current_step_cls_wise_tp = current_step_cls_wise_perform_dict['current_step_cls_wise_tp']
    current_step_label_list = current_step_cls_wise_perform_dict['current_step_label_list']
    current_step_soft_label_list = current_step_cls_wise_perform_dict['current_step_soft_label_list']

    for i in range(len(LABEL_LIST)):
        if i in customer_label_index_list:
            current_step_cls_wise_f1_dist[LABEL_LIST[i]] = cls_wise_f1[i]
            current_step_cls_wise_auc_dist[LABEL_LIST[i]] = cls_wise_auc[i]
            current_step_cls_wise_acc_dist[LABEL_LIST[i]] = cls_wise_acc[i]
            current_step_cls_wise_mcc_dist[LABEL_LIST[i]] = cls_wise_mcc[i]
            current_step_cls_wise_precision_dist[LABEL_LIST[i]] = cls_wise_precision[i]
            current_step_cls_wise_recall_dist[LABEL_LIST[i]] = cls_wise_recall[i]
            current_step_cls_wise_specificity_dist[LABEL_LIST[i]] = cls_wise_specificity[i]
            current_step_pred_threshold_dist[LABEL_LIST[i]] = pred_threshold_list[i]
            current_step_label_name_list.append(LABEL_LIST[i])
    
    # ------ get class-average performance for current step classes ------
    cls_avg_f1 = sum(current_step_cls_wise_f1) / len(current_step_cls_wise_f1)
    cls_avg_auc = sum(current_step_cls_wise_auc) / len(current_step_cls_wise_auc)
    cls_avg_acc = sum(current_step_cls_wise_acc) / len(current_step_cls_wise_acc)
    cls_avg_mcc = sum(current_step_cls_wise_mcc) / len(current_step_cls_wise_mcc)
    cls_avg_precision = sum(current_step_cls_wise_precision) / len(current_step_cls_wise_precision)
    cls_avg_recall = sum(current_step_cls_wise_recall) / len(current_step_cls_wise_recall)
    cls_avg_specificity = sum(current_step_cls_wise_specificity) / len(current_step_cls_wise_specificity)

    total_tn = sum(current_step_cls_wise_tn)
    total_fp = sum(current_step_cls_wise_fp)
    total_fn = sum(current_step_cls_wise_fn)
    total_tp = sum(current_step_cls_wise_tp)

    overall_precision = compute_precision(total_tp, total_tn, total_fp, total_fn)
    overall_recall = compute_recall(total_tp, total_tn, total_fp, total_fn)
    overall_specificity = compute_specificity(total_tp, total_tn, total_fp, total_fn)
    micro_avg_f1 = compute_f1(overall_precision, overall_recall)
    micro_avg_mcc = compute_mcc(total_tp, total_tn, total_fp, total_fn)
    micro_avg_acc = compute_acc(total_tp, total_tn, total_fp, total_fn)
    flatten_label_list = np.concatenate(current_step_label_list)
    flatten_pred_list = np.concatenate(current_step_soft_label_list)
    micro_avg_auc = compute_auc(flatten_label_list, flatten_pred_list)

    org_f1_score = eval_result['f1_score']
    org_auc_score = eval_result['auc_score']

    cls_avg_f1 = sum(current_step_cls_wise_f1) / len(current_step_cls_wise_f1)

    # ------ record all results ------
    result_dict = {}
    result_dict["class_average_f1"] = cls_avg_f1
    result_dict["micro_avg_f1"] = micro_avg_f1
    result_dict["class_average_auc"] = cls_avg_auc
    result_dict["micro_avg_auc"] = micro_avg_auc
    result_dict["class_average_mcc"] = cls_avg_mcc
    result_dict["micro_avg_mcc"] = micro_avg_mcc
    result_dict["class_average_acc"] = cls_avg_acc
    result_dict["micro_avg_acc"] = micro_avg_acc
    result_dict["class_average_precision"] = cls_avg_precision
    result_dict["class_average_recall"] = cls_avg_recall
    result_dict["class_average_specificity"] = cls_avg_specificity
    result_dict["current_step_cls_wise_f1"] = current_step_cls_wise_f1
    result_dict["current_step_cls_wise_auc"] = current_step_cls_wise_auc
    result_dict["current_step_cls_wise_mcc"] = current_step_cls_wise_mcc
    result_dict["current_step_cls_wise_acc"] = current_step_cls_wise_acc
    result_dict["current_step_cls_wise_precision"] = current_step_cls_wise_precision
    result_dict["current_step_cls_wise_recall"] = current_step_cls_wise_recall
    result_dict["current_step_cls_wise_specificity"] = current_step_cls_wise_specificity
    result_dict["current_step_label_name_list"] = current_step_label_name_list

    result_dict["current_step_cls_wise_f1_dist"] = current_step_cls_wise_f1_dist
    result_dict["current_step_cls_wise_auc_dist"] = current_step_cls_wise_auc_dist
    result_dict["current_step_cls_wise_acc_dist"] = current_step_cls_wise_acc_dist
    result_dict["current_step_cls_wise_mcc_dist"] = current_step_cls_wise_mcc_dist
    result_dict["current_step_cls_wise_precision_dist"] = current_step_cls_wise_precision_dist
    result_dict["current_step_cls_wise_recall_dist"] = current_step_cls_wise_recall_dist
    result_dict["current_step_cls_wise_specificity_dist"] = current_step_cls_wise_specificity_dist
    result_dict["current_step_pred_threshold_dist"] = current_step_pred_threshold_dist
    
    return result_content, result_dict, cls_avg_f1

def write_to_logger(inc_step, client_index, comm_round, result_dict, logger):
    cls_avg_f1 = result_dict["class_average_f1"]
    micro_avg_f1 = result_dict["micro_avg_f1"]
    cls_avg_auc = result_dict["class_average_auc"]
    micro_avg_auc = result_dict["micro_avg_auc"]
    cls_avg_mcc = result_dict["class_average_mcc"]
    micro_avg_mcc = result_dict["micro_avg_mcc"]
    cls_avg_acc = result_dict["class_average_acc"]
    micro_avg_acc = result_dict["micro_avg_acc"]
    cls_avg_precision = result_dict["class_average_precision"]
    cls_avg_recall = result_dict["class_average_recall"]
    cls_avg_specificity = result_dict["class_average_specificity"]

    current_step_cls_wise_precision_dist = result_dict["current_step_cls_wise_precision_dist"] 
    current_step_cls_wise_recall_dist = result_dict["current_step_cls_wise_recall_dist"]
    current_step_cls_wise_specificity_dist = result_dict["current_step_cls_wise_specificity_dist"]
    current_step_pred_threshold_dist = result_dict["current_step_pred_threshold_dist"]
    current_step_cls_wise_f1_dist = result_dict["current_step_cls_wise_f1_dist"]
    current_step_cls_wise_auc_dist = result_dict["current_step_cls_wise_auc_dist"]
    current_step_cls_wise_acc_dist = result_dict["current_step_cls_wise_acc_dist"]
    current_step_cls_wise_mcc_dist = result_dict["current_step_cls_wise_mcc_dist"]

    result_content = []
     # ------ record logger ------
    logger.info("* Evaluation | inc_step: {0}, comm_round: {1}, client: {2}, fedAvg *".format(inc_step, comm_round + 1, client_index + 1))
    logger.info("* class-wise prediction threshold = {}".format(current_step_pred_threshold_dist))
    logger.info("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1 * 100, micro_avg_f1 * 100, current_step_cls_wise_f1_dist))
    logger.info("* cls-avg (macro-avg) AUC = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc * 100, micro_avg_auc * 100, current_step_cls_wise_auc_dist))
    logger.info("* cls-avg (macro-avg) MCC = {:.2f}% | micro_avg_mcc score = {:.2f}% | cls-wise MCC score = {}".format(cls_avg_mcc * 100, micro_avg_mcc * 100, current_step_cls_wise_mcc_dist))
    logger.info("* cls-avg (macro-avg) ACC = {:.2f}% | micro_avg_acc score = {:.2f}% | cls-wise ACC score = {}".format(cls_avg_acc * 100, micro_avg_acc * 100, current_step_cls_wise_acc_dist))
    logger.info("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision * 100, current_step_cls_wise_precision_dist))
    logger.info("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall * 100, current_step_cls_wise_recall_dist))
    logger.info("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity * 100, current_step_cls_wise_specificity_dist))
    # logger.info("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))
    
    result_content.append("* Evaluation | inc_step: {0}, comm_round: {1}, client: {2}, fedAvg *".format(inc_step, comm_round + 1, client_index + 1))
    result_content.append("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1 * 100, micro_avg_f1 * 100, current_step_cls_wise_f1_dist))
    result_content.append("* cls-avg (macro-avg) AUC = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc * 100, micro_avg_auc * 100, current_step_cls_wise_auc_dist))
    result_content.append("* cls-avg (macro-avg) MCC = {:.2f}% | micro_avg_mcc score = {:.2f}% | cls-wise MCC score = {}".format(cls_avg_mcc * 100, micro_avg_mcc * 100, current_step_cls_wise_mcc_dist))
    result_content.append("* cls-avg (macro-avg) ACC = {:.2f}% | micro_avg_acc score = {:.2f}% | cls-wise ACC score = {}".format(cls_avg_acc * 100, micro_avg_acc * 100, current_step_cls_wise_acc_dist))
    result_content.append("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision * 100, current_step_cls_wise_precision_dist))
    result_content.append("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall * 100, current_step_cls_wise_recall_dist))
    result_content.append("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity * 100, current_step_cls_wise_specificity_dist))
    # result_content.append("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))

    return result_content

def write_to_tensorboard(inc_step, client_index, comm_round, result_dict, tensorboard_writer):
    cls_avg_f1 = result_dict["class_average_f1"] 
    cls_avg_auc = result_dict["class_average_auc"]
    cls_avg_mcc = result_dict["class_average_mcc"]
    cls_avg_acc = result_dict["class_average_acc"]
    cls_avg_precision = result_dict["class_average_precision"]
    cls_avg_recall = result_dict["class_average_recall"]
    cls_avg_specificity = result_dict["class_average_specificity"]

    # ------ tensorboard_writer ------
    tensorboard_writer.add_scalar('class_average_f1/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_f1, comm_round)
    tensorboard_writer.add_scalar('class_average_auc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_auc, comm_round)
    tensorboard_writer.add_scalar('class_average_mcc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_mcc, comm_round)
    tensorboard_writer.add_scalar('class_average_acc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_acc, comm_round)
    tensorboard_writer.add_scalar('class_average_precision/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_precision, comm_round)
    tensorboard_writer.add_scalar('class_average_recall/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_recall, comm_round)
    tensorboard_writer.add_scalar('class_average_specificity/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_specificity, comm_round)
    # tensorboard_writer.add_scalar('org_F1_Centralized/client_{0}'.format(client_index+1), org_f1_score, comm_round)
    # tensorboard_writer.add_scalar('org_AUC_Centralized/client_{0}'.format(client_index+1), org_auc_score, comm_round)


def compute_global_predition(all_client_soft_pred_list, all_client_label_list):
    print("need to compute compute_global_predition from all_client_soft_pred_list")
    combined_soft_pred_list = []
    combined_label_list = []

    for cls_index, _ in enumerate(all_client_soft_pred_list[0]):
        cls_soft_pred_4_all_client = []
        for client_index, _ in enumerate(all_client_soft_pred_list):          
            cls_soft_pred = all_client_soft_pred_list[client_index][cls_index]
            cls_label = all_client_label_list[client_index][cls_index]
            cls_soft_pred_4_all_client.append(cls_soft_pred)
        cls_soft_pred_4_all_client_matrix = np.stack(cls_soft_pred_4_all_client)
        cls_soft_pred_4_all_client_matrix_max = np.max(cls_soft_pred_4_all_client_matrix, axis=0)
        combined_soft_pred_list.append(cls_soft_pred_4_all_client_matrix_max)
        combined_label_list.append(all_client_label_list[0][cls_index])
    return combined_soft_pred_list, combined_label_list
        

def perform_eval_for_federated_train(
        inc_step, comm_round, args, logger, tensorboard_writer, model, accelerator, device, return_feature=False, test_trainset=False):
    if "medmnist_chest" in args.dataset_name:
        if args.seperate_background_class:
            LABEL_LIST = CHESTMNIST_LABEL_LIST_W_BACKGROUND
        else:
            LABEL_LIST = CHESTMNIST_LABEL_LIST
    elif "medmnist_path" in args.dataset_name:
        LABEL_LIST = CANCER_LABEL_LIST
    elif "medmnist_derma" in args.dataset_name:
        LABEL_LIST = SKIN_LABEL_LIST
    elif "cifar10" in args.dataset_name:
        LABEL_LIST = CIFAR10_LABEL_LIST
    elif "voc2012" in args.dataset_name:
        LABEL_LIST = VOC2012_LABEL_CLASSES
    elif "xray14" in args.dataset_name:
        LABEL_LIST = XRAY14_LABEL_CLASSES
    elif "coco2017" in args.dataset_name:
        LABEL_LIST = COCO2017_LABEL_CLASSES
    else:
        raise ValueError("Something wrong with the LABEL_LIST!")
    
    model_config = model_configs[args.encoder_name]
    
    final_result_content = []
    client_avg_cls_avg_f1 = 0.
    all_client_result_dict = []
    
    inc_step_all_features = []
    inc_step_all_soft_pred = []
    inc_step_all_labels = []

    all_client_soft_pred_list = []
    all_client_label_list = []

    global_result_dict = {}

    with torch.no_grad():
        for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):
            if args.evaluate_only_first_client_flag and client_index > 0:
                continue
            task_name = task_configs[args.task_config_key]["task_name"]
            task_output_dir = os.path.join(args.output_dir, "checkpoints", "client{}_{}".format(client_index + 1, task_key))
            task_trainer_class = task_configs[args.task_config_key]["task_trainer"]
            task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)

            eval_result = task_trainer.eval(model, return_feature, test_trainset)
            if isinstance(eval_result, list):
                eval_result = eval_result[0]

            label_list = eval_result['label_list']
            soft_pred_list = eval_result['pred_list']
            all_client_soft_pred_list.append(soft_pred_list)
            all_client_label_list.append(label_list)

            if return_feature:
                inc_step_all_features.append(eval_result['all_features'])
                inc_step_all_labels.append(eval_result['all_labels'])
                inc_step_all_soft_pred.append(eval_result['all_soft_predicts'])

            hard_pred_list, pred_threshold_list = convert_probability_to_hard_prediction(args, label_list, soft_pred_list)

            # ------ get current client confusion matrix ------
            cm_matrix = compute_class_wise_confusion_matrix(hard_pred_list, label_list)

            # ------ get current client class-wise performance ------
            cls_wise_perform_dict = compute_class_wise_performance(cm_matrix, soft_pred_list, label_list)

            # ------ get current client class-wise performance evaluation for current step classes # (for incremental new classes) ------
            current_step_cls_wise_perform_dict, customer_label_index_list = compute_current_step_cls_wise_performance(
                args, LABEL_LIST, inc_step, cls_wise_perform_dict, label_list, soft_pred_list)

            # ------ summarize current client class-wise & class-average performance ------
            result_content, result_dict, cls_avg_f1 = summarize_per_client_performance(
                args, cls_wise_perform_dict, current_step_cls_wise_perform_dict, pred_threshold_list, eval_result, inc_step, LABEL_LIST, customer_label_index_list)
            client_avg_cls_avg_f1 += cls_avg_f1
            all_client_result_dict.append(result_dict)
            final_result_content.extend(result_content)

            # ------ write to logger ------
            result_content = write_to_logger(inc_step, client_index, comm_round, result_dict, logger)
            final_result_content.extend(result_content)

            # ------ write to tensorboard ------
            write_to_tensorboard(inc_step, client_index, comm_round, result_dict, tensorboard_writer)
            
            del task_trainer
            accelerate_and_free_cache(accelerator)

        if args.evaluate_only_first_client_flag:
            client_avg_cls_avg_f1 = client_avg_cls_avg_f1
            tensorboard_writer.add_scalar('average_f1/step_{0}_client_avg'.format(inc_step), client_avg_cls_avg_f1, comm_round)
        else:
            client_avg_cls_avg_f1 = client_avg_cls_avg_f1 / len(args.ordered_fcl_tasks[inc_step])
            tensorboard_writer.add_scalar('average_f1/step_{0}_client_avg'.format(inc_step), client_avg_cls_avg_f1, comm_round)

        if comm_round != -1:
            logger.info("*** Evaluation | inc_step: {}, comm_round: {} | client-avg cls-avg F1 score = {:.2f}%".format(
                    inc_step, comm_round, client_avg_cls_avg_f1 * 100))
            final_result_content.append("Evaluation | inc_step: {}, comm_round: {} | client-avg cls-avg F1 score = {:.2f}%".format(
                    inc_step, comm_round, client_avg_cls_avg_f1 * 100))
        else:
            logger.info("*** Evaluation | inc_step: {}, After federated learning finished | client-avg cls-avg F1 score = {:.2f}%".format(
                    inc_step, client_avg_cls_avg_f1 * 100))
            final_result_content.append("Evaluation | inc_step: {}, After federated learning finished | client-avg cls-avg F1 score = {:.2f}%".format(
                    inc_step, client_avg_cls_avg_f1 * 100))
            
        if args.client_specific_head and (not args.test_train_set) and args.do_test:
            combined_soft_pred_list, combined_label_list = compute_global_predition(all_client_soft_pred_list, all_client_label_list)
            global_hard_pred_list, global_pred_threshold_list = convert_probability_to_hard_prediction(args, combined_label_list, combined_soft_pred_list)
            global_cm_matrix = compute_class_wise_confusion_matrix(global_hard_pred_list, combined_label_list)
            global_cls_wise_perform_dict = compute_class_wise_performance(global_cm_matrix, combined_soft_pred_list, combined_label_list)
            current_step_global_cls_wise_perform_dict, global_customer_label_index_list = compute_current_step_cls_wise_performance(
                args, LABEL_LIST, inc_step, global_cls_wise_perform_dict, combined_label_list, combined_soft_pred_list)
            _, global_result_dict, _ = summarize_per_client_performance(
                args, global_cls_wise_perform_dict, current_step_global_cls_wise_perform_dict, global_pred_threshold_list, eval_result, inc_step, LABEL_LIST, global_customer_label_index_list)

    return final_result_content, client_avg_cls_avg_f1, all_client_result_dict, global_result_dict, inc_step_all_features, inc_step_all_soft_pred, inc_step_all_labels

