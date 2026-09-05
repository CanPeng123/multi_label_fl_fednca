import argparse

from configs.model_configs import ALLOWED_CL_ENCODERS


def get_parser():
    # --------------- Create a parser ---------------
    parser = argparse.ArgumentParser()

    # --------------- Add arguments ---------------
    ## Model parameters
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=ALLOWED_CL_ENCODERS,
                         help="The name of the base pretrained encoder.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True,
                         help="Name of pretrained model weights to load.")
    parser.add_argument("--client_specific_head", action="store_true",
                         help="Flag for whether using client specific head for each client or using shared head for all clients.")
    parser.add_argument("--obtain_class_wise_feature", action="store_true",
                         help="Flag for whether using class-specific MLP for each class or attention block to obtain single-semantic feature.")
    parser.add_argument("--projection_type", default="class_wise_MLP", type=str,
                         help="Choose from: class_wise_MLP, Attention_learnable_random_init, Attention_learnable_etf_init, Attention_fixed_etf_init, DETR_Attention_fixed_etf_init.")
    parser.add_argument("--norm_type", type=str, default="batch_norm",
                         help="Choose from batch_norm, layer_norm, instance_norm.")

    parser.add_argument("--optimizer_mode", default="none", type=str,
                         help="The name of optimization mode. Choose from: dat, adapter, frozen, full, none.")
    parser.add_argument("--debug", type=int, default=0, help="If True, debug the code with minimum setting")

    parser.add_argument("--do_train", action="store_true", help="If True, train the model.")
    parser.add_argument("--do_test", action="store_true", help="If True, evaluate pre-trained model.")
    parser.add_argument("--test_train_set", action="store_true", help="If True, evaluate pre-trained model on the trainset.")
    parser.add_argument("--multiplicity_1", action="store_true",
                         help="If True, evaluate pre-trained model only on data with multiplicity = 1 (single label).")

    # Arguments specific to Adapters algorithm
    parser.add_argument("--adapter_reduction_factor", type=int, default=0, help="Downsampling ratio for adapter layers")

    parser.add_argument("--output_dir", type=str, required=True,
                         help="Name of output directory, where all experiment results and checkpoints are saved.")

    parser.add_argument("--do_wandb_logging", action="store_true", help="Log experiments in W&B.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Maximum number of epochs to train.")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--comm_rounds", type=int, default=20, help="Number of communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of communication rounds.")

    parser.add_argument("--data_dir", type=str, required=True, default="/path/to/dataset",
                         help="Directory where the dataset is stored")
    parser.add_argument("--json_text_folder", type=str, required=True, default="json_text_folder",
                         help="Name of the dataset json text file.")
    parser.add_argument("--json_img_folder", type=str, required=True, default="json_img_folder",
                         help="Name of the dataset json image file.")

    parser.add_argument("--dataset_name", type=str,
                         help="Choose from medmnist_derma, medmnist_chest, medmnist_path, cifar10, voc2012, xray14, coco2017.")
    parser.add_argument("--seperate_background_class", action="store_true",
                         help="For samples with no pulse (healthy sample), regard them as a new class.")

    parser.add_argument("--num_cl_tasks", type=int, default=1, help="Number of incremental tasks.")
    parser.add_argument("--num_fl_tasks", type=int, default=1, help="Number of federated tasks.")
    parser.add_argument("--partial_client_join", action="store_true",
                         help="Flag for only part of clients join training per round.")
    parser.add_argument("--num_of_client_per_round", type=int, default=-1,
                         help="Number of clients participate in each round if partial_client_join.")

    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--customer_label_list", nargs="+", help="Manually entering interested label names.")

    parser.add_argument("--loss_type", type=str, required=True, default="binary_ce",
                         help="Choose from: binary_ce, binary_ce_w_prob, focal_binary_ce, focal_binary_ce_no_alpha, cross_entropy, mse_loss, binary_ce_w_dual_reg_rejection_n_contrastive_all_PSC, binary_ce_w_dual_reg_rejection_n_contrastive_all_cosine, binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC, binary_ce_w_dual_reg_rejection_n_contrastive_topK_cosine.")
    parser.add_argument("--focal_bce_flag", action="store_true",
                         help="Flag for using focal binary loss instead of binary loss.")

    parser.add_argument("--rejection_loss_threshold", default=0.3, type=float,
                         help="The negative feature rejection loss to remove possible noisy effect from negative features.")
    parser.add_argument("--hyparam_rejection_loss", default=1, type=float,
                         help="The hyperparameter for balancing regularization loss.")
    parser.add_argument("--hyparam_contractive_loss", default=1, type=float,
                         help="The hyperparameter for balancing regularization loss.")

    parser.add_argument("--classifier_type", type=str, required=True, default="FC_Classifier",
                         help="Choose from: FC_Classifier, MultiLabel_ETF_Classifier, MultiLabel_ETF_Classifier_w_feature_normalized, Dual_Classifier.")

    parser.add_argument("--cl_method", type=str, required=True, default="direct_ft", help="Choose from: direct_ft.")
    parser.add_argument("--fl_method", type=str, required=True, default="centralized",
                         help="Choose from: centralized, standard_aggregation.")
    parser.add_argument("--evaluate_only_first_client_flag", action="store_true",
                         help="Flag for only the first client model for general FL since the received global model is the same for all clients.")

    parser.add_argument("--get_pred_probability_function", default="sigmoid",
                         help="Choose from: sigmoid, softmax, non.")
    parser.add_argument("--pred_threshold_type", type=str, required=True, default="fix_05",
                         help="Choose from fix_05, PR_curve, class_mean, max_w_tolerance.")
    parser.add_argument("--max_w_tolerance_value", default=0.02, type=float,
                         help="The tolerance to find the hard prediction using max_w_tolerance.")

    parser.add_argument("--img_augmentation", action="store_true",
                         help="Apply traditional data augmentation to image data.")
    parser.add_argument("--CenterLoss_regularization", action="store_true",
                         help="Apply Center Loss regularization to the model training.")
    parser.add_argument("--HNM_regularization", action="store_true",
                         help="Apply Hard Negative Mining (HNM) regularization to the model training.")

    parser.add_argument("--bce_w_pos_weight", action="store_true", help="pos_weight for more balanced training.")
    parser.add_argument("--reg_warm_up", action="store_true", help="Warm-up strategy for regularization terms.")
    parser.add_argument("--pos_weight_mode", type=str, default="client", choices=["global", "client"],
                         help="How to compute BCE pos_weight: global (shared across clients) or client (per-client).")

    return parser
