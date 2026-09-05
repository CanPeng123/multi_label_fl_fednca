# FedNCA-ML

Code for **"Neural Collapse-Inspired Multi-Label Federated Learning under Label-Distribution Skew"**
([arXiv:2509.12544](https://arxiv.org/abs/2509.12544))

## Running

Each script under `run_bash/` launches one FL training run via
[`accelerate`](https://github.com/huggingface/accelerate):

- `fednca.sh` — the full FedNCA-ML method: ETF classifier + the class-wise cross-attention
  projection module (LADM) + both NC-inspired regularization losses
- `fednca_etf_cls_wo_ladm.sh` — ablation baseline: ETF classifier alone, without LADM

Example:

```bash
export CODE_PTH=/path/to/this/repo
export DATA_PTH=/path/to/your/dataset/root
bash run_bash/fednca.sh
```

Key arguments (see `utils/arg_parser.py` for the full list):

- `--fl_method standard_aggregation` with `--classifier_type MultiLabel_ETF_Classifier`
  — the FedNCA-ML method
- `--loss_type binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC` — BCE + rejection
  regularization + prototype-contrastive regularization (the two NC-inspired regularizers;
  their weights are controlled independently via `--hyparam_rejection_loss` and
  `--hyparam_contractive_loss`)
- `--obtain_class_wise_feature --projection_type DETR_Attention_fixed_etf_init` — enables
  the class-wise cross-attention projection module (LADM)

To evaluate a trained checkpoint, swap `--do_train` for `--do_test` and set
`--exp_name` to the checkpoint directory produced under `./logs` by the training run.

## Citation

```bibtex
@article{peng2025fednca,
  title   = {Neural Collapse-Inspired Multi-Label Federated Learning under Label-Distribution Skew},
  author  = {Peng, Can and Liu, Yuyuan and Yang, Yingyu and Saha, Pramit and Yang, Qianye and Noble, J. Alison},
  journal = {arXiv preprint arXiv:2509.12544},
  year    = {2025}
}
```
