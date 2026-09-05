import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from typing import Dict
from torch.optim import AdamW
import numpy as np
from transformers import get_polynomial_decay_schedule_with_warmup


class TaskTrainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kl_criterion = kl_loss

    def train(self, model, task_key) -> (float, Dict):
        if not os.path.isdir(self.task_output_dir):
            os.makedirs(self.task_output_dir, exist_ok=True)

        model = self.accelerator.prepare(model)

        optimizer = self.create_optimizer(model, self.args.optimizer_mode)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_training_steps=self.max_steps,
            lr_end=0,
            power=1,
        )

        model.zero_grad()
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)

        loader = self.train_dataloader
        for epoch in range(self.local_epochs):
            all_data = []
            
            model.train()
            num_batches = 0

            for step, batch in enumerate(tqdm(loader, desc="Training local epoch {}/{}".format(epoch + 1, self.local_epochs))):
                if self.args.debug > 0 and step > self.args.debug:
                    break
                loss = self.train_step(model, step, batch, optimizer, scheduler, hooks=None, epoch=epoch)

        self.accelerator.wait_for_everyone()

        del optimizer, scheduler
        unwrapped_model = self.accelerator.unwrap_model(model)

        torch.cuda.empty_cache()
        self.accelerator.free_memory()

        return loss, unwrapped_model

    def eval_one_loader(self, model, loader, return_feature=False):
        eval_result = {}
        samples_seen = 0

        model.eval()
        model = self.accelerator.prepare(model)
        
        total_batch = 0
        f1_score = 0
        auc_score = 0
        final_label_list = []
        final_pred_list = []
        if return_feature:
            all_features = []
            all_labels = []
            all_soft_predicts = []

        for step, batch in enumerate(tqdm(loader, desc=f"Evaluating on {self.task_key} test set")):
            if self.args.debug > 0 and step > self.args.debug:
                break
            output = self.forward_pass(model, batch, do_eval=True)
            logits = output[1]
            latent_feature = output[0]


            if self.obtain_class_wise_feature:
                latent_feature = latent_feature.permute(1, 0, 2)  # [B, 10, 512]
            target = batch["target_scores"].to(self.device)  # one-hot label
            logits, target = self.accelerator.gather((logits, target))
            if step == len(loader) - 1:  # for the last batch
                logits = logits[: len(loader.dataset) - samples_seen]
                target = target[: len(loader.dataset) - samples_seen]
                latent_feature = latent_feature[: len(loader.dataset) - samples_seen]
            else:
                samples_seen += target.shape[0]

            if return_feature:
                all_features.extend(f for f in latent_feature.cpu())
                all_labels.extend([l for l in target.cpu()])
            
            if return_feature:
                label_list, soft_pred_list, probabilities = self.get_prediction_probability(logits, target, return_raw_pred=True) 
                all_soft_predicts.extend(p for p in probabilities.cpu())
            else:
                label_list, soft_pred_list = self.get_prediction_probability(logits, target, return_raw_pred=False)  # Prediction and label list for each class 

            if step == 0:
                final_label_list = label_list
                final_pred_list = soft_pred_list
            else:
                for cls_index, cls_wise_label in enumerate(final_label_list):
                    final_label_list[cls_index] = np.concatenate((final_label_list[cls_index], label_list[cls_index]))
                    final_pred_list[cls_index] = np.concatenate((final_pred_list[cls_index], soft_pred_list[cls_index]))

            f1, auc = self.compute_f1_score_with_logits(logits, target)
            f1_score += f1
            auc_score += auc
            total_batch += 1
        
        f1_score = (f1_score / total_batch) * 100.0
        auc_score = (auc_score / total_batch) * 100.0 

        eval_result['label_list'] = final_label_list
        eval_result['pred_list'] = final_pred_list
        eval_result['f1_score'] = f1_score
        eval_result['auc_score'] = auc_score
        if return_feature:
            eval_result['all_features'] = all_features
            eval_result['all_labels'] = all_labels
            eval_result['all_soft_predicts'] = all_soft_predicts
        

        model.train()
        torch.cuda.empty_cache()
        self.accelerator.free_memory()

        return eval_result
        
    def eval(self, model, return_feature=False, test_trainset=False):
        if 'nlvr2' in self.task_key:
            loader = self.nlvr_val_dataloader
        else:
            if test_trainset:
                loader = self.train_dataloader
            else:
                loader = self.test_dataloader

        # set local adapter
        if "dat" in self.args.optimizer_mode:  # dat for FedDat
            model.activate_gating()
        elif "adapter" in self.args.optimizer_mode:
            model.set_active_adapter('adapter')
        
        eval_score = self.eval_one_loader(model, loader, return_feature)
        
        if 'dat' in self.args.optimizer_mode:
            model.deactivate_gating()
            model.set_active_adapter('adapter_0')
            eval_score_0 = self.eval_one_loader(model, loader)

            model.deactivate_gating()
            model.set_active_adapter('adapter_1')
            eval_score_1 = self.eval_one_loader(model, loader)
            return [eval_score, eval_score_0, eval_score_1]
        return eval_score
       

    def forward_pass(self, model, batch, do_eval: bool = False) -> tuple:
        inputs = self.batch2inputs_converter(batch)

        if "albef" in self.args.encoder_name and not do_eval:
            inputs["train"] = True

        if do_eval is True:
            with torch.no_grad():
                output = model(client_key=self.client_key, **inputs)
        else:
            output = model(client_key=self.client_key, **inputs)
        return output

    def train_step(
        self,
        model,
        step,
        batch,
        optimizer=None,
        scheduler=None,
        hooks=None,
        epoch=None,
    ):
        if isinstance(batch, dict) and "target_scores" in batch.keys():
            target = batch["target_scores"].to(self.device)
            target_labels = [torch.nonzero(row, as_tuple=True)[0].tolist() for row in target]

        if "dat" in self.args.optimizer_mode:
            loss_0 = 0.
            return loss_0
        else:
            if "adapter" in self.args.optimizer_mode:
                model.set_active_adapter('adapter') #model.module.set_active_adapter('adapter')
            # model.learnable_text_embeddings.requires_grad= True
            logits = self.forward_pass(model, batch, do_eval=False)

            loss_ce = self.loss_criterion(logits, target) * target.shape[1]
            loss=loss_ce

            self.accelerator.backward(loss)
            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
  
            return loss
          
    def create_optimizer(self, model, mode='full'):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay)) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.adam_epsilon,
            betas=(0.9, 0.98),
        )
        return optimizer

def kl_loss(output, target, temp=3):
    if output.shape[-1]>3000:
        p = F.log_softmax(output / temp, dim=-1)
        q = F.softmax(target / temp, dim=-1)
    else:
        p = F.log_softmax(output / temp, dim=1)
        q = F.softmax(target / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean")
    l_kl = l_kl * temp**2
    return l_kl
