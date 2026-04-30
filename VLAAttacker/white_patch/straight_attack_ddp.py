import os
import pickle

import numpy as np
import swanlab
import torch
import torch.distributed as dist
import torchvision
import transformers
from TMA_ddp import OpenVLAAttacker as _BaseOpenVLAAttacker
from straight_attack_metrics import (
    DIRECTION_METRIC_KEYS,
    accumulate_direction_metrics,
    build_direction_log_payload,
    calculate_direction_offset_metrics,
    empty_direction_metrics,
    normalize_target_direction,
)
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast


class OpenVLAAttacker(_BaseOpenVLAAttacker):
    def __init__(self, *args, target_direction=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_direction = normalize_target_direction(target_direction)
        self._reset_direction_buffers()

    def _reset_direction_buffers(self):
        self.direction_history = {"train": {}, "val": {}}

    def _record_direction_history(self, payload, split):
        if not payload:
            return
        history = self.direction_history[split]
        for key, value in payload.items():
            history.setdefault(key, []).append(value)

    def _reduce_direction_metrics(self, metrics):
        reduced = self._reduce_metrics([metrics[key] for key in DIRECTION_METRIC_KEYS])
        return {key: float(reduced[idx].item()) for idx, key in enumerate(DIRECTION_METRIC_KEYS)}

    def save_info(self, path):
        super().save_info(path)
        if self.target_direction is None:
            return

        for split_history in self.direction_history.values():
            for key, values in split_history.items():
                filename = key.replace("TRAIN_", "train_").replace("VAL_", "val_") + ".pkl"
                with open(os.path.join(path, filename), "wb") as file:
                    pickle.dump(values, file)

    def patchattack_unconstrained(
        self,
        train_dataloader,
        val_dataloader,
        num_iter=5000,
        target_action=np.zeros(7),
        patch_size=[3, 50, 50],
        alpha=1 / 255,
        accumulate_steps=1,
        maskidx=[],
        warmup=20,
        filterGripTrainTo1=False,
        geometry=False,
        colorjitter=False,
        innerLoop=1,
        patch=None,
    ):
        self.val_CE_loss = []
        self.val_L1_loss = []
        self.val_ASR = []
        self.val_inner_relatived_distance = []
        self.train_CE_loss = []
        self.train_inner_avg_loss = []
        self.train_inner_relatived_distance = []
        self._reset_direction_buffers()
        if patch is None:
            patch = self._create_shared_patch()
        target_action = self.base_tokenizer(self.action_tokenizer(target_action)).input_ids[2:]
        target_action.append(2)
        target_action = list(target_action)
        target_action = torch.tensor(target_action).to(self.device)
        for idx in range(len(target_action)):
            if idx not in maskidx:
                target_action[idx] = -100
        if self.is_rank_zero():
            print(f"target_action: {target_action}")
        if self.optimizer == "adamW":
            optimizer = transformers.AdamW([patch], lr=alpha)
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=int(num_iter / accumulate_steps),
                num_cycles=0.5,
                last_epoch=-1,
            )
        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)
            if len(maskidx) == 1 and maskidx[0] == 6 and filterGripTrainTo1:
                labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            else:
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                input_ids = data["input_ids"].to(self.device)

            newlabels = []
            for j in range(labels.shape[0]):
                temp_label = labels[j].clone()
                temp_label[temp_label != -100] = target_action
                newlabels.append(temp_label.unsqueeze(0))
            newlabels = torch.cat(newlabels, dim=0)
            inner_avg_loss = 0
            inner_relatived_distance = 0
            train_direction_metrics = empty_direction_metrics()
            for inner_loop in range(innerLoop):
                if not geometry and not colorjitter:
                    modified_images = self.randomPatchTransform.paste_patch_fix(
                        pixel_values, patch, mean=self.mean, std=self.std
                    )
                else:
                    modified_images = self.randomPatchTransform.apply_random_patch_batch(
                        pixel_values, patch, mean=self.mean, std=self.std, geometry=geometry
                    )
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=modified_images.to(torch.bfloat16).to(self.device),
                    labels=newlabels,
                )
                loss = output.loss / accumulate_steps
                inner_avg_loss += loss.item()
                action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                temp_label = newlabels[:, 1:].clone()
                mask = temp_label > self.action_tokenizer.action_token_begin_idx
                continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(temp_label[mask].cpu().numpy())
                )
                continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                inner_relatived_distance += self.calculate_relative_distance_target(
                    continuous_actions_pred, continuous_actions_gt
                )
                accumulate_direction_metrics(
                    train_direction_metrics,
                    calculate_direction_offset_metrics(continuous_actions_pred, maskidx, self.target_direction),
                )
                loss.backward()
                if self.optimizer == "adamW":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        self._sync_patch_grad(patch)
                        log_patch_grad = patch.grad.detach().mean().item()
                        optimizer.step()
                        patch.data = patch.data.clamp(0, 1)
                        optimizer.zero_grad()
                        self.vla.zero_grad()
                        self._broadcast_patch(patch)
                    else:
                        log_patch_grad = patch.grad.detach().mean().item()
                elif self.optimizer == "pgd":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        self._sync_patch_grad(patch)
                        log_patch_grad = patch.grad.detach().mean().item()
                        patch.data = (patch.data - alpha * patch.grad.detach().sign()).clamp(0, 1)
                        self.vla.zero_grad()
                        patch.grad.zero_()
                        self._broadcast_patch(patch)
                    else:
                        log_patch_grad = patch.grad.detach().mean().item()
            inner_avg_loss /= innerLoop
            inner_relatived_distance /= innerLoop

            if self.optimizer == "adamW":
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()

            train_metrics = self._reduce_metrics(
                [loss.item(), log_patch_grad, optimizer.param_groups[0]["lr"], inner_avg_loss, inner_relatived_distance]
            )
            train_metrics[0] /= self.world_size
            train_metrics[1] /= self.world_size
            train_metrics[2] /= self.world_size
            train_metrics[3] /= self.world_size
            train_metrics[4] /= self.world_size
            reduced_train_direction_metrics = self._reduce_direction_metrics(train_direction_metrics)
            train_direction_payload = build_direction_log_payload("TRAIN", reduced_train_direction_metrics, maskidx)

            if self.is_rank_zero():
                self.loss_buffer.append(float(train_metrics[0].item()))
                print(f"target_loss: {train_metrics[0].item()}")
                if self.use_swanlab:
                    train_log_payload = {
                        "TRAIN_attack_loss(CE)": float(train_metrics[0].item()),
                        "TRAIN_patch_gradient": float(train_metrics[1].item()),
                        "TRAIN_LR": float(train_metrics[2].item()),
                        "TRAIN_inner_avg_loss": float(train_metrics[3].item()),
                        "TRAIN_inner_relatived_distance": float(train_metrics[4].item()),
                    }
                    train_log_payload.update(train_direction_payload)
                    swanlab.log(train_log_payload, step=i)
                self.train_CE_loss.append(float(train_metrics[0].item()))
                self.train_inner_avg_loss.append(float(train_metrics[3].item()))
                self.train_inner_relatived_distance.append(float(train_metrics[4].item()))
                self._record_direction_history(train_direction_payload, split="train")

            if i % 100 == 0 and self.is_rank_zero():
                self.plot_loss()

            if i % 100 == 0:
                avg_CE_loss_sum = 0.0
                avg_L1_loss_sum = 0.0
                val_num_sample = 0.0
                success_attack_num = 0.0
                val_inner_relatived_distance_sum = 0.0
                val_direction_metrics = empty_direction_metrics()
                all_02other_success, all_gt_0_num, all_12other_success, all_gt_1_num, all_other20_success, all_gt_others_num = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                saved_continuous_actions_pred = None
                saved_continuous_actions_gt = None
                saved_modified_images = None
                if self.is_rank_zero():
                    print("evaluating...")
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        try:
                            data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            data = next(val_iterator)
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.device)
                        attention_mask = data["attention_mask"].to(self.device)
                        input_ids = data["input_ids"].to(self.device)

                        if len(maskidx) == 1 and maskidx[0] == 6:
                            pre_ids = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
                            pre_output: CausalLMOutputWithPast = self.vla(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pre_ids.to(torch.bfloat16).to(self.device),
                                labels=labels,
                            )
                            pre_action_logits = pre_output.logits[
                                :, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1
                            ]
                            pre_action_preds = pre_action_logits.argmax(dim=2)
                            pre_action_gt = labels[:, 1:].to(self.device)
                            pre_mask = pre_action_gt > self.action_tokenizer.action_token_begin_idx

                            formulate_pre_pred = pre_action_preds[pre_mask].view(
                                pre_action_preds[pre_mask].shape[0] // 7, 7
                            )
                            formulate_pre_gt = pre_action_gt[pre_mask].view(
                                pre_action_gt[pre_mask].shape[0] // 7, 7
                            )
                            correct_index = []
                            for del_idx in range(formulate_pre_pred.shape[0]):
                                if formulate_pre_pred[del_idx, -1] == formulate_pre_gt[del_idx, -1]:
                                    correct_index.append(del_idx)
                            if len(correct_index) == 0:
                                continue
                            labels = labels[correct_index, :]
                            attention_mask = attention_mask[correct_index, :]
                            input_ids = input_ids[correct_index, :]
                            pixel_values = [pixel_values[k] for k in correct_index]
                        val_num_sample += float(labels.shape[0])
                        if not geometry and not colorjitter:
                            modified_images = self.randomPatchTransform.paste_patch_fix(
                                pixel_values, patch, mean=self.mean, std=self.std
                            )
                        else:
                            modified_images = self.randomPatchTransform.apply_random_patch_batch(
                                pixel_values, patch, mean=self.mean, std=self.std, geometry=geometry
                            )
                        newlabels = []
                        for k in range(labels.shape[0]):
                            temp_label = labels[k].clone()
                            temp_label[temp_label != -100] = target_action
                            newlabels.append(temp_label.unsqueeze(0))
                        newlabels = torch.cat(newlabels, dim=0)
                        output: CausalLMOutputWithPast = self.vla(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=modified_images.to(torch.bfloat16).to(self.device),
                            labels=newlabels,
                        )
                        action_logits = output.logits[:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = newlabels[:, 1:].to(action_preds.device)
                        mask = action_gt > self.action_tokenizer.action_token_begin_idx
                        continuous_actions_pred = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        val_inner_relatived_distance_sum += float(
                            self.calculate_relative_distance_target(continuous_actions_pred, continuous_actions_gt).item()
                        )
                        accumulate_direction_metrics(
                            val_direction_metrics,
                            calculate_direction_offset_metrics(continuous_actions_pred, maskidx, self.target_direction),
                        )
                        if len(maskidx) == 1 and maskidx[0] == 6:
                            temp_02other_success, gt_0_num, temp_12other_success, gt_1_num, temp_other20_success, gt_others_num = (
                                self.calculate_01_ASR(pred=action_preds[mask], gt=labels[:, 1:][mask])
                            )
                            all_02other_success += temp_02other_success
                            all_gt_0_num += gt_0_num
                            all_12other_success += temp_12other_success
                            all_gt_1_num += gt_1_num
                            all_other20_success += temp_other20_success
                            all_gt_others_num += gt_others_num
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                        temp_continuous_actions_pred = continuous_actions_pred.view(
                            continuous_actions_pred.shape[0] // len(maskidx), len(maskidx)
                        )
                        temp_continuous_actions_gt = continuous_actions_gt.view(
                            continuous_actions_gt.shape[0] // len(maskidx), len(maskidx)
                        )

                        for idx in range(temp_continuous_actions_pred.shape[0]):
                            if temp_continuous_actions_pred.ndim == 2:
                                flag = True
                                for idy in range(temp_continuous_actions_pred.shape[1]):
                                    if temp_continuous_actions_pred[idx, idy] != temp_continuous_actions_gt[idx, idy]:
                                        flag = False
                                if flag:
                                    success_attack_num += 1
                            else:
                                if continuous_actions_pred[idx] == continuous_actions_gt[idx]:
                                    success_attack_num += 1
                        avg_L1_loss_sum += float(action_l1_loss.item())
                        avg_CE_loss_sum += float(output.loss.item())
                        saved_continuous_actions_pred = continuous_actions_pred.detach().cpu()
                        saved_continuous_actions_gt = continuous_actions_gt.detach().cpu()
                        saved_modified_images = modified_images[:, 0:3, :, :].detach().cpu()

                reduced = self._reduce_metrics(
                    [
                        avg_CE_loss_sum,
                        avg_L1_loss_sum,
                        success_attack_num,
                        val_num_sample,
                        val_inner_relatived_distance_sum,
                        all_02other_success,
                        all_gt_0_num,
                        all_12other_success,
                        all_gt_1_num,
                        all_other20_success,
                        all_gt_others_num,
                    ]
                )
                reduced_val_direction_metrics = self._reduce_direction_metrics(val_direction_metrics)
                global_val_num_sample = max(float(reduced[3].item()), 1.0)
                avg_CE_loss = float(reduced[0].item()) / global_val_num_sample
                avg_L1_loss = float(reduced[1].item()) / global_val_num_sample
                ASR = float(reduced[2].item()) / global_val_num_sample
                val_inner_relatived_distance = float(reduced[4].item()) / global_val_num_sample
                val_direction_payload = build_direction_log_payload("VAL", reduced_val_direction_metrics, maskidx)

                if self.is_rank_zero():
                    if len(maskidx) == 1 and maskidx[0] == 6:
                        ASR_02other = float(reduced[5].item()) / float(reduced[6].item()) if reduced[6].item() != 0 else 0.0
                        ASR_12other = float(reduced[7].item()) / float(reduced[8].item()) if reduced[8].item() != 0 else 0.0
                        ASR_other20 = float(reduced[9].item()) / float(reduced[10].item()) if reduced[10].item() != 0 else 0.0
                        denom = float(reduced[6].item() + reduced[8].item())
                        ALL_ASR_6 = float(reduced[5].item() + reduced[7].item()) / denom if denom != 0 else 0.0
                        if self.use_swanlab:
                            val_log_payload = {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR(pred0-AllCorrect)": ASR,
                                "ASR_02other": ASR_02other,
                                "ASR_12other": ASR_12other,
                                "ASR_other20": ASR_other20,
                                "ALL_ASR_6": ALL_ASR_6,
                                "VAL_inner_relatived_distance": val_inner_relatived_distance,
                            }
                            val_log_payload.update(val_direction_payload)
                            swanlab.log(val_log_payload, step=i)
                    else:
                        if self.use_swanlab:
                            val_log_payload = {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR": ASR,
                                "VAL_inner_relatived_distance": val_inner_relatived_distance,
                            }
                            val_log_payload.update(val_direction_payload)
                            swanlab.log(val_log_payload, step=i)

                    if avg_L1_loss < self.min_val_avg_L1_loss and saved_continuous_actions_pred is not None:
                        self.min_val_avg_L1_loss = avg_L1_loss
                        temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                        os.makedirs(temp_save_dir, exist_ok=True)
                        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        torch.save(
                            saved_continuous_actions_pred,
                            os.path.join(val_related_file_path, "continuous_actions_pred.pt"),
                        )
                        torch.save(
                            saved_continuous_actions_gt,
                            os.path.join(val_related_file_path, "continuous_actions_gt.pt"),
                        )
                        modified_images = self.randomPatchTransform.denormalize(
                            saved_modified_images, mean=self.mean[0], std=self.std[0]
                        )
                        pil_imgs = []
                        for o in range(modified_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                            pil_imgs.append(pil_img)
                        if self.use_swanlab:
                            swanlab.log({"AdvImg": [swanlab.Image(pil_img) for pil_img in pil_imgs]})
                    temp_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    if saved_continuous_actions_pred is not None:
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        torch.save(
                            saved_continuous_actions_pred,
                            os.path.join(val_related_file_path, "continuous_actions_pred.pt"),
                        )
                        torch.save(
                            saved_continuous_actions_gt,
                            os.path.join(val_related_file_path, "continuous_actions_gt.pt"),
                        )
                    self.val_CE_loss.append(avg_CE_loss)
                    self.val_L1_loss.append(avg_L1_loss)
                    self.val_ASR.append(ASR)
                    self.val_inner_relatived_distance.append(val_inner_relatived_distance)
                    self._record_direction_history(val_direction_payload, split="val")
                    self.save_info(path=self.save_dir)

    @staticmethod
    def _attack_entry(rank, instance_params, world_size):
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        instance = OpenVLAAttacker(**instance_params)
        instance.attack_ddp(rank, world_size)
