import os
import pickle

import numpy as np
import swanlab
import torch
import torchvision
import transformers
from TMA import OpenVLAAttacker as _BaseOpenVLAAttacker
from straight_attack_metrics import (
    accumulate_direction_metrics,
    build_direction_log_payload,
    calculate_direction_offset_metrics,
    empty_direction_metrics,
    normalize_target_direction,
)
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast


class OpenVLAAttacker(_BaseOpenVLAAttacker):
    def __init__(self, vla, processor, save_dir="", optimizer="pgd", resize_patch=False, target_direction=None):
        super().__init__(vla, processor, save_dir=save_dir, optimizer=optimizer, resize_patch=resize_patch)
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
        args=None,
    ):
        self.val_CE_loss = []
        self.val_L1_loss = []
        self.val_ASR = []
        self.val_inner_relatived_distance = []
        self.train_CE_loss = []
        self.train_inner_avg_loss = []
        self.train_inner_relatived_distance = []
        self._reset_direction_buffers()

        patch = torch.rand(patch_size).to(self.device)
        patch.requires_grad_(True)
        patch.retain_grad()
        target_action = self.base_tokenizer(self.action_tokenizer(target_action)).input_ids[2:]
        target_action.append(2)
        target_action = list(target_action)
        target_action = torch.tensor(target_action).to(self.device)
        for idx in range(len(target_action)):
            if idx not in maskidx:
                target_action[idx] = -100
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
            torch.cuda.empty_cache()
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
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
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
                log_patch_grad = patch.grad.detach().mean().item()
                if self.optimizer == "adamW":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        optimizer.step()
                        patch.data = patch.data.clamp(0, 1)
                        optimizer.zero_grad()
                        self.vla.zero_grad()
                        torch.cuda.empty_cache()
                elif self.optimizer == "pgd":
                    if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                        patch.data = (patch.data - alpha * patch.grad.detach().sign()).clamp(0, 1)
                        self.vla.zero_grad()
                        patch.grad.zero_()
            inner_avg_loss /= innerLoop
            inner_relatived_distance /= innerLoop
            train_direction_payload = build_direction_log_payload("TRAIN", train_direction_metrics, maskidx)

            if self.optimizer == "adamW":
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()

            self.loss_buffer.append(loss.item())
            print(f"target_loss: {loss.item()}")
            if args.swanlab_project != "false":
                train_log_payload = {
                    "TRAIN_attack_loss(CE)": loss.item(),
                    "TRAIN_patch_gradient": log_patch_grad,
                    "TRAIN_LR": optimizer.param_groups[0]["lr"],
                    "TRAIN_inner_avg_loss": inner_avg_loss,
                    "TRAIN_inner_relatived_distance": inner_relatived_distance,
                }
                train_log_payload.update(train_direction_payload)
                swanlab.log(train_log_payload, step=i)
            self.train_CE_loss.append(loss.item())
            self.train_inner_avg_loss.append(inner_avg_loss)
            self.train_inner_relatived_distance.append(inner_relatived_distance)
            self._record_direction_history(train_direction_payload, split="train")
            if i % 100 == 0:
                self.plot_loss()

            if i % 100 == 0:
                avg_CE_loss = 0
                avg_L1_loss = 0
                val_num_sample = 0
                success_attack_num = 0
                val_inner_relatived_distance = 0
                val_direction_metrics = empty_direction_metrics()
                all_02other_success, all_gt_0_num, all_12other_success, all_gt_1_num, all_other20_success, all_gt_others_num = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                print("evaluating...")
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        try:
                            data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            data = next(val_iterator)
                        torch.cuda.empty_cache()
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.device)
                        attention_mask = data["attention_mask"].to(self.device)
                        input_ids = data["input_ids"].to(self.device)

                        if len(maskidx) == 1 and maskidx[0] == 6:
                            pre_ids = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
                            pre_output: CausalLMOutputWithPast = self.vla(
                                input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                                pixel_values=pre_ids.to(torch.bfloat16).to(self.device),
                                labels=labels,
                            )
                            pre_action_logits = pre_output.logits[
                                :, self.vla.vision_backbone.featurizer.patch_embed.num_patches : -1
                            ]
                            pre_action_preds = pre_action_logits.argmax(dim=2)
                            pre_action_gt = labels[:, 1:].to(self.device)
                            pre_mask = pre_action_gt > self.action_tokenizer.action_token_begin_idx

                            formulate_pre_pred = pre_action_preds[pre_mask].view(pre_action_preds[pre_mask].shape[0] // 7, 7)
                            formulate_pre_gt = pre_action_gt[pre_mask].view(pre_action_gt[pre_mask].shape[0] // 7, 7)
                            correct_index = []
                            for del_idx in range(formulate_pre_pred.shape[0]):
                                if formulate_pre_pred[del_idx][-1] == formulate_pre_gt[del_idx][-1]:
                                    correct_index.append(del_idx)
                            if len(correct_index) == 0:
                                print("No Correct in Val!")
                                continue
                            labels = labels[correct_index, :]
                            attention_mask = attention_mask[correct_index, :]
                            input_ids = input_ids[correct_index, :]
                            pixel_values = [pixel_values[k] for k in correct_index]
                        val_num_sample += labels.shape[0]
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
                            input_ids=input_ids.to(self.device),
                            attention_mask=attention_mask.to(self.device),
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
                        val_inner_relatived_distance += self.calculate_relative_distance_target(
                            continuous_actions_pred, continuous_actions_gt
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
                        avg_L1_loss += action_l1_loss.item()
                        avg_CE_loss += output.loss.item()
                    avg_L1_loss /= val_num_sample
                    avg_CE_loss /= val_num_sample
                    ASR = success_attack_num / val_num_sample
                    val_inner_relatived_distance /= val_num_sample
                    val_direction_payload = build_direction_log_payload("VAL", val_direction_metrics, maskidx)
                    if len(maskidx) == 1 and maskidx[0] == 6:
                        ASR_02other = all_02other_success / all_gt_0_num if all_gt_0_num != 0 else 0
                        ASR_12other = all_12other_success / all_gt_1_num if all_gt_1_num != 0 else 0
                        ASR_other20 = all_other20_success / all_gt_others_num if all_gt_others_num != 0 else 0
                        ALL_ASR_6 = (
                            (all_02other_success + all_12other_success) / (all_gt_0_num + all_gt_1_num)
                            if (all_gt_0_num + all_gt_1_num) != 0
                            else 0
                        )
                        if args.swanlab_project != "false":
                            val_log_payload = {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR(pred0-AllCorrect)": ASR,
                                "ASR_02other": ASR_02other,
                                "ASR_12other": ASR_12other,
                                "ASR_other20": ASR_other20,
                                "ALL_ASR_6": ALL_ASR_6,
                                "inner_relatived_distance": inner_relatived_distance,
                            }
                            val_log_payload.update(val_direction_payload)
                            swanlab.log(val_log_payload, step=i)
                    else:
                        if args.swanlab_project != "false":
                            val_log_payload = {
                                "VAL_avg_CE_loss": avg_CE_loss,
                                "VAL_avg_L1_loss": avg_L1_loss,
                                "VAL_ASR": ASR,
                                "VAL_inner_relatived_distance": val_inner_relatived_distance,
                            }
                            val_log_payload.update(val_direction_payload)
                            swanlab.log(val_log_payload, step=i)

                    if avg_L1_loss < self.min_val_avg_L1_loss:
                        self.min_val_avg_L1_loss = avg_L1_loss
                        temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                        os.makedirs(temp_save_dir, exist_ok=True)
                        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        torch.save(
                            continuous_actions_pred.detach().cpu(),
                            os.path.join(val_related_file_path, "continuous_actions_pred.pt"),
                        )
                        torch.save(
                            continuous_actions_gt.detach().cpu(),
                            os.path.join(val_related_file_path, "continuous_actions_gt.pt"),
                        )
                        modified_images = self.randomPatchTransform.denormalize(
                            modified_images[:, 0:3, :, :].detach().cpu(),
                            mean=self.mean[0],
                            std=self.std[0],
                        )
                        pil_imgs = []
                        for o in range(modified_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                            pil_imgs.append(pil_img)
                        if args.swanlab_project != "false":
                            swanlab.log({"AdvImg": [swanlab.Image(pil_img) for pil_img in pil_imgs]})
                    temp_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    torch.save(
                        continuous_actions_pred.detach().cpu(),
                        os.path.join(val_related_file_path, "continuous_actions_pred.pt"),
                    )
                    torch.save(
                        continuous_actions_gt.detach().cpu(),
                        os.path.join(val_related_file_path, "continuous_actions_gt.pt"),
                    )
                    modified_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(),
                        mean=self.mean[0],
                        std=self.std[0],
                    )
                self.val_CE_loss.append(avg_CE_loss)
                self.val_L1_loss.append(avg_L1_loss)
                self.val_ASR.append(success_attack_num / val_num_sample)
                self.val_inner_relatived_distance.append(val_inner_relatived_distance)
                self._record_direction_history(val_direction_payload, split="val")
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()
