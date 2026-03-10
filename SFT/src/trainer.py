from trl import SFTTrainer, SFTConfig
import torch

class CustomTrainerStage1(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states
        if isinstance(predict_embeddings, (tuple, list)):
            predict_embeddings = predict_embeddings[-1]
            
        image_out_mask = inputs["image_out_mask"]
        
        # 检查是否所有样本都没有 latent (image_out_mask 全为 0)
        # 如果是，则只计算 CE Loss，避免后续 Sim Loss 计算报错
        if image_out_mask.sum() == 0:
            loss = 0.1 * ce_loss
            return (loss, outputs) if return_outputs else loss

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        if hasattr(outputs, "inputs_embeds") and outputs.inputs_embeds is not None:
            input_embeddings = outputs.inputs_embeds
        else:
            input_embeddings = model.get_input_embeddings()(inputs["input_ids"])
            
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        mse_loss = torch.nn.functional.mse_loss(shift_predict_embeddings, gt_embeddings)
        
        # .detach() is required because we are logging a scalar tensor and HF Trainer/WandB 
        # might try to deepcopy it which fails if it's part of the graph.
        # .item() converts it to a python float which is safer for logging.
        self.log({
            "cosine_loss": cosine_loss.detach().item(),
            "mse_loss": mse_loss.detach().item()
        })
        loss = 0.1 * ce_loss + cosine_loss + 0.1 * mse_loss
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算 Stage 2 (Bridge) 的训练损失
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        predict_embeddings = outputs.hidden_states
        if isinstance(predict_embeddings, (tuple, list)):
            predict_embeddings = predict_embeddings[-1]
            
        image_out_mask = inputs["image_out_mask"]

        # 将 batch 分为 Image 分支 (前半部分) 和 Text 分支 (后半部分)
        batch_size = predict_embeddings.shape[0] // 2
        
        # --- 1. 提取 Embeddings ---
        # 我们需要从序列中提取 latent tokens。
        # mask 指示了它们的位置。
        
        # 应用与 Stage 1 相同的逻辑来对齐预测和目标
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        
        # 展平后的预测 Embeddings (整个 batch)
        # shape: [Total_Latents_In_Batch, Hidden_Size]
        all_pred_embeddings = predict_embeddings[..., :-1, :][shift_image_mask != 0].contiguous()
        
        # 展平后的目标 Embeddings
        if hasattr(outputs, "inputs_embeds") and outputs.inputs_embeds is not None:
            input_embeddings = outputs.inputs_embeds
        else:
            input_embeddings = model.get_input_embeddings()(inputs["input_ids"])
            
        all_gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask != 0].contiguous()
        
        # 分割为 Image 和 Text 两部分
        # 假设 batch 构造为 [Img_1, Img_2, ..., Txt_1, Txt_2, ...]
        # 每个样本的 latent token 数量相同 (latent_size)，所以可以直接对半切分
        
        half_len = all_pred_embeddings.shape[0] // 2
        
        img_pred = all_pred_embeddings[:half_len]
        txt_pred = all_pred_embeddings[half_len:]
        
        img_gt = all_gt_embeddings[:half_len]
        txt_gt = all_gt_embeddings[half_len:] # 如果目标相同，这部分应该等于 img_gt
        
        # --- 2. 计算损失 ---
        
        # (1) Image -> GT (L_sim_img)
        sim_loss_img_cosine = 1 - torch.nn.functional.cosine_similarity(img_gt, img_pred).mean()
        sim_loss_img_mse = torch.nn.functional.mse_loss(img_pred, img_gt)
        sim_loss_img = sim_loss_img_cosine + 0.1 * sim_loss_img_mse
        
        # (2) Text -> GT (L_sim_txt)
        sim_loss_txt_cosine = 1 - torch.nn.functional.cosine_similarity(txt_gt, txt_pred).mean()
        sim_loss_txt_mse = torch.nn.functional.mse_loss(txt_pred, txt_gt)
        sim_loss_txt = sim_loss_txt_cosine + 0.1 * sim_loss_txt_mse
        
        # (3) Consistency (L_cons): Text Pred <-> Image Pred (detached)
        # 对 Image 分支的预测做 detach，只更新 Text 分支去逼近 Image 分支
        sim_loss_cons = 1 - torch.nn.functional.cosine_similarity(img_pred.detach(), txt_pred).mean()
        
        # 权重配置 (默认来自计划: lambda=0.1, alpha=1, beta=1)
        loss_weight_ce = 0.1
        loss_weight_sim_txt = 1.0
        loss_weight_cons = 1.0
        
        total_loss = (loss_weight_ce * ce_loss) + sim_loss_img + (loss_weight_sim_txt * sim_loss_txt) + (loss_weight_cons * sim_loss_cons)
        
        # 记录日志
        self.log({
            "sim_loss_img_cosine": sim_loss_img_cosine.detach().item(),
            "sim_loss_img_mse": sim_loss_img_mse.detach().item(),
            "sim_loss_txt_cosine": sim_loss_txt_cosine.detach().item(),
            "sim_loss_txt_mse": sim_loss_txt_mse.detach().item(),
            "sim_loss_cons": sim_loss_cons.detach().item(),
            "ce_loss": ce_loss.detach().item()
        })
        
        return (total_loss, outputs) if return_outputs else total_loss

class CustomTrainerStage3(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss