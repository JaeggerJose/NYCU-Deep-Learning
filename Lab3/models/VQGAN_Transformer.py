import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
    def load_transformer_checkpoint(self, load_ckpt_path):
        checkpoint = torch.load(load_ckpt_path, map_location='cpu')
        self.transformer.load_state_dict(checkpoint['model_state_dict'])


    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # x: input image, shape (B, 3, H, W)
        # z_hat = E(x), zq = q(z_hat) = argmin_{z} ||z_hat(i, j) - z(k)||^2 = codebook_mapping
        zq, latent, q_loss = self.vqgan.encode(x) # 從VQGAN encoder會return codebook_mapping, codebook_indices, q_loss
        # zq = codebook_mapping (quantized features)
        # latent = codebook_indices (discrete token indices)
        return latent, zq
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return lambda ratio: (math.cos(math.pi * ratio) + 1.0) / 2.0
        elif mode == "square":
            return lambda ratio: 1 - ratio ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        # x: input image, shape (B, 3, H, W)
        # encode x to zq and latent
        _, z_indices = self.encode_to_z(x)  # z_indices: shape could be (B, H, W) -> flatten to (B, num_image_tokens)
        
        # Ensure z_indices is properly flattened to (B, num_image_tokens) and is long type
        if len(z_indices.shape) > 2:
            z_indices = z_indices.flatten(1)
        z_indices = z_indices.long()
        
        if z_indices.shape[1] != self.num_image_tokens:
            z_indices = z_indices[:, :self.num_image_tokens]
        z_indices = torch.clamp(z_indices, 0, self.mask_token_id - 1)
        
        # get mask schedule
        ratio = torch.rand(z_indices.shape[0], device=z_indices.device)  # random batch ratios
        # create mask for z_indices
        # Apply gamma function to each ratio and expand to match z_indices shape
        mask_ratios = torch.tensor([self.gamma(r.item()) for r in ratio], device=z_indices.device)  # shape (B,)
        mask_ratios = mask_ratios.unsqueeze(1).expand(-1, z_indices.shape[1])  # shape (B, num_image_tokens)
        # create a boolean mask based on the gamma function
        mask = torch.rand_like(z_indices, dtype=torch.float) < mask_ratios  # mask: shape (B, num_image_tokens), boolean mask
        z_indices[mask] = self.mask_token_id  # replace masked tokens with mask token id
        # z_indices: shape (B, num_image_tokens), with masked tokens replaced by mask_token_id
        # feed z_indices to transformer
        logits = self.transformer(z_indices)  # logits: shape (B, num_image_tokens, num_codebook_vectors)
        # Return raw logits for loss calculation (softmax will be applied in cross_entropy)
        return logits, z_indices

    @torch.no_grad()
    def inpainting(self, step, total_iter):
        # Step 1: 計算 mask ratio（從 gamma 函數動態調整）
        ratio = step / total_iter
        mask_ratio = self.gamma(ratio)

        # Step 2: 對目前的 token 預測 logits
        logits = self.transformer(self.z_indices_predict)

        # Step 3: 計算每個位置的 softmax confidence + 對應 token
        prob_logits = torch.softmax(logits / self.choice_temperature, dim=-1)
        confidence, z_indices_predict = torch.max(prob_logits, dim=-1)

        # Step 4: 只在目前還是 mask 的位置中，選 confidence 最低的 tokens 進行 mask
        flat_confidence = confidence.view(-1)
        current_mask = self.mask_bc.view(-1)
        candidates = current_mask.nonzero(as_tuple=True)[0]  # 還沒確定的 token 位置

        num_to_mask = int(mask_ratio * flat_confidence.numel())
        sorted_candidates = candidates[torch.argsort(flat_confidence[candidates])]
        indices_to_mask = sorted_candidates[:num_to_mask]

        # Step 5: 更新遮罩（累積遮罩）
        new_mask = self.mask_bc.clone()
        new_mask.view(-1)[...] = False  # 清空
        new_mask.view(-1)[indices_to_mask] = True  # 只留下這輪需要繼續預測的

        # Step 6: 在 masked 區域填入 mask token；其他位置填入預測 token
        z_indices_predict = torch.where(
            new_mask,  # 需要繼續預測的
            self.mask_token_id,
            z_indices_predict  # 已經信心夠高的 token
        )

        # Step 7: 保留原始圖像中不需要修復的區域
        z_indices_predict = torch.where(
            ~self.original_mask,  # 非 inpaint 區域，保留原圖 token
            self.original_z_indices,
            z_indices_predict
        )

        # Step 8: 更新內部狀態
        self.z_indices_predict = z_indices_predict
        self.mask_bc = new_mask
        return z_indices_predict, new_mask

    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}