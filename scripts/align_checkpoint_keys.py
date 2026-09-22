import os
import torch

def align():
    ckpt_path = "checkpoints/nexoralm_pretrain_gate_ckpt.pt"
    if not os.path.exists(ckpt_path):
        print(f"{ckpt_path} not found.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace(".norm1.", ".input_norm.").replace(".norm2.", ".post_attention_norm.")
        new_sd[new_k] = v
    ckpt["model_state_dict"] = new_sd
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, "checkpoints/nexoralm_base.pt")
    print("Aligned keys in nexoralm_pretrain_gate_ckpt.pt and created nexoralm_base.pt")

if __name__ == "__main__":
    align()
