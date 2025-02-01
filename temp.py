import os
import torch


if __name__=="__main__":
    path_ = "/home/zhangtonghe/dppo/log/gym/gym-finetune/ant-medium-expert-v0_ppo_reflow_mlp_ta4_td20_flowppo_2k/2025-01-31_11-18-02_42/checkpoint/state_0.pt"
    path_save ="/home/zhangtonghe/dppo/log/gym/gym-pretrain/ant-medium-expert-v0_pre_reflow_mlp_ta4_td20/2024-12-27_11-02-30_42/checkpoint/best.pt"
    data = torch.load(path_, weights_only=True)
    
    print(f"""data.keys()={data["model"].keys()}""")