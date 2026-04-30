import torch
import os
import numpy as np
import swanlab
import argparse
import random
import uuid
import torch.distributed as dist

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_exp_id():
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        exp_id = str(uuid.uuid4())
        print(f"Generated exp_id on rank 0: {exp_id}")
    else:
        exp_id = None
    exp_id_list = [exp_id]
    dist.broadcast_object_list(exp_id_list, src=0)
    return exp_id_list[0]

def main(args):
    # Initialize DDP first
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    pwd = os.getcwd()
    exp_id = str(get_exp_id())

    if "bridge_orig" in args.dataset:
        vla_path = f"{pwd}/models/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = f"{pwd}/models/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = f"{pwd}/models/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = f"{pwd}/models/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = f"{pwd}/models/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"

    set_seed(42)
    rank = int(os.environ.get("RANK", 0))
    target = ''.join(str(i) for i in args.maskidx)

    name = f"{args.dataset}_{vla_path}_GA{args.accumulate}_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_target{target}_inner_loop{args.innerLoop}_geometry{args.geometry}_patch_size{args.patch_size}_seed42-{exp_id}"

    if args.swanlab_project != "false" and rank == 0:
        swanlab.init(project=args.swanlab_project, experiment_name=name, config={
            "iteration": args.iter, "learning_rate": args.lr,
            "attack_target": args.maskidx, "accumulate_steps": args.accumulate
        })

    print(f"exp_id:{exp_id}")
    path = f"{pwd}/run/TMA/{exp_id}"
    os.makedirs(path, exist_ok=True)

    import sys
    sys.path.append(f"{pwd}/VLAAttacker/white_patch")
    from TMA_ddp import OpenVLAAttacker

    world_size = dist.get_world_size()
    instance_params = {
        "vla_path": vla_path,
        "dataset_name": args.dataset,
        "save_dir": path,
        "resize_patch": args.resize_patch,
        "patch_size": args.patch_size,
        "lr": args.lr,
        "bs": args.bs,
        "warmup": args.warmup,
        "num_iter": args.iter,
        "maskidx": args.maskidx,
        "innerLoop": args.innerLoop,
        "geometry": args.geometry,
        "use_swanlab": args.swanlab_project != "false",
        "target_action": args.targetAction,
        "accumulate_steps": args.accumulate,
        "server": pwd,
        "filter_grip_train_to1": args.filterGripTrainTo1,
    }

    OpenVLAAttacker._attack_entry(rank, instance_params, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maskidx", nargs='+', type=int, default=[0])
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--server", type=str, default=os.getcwd())
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iter", type=int, default=2000)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--filterGripTrainTo1", type=str, default="false")
    parser.add_argument("--geometry", type=str, default="true")
    parser.add_argument("--patch_size", type=str, default="3,50,50")
    parser.add_argument("--swanlab_project", type=str, default="VLA-Attack")
    parser.add_argument("--innerLoop", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="libero_spatial_no_noops")
    parser.add_argument("--targetAction", type=int, default=0)
    parser.add_argument("--resize_patch", type=str, default="false")

    args = parser.parse_args()
    args.geometry = args.geometry.lower() == "true"
    args.filterGripTrainTo1 = args.filterGripTrainTo1.lower() == "true"
    args.resize_patch = args.resize_patch.lower() == "true"
    args.patch_size = [int(x) for x in args.patch_size.split(',')]

    print("Paramters:")
    print(f" maskidx:{args.maskidx}\n lr:{args.lr} \n server:{args.server} \n device:{args.device} ")
    print(f"tags:{args.tags.split(',')}")

    main(args)
    dist.destroy_process_group()
