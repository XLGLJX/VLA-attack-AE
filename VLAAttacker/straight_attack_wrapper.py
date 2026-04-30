import argparse
import os
import random

import numpy as np
import swanlab
import torch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

ACTION_DIM = 7


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_target_action(maskidx, target_action_values):
    assert len(target_action_values) <= ACTION_DIM, f"targetAction length must be <= {ACTION_DIM}."
    for axis in maskidx:
        assert 0 <= axis < ACTION_DIM, f"maskidx {axis} is out of range for {ACTION_DIM}-DoF actions."

    full_target_action = np.zeros(ACTION_DIM, dtype=float)
    if target_action_values:
        full_target_action[: len(target_action_values)] = np.array(target_action_values, dtype=float)

    explicit_dims = set(range(len(target_action_values)))
    effective_maskidx = [axis for axis in maskidx if axis in explicit_dims]
    assert effective_maskidx, "No attack dimensions remain after aligning maskidx with targetAction."

    return effective_maskidx, full_target_action


def derive_target_direction(maskidx, target_action):
    target_direction = [0.0, 0.0, 0.0]
    for axis in maskidx:
        if axis in [0, 1, 2]:
            target_direction[axis] = float(target_action[axis])

    assert any(value != 0 for value in target_direction), (
        "Straight attack requires at least one non-zero xyz targetAction on masked dims to derive targetDirection."
    )
    return target_direction


def main(args):
    import sys
    import uuid

    exp_id = str(uuid.uuid4())
    sys.path.append(f"{args.server}/VLAAttacker/white_patch")
    from straight_attack import OpenVLAAttacker
    from openvla_dataloader import get_dataloader
    if "bridge_orig" in args.dataset:
        vla_path = f"{args.server}/models/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = f"{args.server}/models/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = f"{args.server}/models/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = f"{args.server}/models/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = f"{args.server}/models/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"

    set_seed(42)
    args.maskidx, args.targetAction = resolve_target_action(args.maskidx, args.targetAction)
    args.targetDirection = derive_target_direction(args.maskidx, args.targetAction)
    target = "".join(str(i) for i in args.maskidx)
    name = (
        f"straight_attack_{args.dataset}_{vla_path}_GA{args.accumulate}_lr{format(args.lr, '.0e')}"
        f"_iter{args.iter}_warmup{args.warmup}_filterGripTrainTo1{args.filterGripTrainTo1}"
        f"_target{target}_inner_loop{args.innerLoop}_geometry{args.geometry}"
        f"_patch_size{args.patch_size}_seed42-{exp_id}"
    )
    if args.swanlab_project != "false":
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=name,
            config={
                "iteration": args.iter,
                "learning_rate": args.lr,
                "attack_target": args.maskidx,
                "accumulate_steps": args.accumulate,
                "target_action": args.targetAction.tolist(),
                "target_direction": args.targetDirection,
                "note": args.swanlab_note,
            },
        )
    print(f"exp_id:{exp_id}")
    path = f"{args.server}/run/straight_attack/{exp_id}"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True, local_files_only=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)
    os.makedirs(path, exist_ok=True)
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=args.bs,
        server=args.server,
        dataset=args.dataset,
        vla_path=vla_path,
    )
    openvla_attacker = OpenVLAAttacker(
        vla,
        processor,
        path,
        optimizer="adamW",
        resize_patch=args.resize_patch,
        target_direction=args.targetDirection,
    )

    openvla_attacker.patchattack_unconstrained(
        train_dataloader,
        val_dataloader,
        num_iter=args.iter,
        target_action=args.targetAction,
        patch_size=args.patch_size,
        alpha=args.lr,
        accumulate_steps=args.accumulate,
        maskidx=args.maskidx,
        warmup=args.warmup,
        filterGripTrainTo1=args.filterGripTrainTo1,
        geometry=args.geometry,
        innerLoop=args.innerLoop,
        args=args,
    )

    print("Attack done!")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maskidx", default="0", type=list_of_ints)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--server", default="xxx", type=str, help="Prefix of the server path")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--iter", default=2000, type=int)
    parser.add_argument("--accumulate", default=1, type=int)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--tags", nargs="+", default=["innerloop50", "DoF6", "50x50", "a6000"])
    parser.add_argument(
        "--filterGripTrainTo1",
        type=str2bool,
        nargs="?",
        default=False,
        help="Remove the gripper 0 traning samples during the attack of target at grip to 0",
    )
    parser.add_argument(
        "--geometry",
        type=str2bool,
        nargs="?",
        default=True,
        help="add geometry trans to path",
    )
    parser.add_argument("--patch_size", default="3,50,50", type=list_of_ints)
    parser.add_argument("--swanlab_project", default="VLA-Attack-Pre", type=str)
    parser.add_argument("--swanlab_note", default="", type=str)
    parser.add_argument("--innerLoop", default=50, type=int)
    parser.add_argument("--dataset", default="bridge_orig", type=str)
    parser.add_argument("--resize_patch", type=str2bool, default=False)
    parser.add_argument("--targetAction", default=[1.0], type=list_of_floats)
    return parser.parse_args()


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def list_of_floats(arg):
    if arg == "":
        return []
    return list(map(float, arg.split(",")))


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = arg_parser()
    print(
        f"Paramters:\n maskidx:{args.maskidx}\n targetAction:{args.targetAction}\n"
        f" lr:{args.lr} \n server:{args.server} \n device:{args.device} "
    )
    print(f"tags:{args.tags}")
    main(args)
