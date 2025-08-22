import argparse
import os
import random
import numpy as np
import torch
from torchvision.utils import save_image

from diffusion_model import DiffusionModel
from evaluator import evaluation_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images for test/new_test and evaluate accuracy using evaluator")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name of diffusion model (e.g., iclevr_checkpoint_210.pth)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for diffusion model (e.g., cuda, cpu). None=auto")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Sampling method")
    parser.add_argument("--timesteps", type=int, default=1000, help="Override total timesteps for DDPM sampling")
    parser.add_argument("--beta1", type=float, default=1e-4, help="Override beta1 for DDPM sampling")
    parser.add_argument("--beta2", type=float, default=0.02, help="Override beta2 for DDPM sampling")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of steps for DDIM sampling")
    parser.add_argument("--run", type=str, default="both", choices=["test", "new_test", "both"], help="Which split to run")
    parser.add_argument("--out-dir", type=str, default="images", help="Directory to save generated images and grids")
    parser.add_argument("--save-grid", action="store_true", help="Save grid PNG for each split")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000000), help="Random seed for reproducible sampling")
    return parser.parse_args()


def load_labels_from_json(dm: DiffusionModel, json_path: str):
    # Use diffusion model helper to map strings -> one-hot labels on device
    return dm.get_custom_context_from_file(json_path)


def generate_and_evaluate(dm: DiffusionModel, json_path: str, sampler: str, out_dir: str,
                          timesteps: int | None, beta1: float | None, beta2: float | None,
                          ddim_steps: int, save_grid: bool):
    os.makedirs(out_dir, exist_ok=True)
    labels = load_labels_from_json(dm, json_path)
    n_samples = labels.shape[0]

    if sampler == "ddpm":
        images, intermediates, _ = dm.sample_with_sampler(
            n_samples=n_samples,
            context=labels,
            sampler="ddpm",
            timesteps=timesteps,
            beta1=beta1,
            beta2=beta2,
        )
    else:
        # Ensure ddim_steps does not exceed training T
        try:
            T, _, _, _ = dm.get_ddpm_params_from_checkpoint(dm.file_dir, dm.checkpoint_name, dm.device)
            ddim_steps = min(ddim_steps, T)
        except Exception:
            pass
        images, intermediates, _ = dm.sample_with_sampler(
            n_samples=n_samples,
            context=labels,
            sampler="ddim",
            ddim_steps=ddim_steps,
        )

    # Save per-image RGB PNGs for visualization
    x_vis = (images + 1) / 2
    for i in range(n_samples):
        save_image(x_vis[i], os.path.join(out_dir, f"{i:03d}.png"))

    # Optional grid
    if save_grid:
        save_image(x_vis, os.path.join(out_dir, "grid.png"), nrow=8)

    # Evaluate using evaluator (which expects CUDA by design)
    if not torch.cuda.is_available():
        raise RuntimeError("evaluator uses CUDA; please run on a CUDA-enabled environment.")
    evaluator = evaluation_model()
    acc = evaluator.eval(images.cuda(), labels.cuda())
    print(f"Accuracy on {os.path.basename(json_path)}: {acc:.4f}")
    return acc


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Seed: {args.seed}")
    # set global seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dm = DiffusionModel(device=args.device, checkpoint_name=args.checkpoint_name)

    def run_split(split_name: str, json_file: str):
        split_dir = os.path.join(args.out_dir, split_name)
        generate_and_evaluate(dm,
                              json_path=json_file,
                              sampler=args.sampler,
                              out_dir=split_dir,
                              timesteps=args.timesteps,
                              beta1=args.beta1,
                              beta2=args.beta2,
                              ddim_steps=args.ddim_steps,
                              save_grid=args.save_grid)

    if args.run in ("test", "both"):
        run_split("test", "data/test.json")

    if args.run in ("new_test", "both"):
        run_split("new_test", "data/new_test.json")


