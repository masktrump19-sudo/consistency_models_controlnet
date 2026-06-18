"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    if args.use_controlnet:
        sample_with_edges(args, model, diffusion, ts)
        dist.barrier()
        logger.log("sampling complete")
        return

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def sample_with_edges(args, model, diffusion, ts):
    """
    Edge-conditional sampling: read images (or edge maps) from ``--edge_dir``,
    extract a binary Canny edge map from each, generate an image conditioned on
    it, and save ``[input | edge | generated]`` triptychs + a .npz for metrics.
    """
    import glob

    from cm import edge_util

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.JPEG")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.edge_dir, e))
        files += glob.glob(os.path.join(args.edge_dir, "**", e), recursive=True)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"no images found in --edge_dir={args.edge_dir}")

    rank, world = dist.get_rank(), dist.get_world_size()
    files = files[rank::world]
    logger.log(f"[rank {rank}] conditioning on {len(files)} images")

    out_dir = logger.get_dir()
    grid_dir = os.path.join(out_dir, "triptychs")
    os.makedirs(grid_dir, exist_ok=True)

    generator = get_generator(args.generator, max(len(files), 1), args.seed)
    all_gen, all_edge = [], []
    bs = args.batch_size
    for start in range(0, len(files), bs):
        batch_files = files[start : start + bs]
        edges, crops = [], []
        for f in batch_files:
            pil = edge_util.load_image_any(f)
            edge, arr = edge_util.canny_edge(
                pil, args.image_size, args.canny_low, args.canny_high
            )
            edges.append(edge)
            crops.append(arr)
        edge_t = th.tensor(np.stack(edges), device=dist_util.dev())
        model_kwargs = {"edge": edge_t}
        if args.class_cond:
            if args.fixed_class >= 0:
                y = th.full(
                    (len(batch_files),), args.fixed_class,
                    device=dist_util.dev(), dtype=th.long,
                )
            else:
                y = th.randint(
                    0, NUM_CLASSES, (len(batch_files),), device=dist_util.dev()
                )
            model_kwargs["y"] = y

        sample = karras_sample(
            diffusion, model,
            (len(batch_files), 3, args.image_size, args.image_size),
            steps=args.steps, model_kwargs=model_kwargs, device=dist_util.dev(),
            clip_denoised=args.clip_denoised, sampler=args.sampler,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            s_churn=args.s_churn, s_tmin=args.s_tmin, s_tmax=args.s_tmax,
            s_noise=args.s_noise, generator=generator, ts=ts,
        )
        sample_np = sample.cpu().numpy()
        for j, f in enumerate(batch_files):
            gen_u8 = edge_util.sample_to_uint8(sample_np[j])
            trip = edge_util.make_triptych(crops[j], edges[j], gen_u8)
            name = os.path.splitext(os.path.basename(f))[0]
            trip.save(os.path.join(grid_dir, f"{name}_triptych.png"))
            all_gen.append(gen_u8)
            all_edge.append((np.clip(edges[j][0], 0, 1) * 255).astype(np.uint8))
        logger.log(f"sampled {start + len(batch_files)}/{len(files)}")

    if all_gen:
        np.savez(
            os.path.join(out_dir, f"edge_samples_rank{rank}.npz"),
            samples=np.stack(all_gen), edges=np.stack(all_edge),
        )
    logger.log(f"saved triptychs to {grid_dir}")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        edge_dir="",
        canny_low=100,
        canny_high=200,
        fixed_class=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
