"""
Interactive / CLI demo for edge-conditioned consistency-model generation.

Two modes:

* CLI:    python scripts/demo.py --model_path CKPT --image PATH_OR_URL
          python scripts/demo.py --model_path CKPT --image_dir DIR
* Gradio: python scripts/demo.py --model_path CKPT --gradio [--share]

In Gradio mode you (or, at the interview, the professor) can drop in *any*
picture grabbed from the web, tweak the Canny thresholds and the number of
sampling steps, and immediately see the [input | edge | generated] triptych.

The architecture defaults to the official class-conditional ImageNet-64
consistency model (num_channels=192, num_res_blocks=3, ...), so usually only
``--model_path`` (your trained ControlNet checkpoint) is required.
"""

import argparse
import glob
import os

import numpy as np
import torch as th

from cm import dist_util, edge_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample


# --------------------------------------------------------------------------- #
# Model loading + single-image generation
# --------------------------------------------------------------------------- #
def build_model(args):
    keys = list(model_and_diffusion_defaults().keys())
    kwargs = args_to_dict(args, keys)
    kwargs["distillation"] = True  # consistency model -> boundary-cond scalings
    model, diffusion = create_model_and_diffusion(**kwargs)

    logger.log(f"loading model from {args.model_path}")
    state = th.load(args.model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.log(
            f"load_state_dict: {len(missing)} missing / {len(unexpected)} unexpected "
            f"keys (a few buffers may be expected; investigate if large)."
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


@th.no_grad()
def generate(model, diffusion, args, pil_image, canny_low, canny_high,
             fixed_class, steps, sampler, ts_str, seed):
    """Returns a PIL triptych [input | edge | generated]."""
    edge, arr = edge_util.canny_edge(pil_image, args.image_size, canny_low, canny_high)
    edge_t = th.tensor(edge[None], device=dist_util.dev())  # (1,1,H,W)

    model_kwargs = {"edge": edge_t}
    if args.class_cond:
        cls = fixed_class if fixed_class is not None and fixed_class >= 0 else 0
        model_kwargs["y"] = th.tensor([cls], device=dist_util.dev(), dtype=th.long)

    ts = None
    if sampler == "multistep":
        ts = tuple(int(x) for x in str(ts_str).split(",")) if ts_str else (0, 106, 200)

    generator = get_generator("determ", 1, int(seed))
    sample = karras_sample(
        diffusion, model, (1, 3, args.image_size, args.image_size),
        steps=steps, model_kwargs=model_kwargs, device=dist_util.dev(),
        clip_denoised=True, sampler=sampler,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        generator=generator, ts=ts,
    )
    gen_u8 = edge_util.sample_to_uint8(sample[0].cpu().numpy())
    return edge_util.make_triptych(arr, edge, gen_u8)


# --------------------------------------------------------------------------- #
# CLI mode
# --------------------------------------------------------------------------- #
def run_cli(model, diffusion, args):
    os.makedirs(args.output, exist_ok=True)
    if args.image:
        sources = [args.image]
    elif args.image_dir:
        sources = []
        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.JPEG"):
            sources += glob.glob(os.path.join(args.image_dir, e))
        sources = sorted(set(sources))
    else:
        raise ValueError("CLI mode needs --image or --image_dir (or use --gradio)")

    for i, src in enumerate(sources):
        pil = edge_util.load_image_any(src)
        trip = generate(
            model, diffusion, args, pil, args.canny_low, args.canny_high,
            args.fixed_class, args.steps, args.sampler, args.ts, args.seed,
        )
        name = os.path.splitext(os.path.basename(str(src)))[0] or f"img{i}"
        out = os.path.join(args.output, f"{name}_triptych.png")
        trip.save(out)
        logger.log(f"saved {out}")


# --------------------------------------------------------------------------- #
# Gradio mode
# --------------------------------------------------------------------------- #
def run_gradio(model, diffusion, args):
    import gradio as gr

    # Work around a known gradio_client bug ("TypeError: argument of type 'bool'
    # is not iterable") that crashes the page when building the API schema for
    # some component configs. Make the schema walker tolerate bool nodes.
    try:
        import gradio_client.utils as _gcu

        _orig_js = _gcu._json_schema_to_python_type

        def _safe_js(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig_js(schema, defs)

        _gcu._json_schema_to_python_type = _safe_js

        _orig_gt = _gcu.get_type

        def _safe_gt(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_gt(schema)

        _gcu.get_type = _safe_gt
    except Exception:
        pass

    def infer(image, canny_low, canny_high, fixed_class, quality, seed):
        if image is None:
            return None
        pil = edge_util.load_image_any(image)
        if quality == "1-step (fast)":
            sampler, steps, ts_str = "onestep", 1, ""
        else:
            # 2-step multistep for a model trained with ~40 scales (start_scales=40),
            # matching the CD-ImageNet recipe. (ts 0,106,200/steps 201 is for the
            # 200-scale CT model and would mismatch this checkpoint.)
            sampler, steps, ts_str = "multistep", 40, "0,22,39"
        cls = int(fixed_class) if args.class_cond else -1
        trip = generate(
            model, diffusion, args, pil, int(canny_low), int(canny_high),
            cls, steps, sampler, ts_str, int(seed),
        )
        return trip

    demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Image(type="pil", label="Input image (drag any web image here)"),
            gr.Slider(0, 400, value=args.canny_low, step=1, label="Canny low threshold"),
            gr.Slider(0, 500, value=args.canny_high, step=1, label="Canny high threshold"),
            gr.Number(value=max(args.fixed_class, 0), label="ImageNet class id (0-999)"),
            gr.Radio(["1-step (fast)", "multi-step (quality)"], value="1-step (fast)",
                     label="Sampling"),
            gr.Number(value=args.seed, label="Random seed"),
        ],
        outputs=gr.Image(type="pil", label="input | canny edge | generated"),
        title="Edge-controlled Consistency Model",
        description="Drop in any image; the model generates a new image whose "
                    "structure follows its Canny edge map.",
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


def main():
    args = create_argparser().parse_args()
    # No setup_dist(): the demo is single-process inference, so we skip the
    # distributed setup (which would force CUDA_VISIBLE_DEVICES to GPU 0).
    # Pick a GPU with e.g.  CUDA_VISIBLE_DEVICES=1 python scripts/demo.py ...
    logger.configure()
    model, diffusion = build_model(args)
    if args.gradio:
        run_gradio(model, diffusion, args)
    else:
        run_cli(model, diffusion, args)


def create_argparser():
    defaults = dict(
        model_path="",
        image="",
        image_dir="",
        output="demo_out",
        gradio=False,
        share=False,
        port=7860,
        steps=1,
        sampler="onestep",
        ts="",
        seed=42,
        fixed_class=-1,
        canny_low=100,
        canny_high=200,
    )
    # Default to the official class-conditional ImageNet-64 CM + ControlNet.
    md = model_and_diffusion_defaults()
    md.update(
        dict(
            image_size=64,
            num_channels=192,
            num_res_blocks=3,
            num_head_channels=64,
            attention_resolutions="32,16,8",
            class_cond=True,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            weight_schedule="uniform",
            use_controlnet=True,
            hint_channels=1,
        )
    )
    defaults.update(md)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
