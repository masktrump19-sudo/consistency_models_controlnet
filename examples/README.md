# Example edge-conditioned generations

Each image is a triptych: **[ input | Canny edge | generated ]**.

Produced by the edge-ControlNet consistency model (frozen ImageNet-64 backbone +
trained ControlNet branch), 2-step sampling. Filenames `c<id>_multi.png` give the
ImageNet class id used. The generated panel's structure follows the input's edge map.
