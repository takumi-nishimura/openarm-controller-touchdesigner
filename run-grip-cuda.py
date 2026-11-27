# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "flash-linear-attention==0.4.2",
#     "mujoco==3.17.0",
#     "openarm-control==0.5.0",
#     "openarm-mujoco==2.3.0",
#     "pillow>=10",
#     "torch==2.9.1",
#     "torchvision==0.24.1",
#     "transformers==5.19.0",
# ]
# ///

"""Run the Grip benchmark with the CUDA decision runtime dependencies."""

from evaluation.grip.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
