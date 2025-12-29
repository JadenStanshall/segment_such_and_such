# segment_such_and_such

This repo is a small semantic segmentation sandbox serving as a preamble to a larger work project. It implements a practical pipeline to generate image/mask pairs using **Segment Anything (SAM)**, then train and run a segmentation model using two different stacks: **NVIDIA TAO Toolkit** and a **PyTorch** implementation (a lightweight **U-Net**).

The work project this mirrors uses a similar segmentation approach as a **free-space / drivable-area verification layer** downstream of classical perception. In this mini-project, the foreground target class is **monitor screens** (screen vs background), purely as a convenient target for learning and iteration.

If you want to run either pipeline, start in `pytorch/README.md` or `tao/README.md` for the most relevant setup and run steps.