"""MS-COCOAI: AI-generated image detection and generator attribution.

Module entry points:
  python -m mscocoai.train        train one model (binary / attribution / multitask)
  python -m mscocoai.evaluate     evaluate a checkpoint on a split, write reports
  python -m mscocoai.clip_probe   extract CLIP embeddings / train the linear probe
  python -m mscocoai.ensemble     logit-average several checkpoints and report
  python -m mscocoai.drive_upload upload / download checkpoints via Drive (no admin)
"""
__version__ = "1.0.0"
