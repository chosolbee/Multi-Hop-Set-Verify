import os

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
assert WANDB_API_KEY is not None, "Please set the WANDB_API_KEY environment variable."

