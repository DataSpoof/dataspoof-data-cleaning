import re


def safe_image_name(name: str) -> str:
    """A valid Docker image name from any model name."""
    s = re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-._")
    return s or "model"
