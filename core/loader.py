from pathlib import Path

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
