from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path("sample-data").resolve()
    for device, offset in (("DeviceA", 0), ("DeviceB", 35)):
        folder = root / device
        folder.mkdir(parents=True, exist_ok=True)
        for index, level in enumerate((45, 105, 175), start=1):
            image = Image.new("RGB", (640, 480), (level + offset, level + offset, level + offset))
            draw = ImageDraw.Draw(image)
            draw.ellipse((220, 100, 420, 360), fill=(180 + offset // 2, 135, 115))
            draw.rectangle((470, 40, 620, 190), fill=(255, 245, 225))
            image.save(folder / f"scene-{index}.jpg", quality=95)
    print(root)


if __name__ == "__main__":
    main()
