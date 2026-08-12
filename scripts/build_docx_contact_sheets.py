"""Build pairwise contact sheets for rendered manuscript QA."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE = Path(r"D:\DPC-SNN_paper_results\manuscript_20260808\qa_pages")
OUTPUT = Path(r"D:\DPC-SNN_paper_results\manuscript_20260808\qa_contact")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pages = sorted(SOURCE.glob("page-*.png"))
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
    for index in range(0, len(pages), 2):
        pair = pages[index : index + 2]
        opened = [Image.open(path).convert("RGB") for path in pair]
        thumb_width = 1180
        resized = []
        for image in opened:
            height = round(image.height * thumb_width / image.width)
            resized.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
        canvas_height = max(image.height for image in resized) + 90
        canvas = Image.new("RGB", (thumb_width * 2 + 60, canvas_height), "#D0D5DD")
        draw = ImageDraw.Draw(canvas)
        for offset, (path, image) in enumerate(zip(pair, resized)):
            x = 20 + offset * (thumb_width + 20)
            canvas.paste(image, (x, 60))
            draw.text((x, 15), path.stem.replace("page-", "Page "), font=font, fill="#17202A")
        canvas.save(OUTPUT / f"contact_{index + 1:02d}_{index + len(pair):02d}.png")


if __name__ == "__main__":
    main()
