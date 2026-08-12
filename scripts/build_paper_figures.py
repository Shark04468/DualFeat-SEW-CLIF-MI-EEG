"""Build paper-facing figures from the sealed DPC-SNN evidence package."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont, ImageOps


INK = "#17202A"
BLUE = "#2457C5"
BLUE_FILL = "#EAF1FF"
GREEN = "#26734D"
GREEN_FILL = "#E9F7EF"
RED = "#B53A32"
RED_FILL = "#FCEDEA"
GOLD = "#A66D00"
GOLD_FILL = "#FFF4D6"
GREY = "#667085"
LIGHT = "#F7F8FA"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def multiline_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    *,
    size: int = 30,
    bold: bool = False,
    fill: str = INK,
    spacing: int = 8,
) -> None:
    x0, y0, x1, y1 = xy
    f = font(size, bold)
    box = draw.multiline_textbbox((0, 0), text, font=f, spacing=spacing, align="center")
    width, height = box[2] - box[0], box[3] - box[1]
    draw.multiline_text(
        ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2),
        text,
        font=f,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    *,
    outline: str,
    fill: str,
    size: int = 28,
    bold: bool = False,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=4)
    multiline_center(draw, xy, text, size=size, bold=bold)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = INK,
    width: int = 6,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    x1, y1 = end
    x0, y0 = start
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        points = [(x1, y1), (x1 - direction * 18, y1 - 12), (x1 - direction * 18, y1 + 12)]
    else:
        direction = 1 if y1 >= y0 else -1
        points = [(x1, y1), (x1 - 12, y1 - direction * 18), (x1 + 12, y1 - direction * 18)]
    draw.polygon(points, fill=fill)


def architecture(output: Path) -> None:
    image = Image.new("RGB", (2600, 1320), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "DualFeat-SEW-CLIF for motor-imagery EEG", font=font(52, True), fill=INK)
    draw.text(
        (70, 112),
        "Frozen heterogeneous representations followed by matched continuous or spiking dynamics",
        font=font(28),
        fill=GREY,
    )

    box(draw, (60, 475, 310, 805), "EEG trial\n-1 to 4 s\n250 Hz", outline=INK, fill=LIGHT, size=31, bold=True)
    box(
        draw,
        (370, 430, 680, 850),
        "Shared preprocessing\n\nCAR\nbaseline -1 to 0 s\ntraining-session gain\nclip at 12",
        outline=BLUE,
        fill=BLUE_FILL,
        size=27,
    )
    arrow(draw, (310, 640), (370, 640), fill=BLUE)

    box(draw, (770, 260, 1110, 555), "ATCNet branch\n\ntemporal-attention\nsequence\n18 x 32", outline=BLUE, fill=BLUE_FILL, size=29, bold=True)
    box(draw, (770, 735, 1110, 1030), "FBCNet branch\n\n9-band spatial filters\nlog-variance sequence\n4 x 288", outline=GOLD, fill=GOLD_FILL, size=29, bold=True)
    arrow(draw, (680, 580), (770, 410), fill=BLUE)
    arrow(draw, (680, 700), (770, 880), fill=GOLD)

    box(draw, (1190, 260, 1490, 555), "Linear projection\n32 -> 64\n\na(t), 18 steps", outline=BLUE, fill=BLUE_FILL, size=28)
    box(draw, (1190, 735, 1490, 1030), "Linear projection\n288 -> 64\n\ninterpolate 4 -> 18", outline=GOLD, fill=GOLD_FILL, size=28)
    arrow(draw, (1110, 410), (1190, 410), fill=BLUE)
    arrow(draw, (1110, 880), (1190, 880), fill=GOLD)

    box(
        draw,
        (1580, 410, 1940, 880),
        "Interaction fusion\n\n[a, f, a x f, |a - f|]\nLayerNorm -> Linear\nGELU + dropout\n\n+ 0.5(a + f)\nLayerNorm\n\n18 x 64",
        outline=GREEN,
        fill=GREEN_FILL,
        size=27,
        bold=True,
    )
    arrow(draw, (1490, 410), (1580, 545), fill=BLUE)
    arrow(draw, (1490, 880), (1580, 745), fill=GOLD)

    box(
        draw,
        (2040, 210, 2500, 1110),
        "Signed event decoder\n\n1 x 1 current encoder\npositive / negative populations\n\nheterogeneous CLIF stem\ndecays: 0.65, 0.90, 0.975\n\n2 x causal depthwise\nSEW-CLIF residual blocks\n(dilation 1, 2)\n\nfinal CLIF re-spiking\n\nspike + membrane mean/std\n96-D readout -> class",
        outline=RED,
        fill=RED_FILL,
        size=27,
        bold=True,
    )
    arrow(draw, (1940, 645), (2040, 645), fill=RED)
    draw.text((1935, 1160), "Only the decoder differs in the matched ANN control.", font=font(27), fill=RED)

    draw.rounded_rectangle((55, 1195, 1510, 1285), radius=12, fill="#F5F7FA", outline="#98A2B3", width=2)
    draw.text(
        (82, 1220),
        "Training: AdamW, cosine schedule, 80 fixed epochs; CE + 0.01 firing-rate regularisation",
        font=font(27),
        fill=INK,
    )
    image.save(output, dpi=(300, 300))


def learning_curve(csv_path: Path, output: Path) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    examples: dict[tuple[str, str], float] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["budget"], row["variant"])
            grouped[key].append(float(row["accuracy"]))
            examples[(row["dataset"], row["budget"])] = float(row["examples_per_class"])

    width, height = 2100, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Decoder accuracy across labelled examples per class", font=font(44, True), fill=INK)
    panels = {
        "bci2a": (80, 150, 710, 835, "BCI Competition IV-2a"),
        "openbmi": (735, 150, 1365, 835, "OpenBMI"),
        "bnci2014_004": (1390, 150, 2020, 835, "BCI Competition IV-2b"),
    }
    for dataset, (x0, y0, x1, y1, title) in panels.items():
        draw.rectangle((x0, y0, x1, y1), outline="#D0D5DD", width=2)
        draw.text((x0 + 18, y0 + 15), title, font=font(29, True), fill=INK)
        plot = (x0 + 80, y0 + 90, x1 - 35, y1 - 90)
        px0, py0, px1, py1 = plot
        draw.line((px0, py1, px1, py1), fill=INK, width=3)
        draw.line((px0, py0, px0, py1), fill=INK, width=3)
        for tick in range(40, 91, 10):
            y = py1 - (tick - 40) / 50 * (py1 - py0)
            draw.line((px0 - 8, y, px1, y), fill="#E4E7EC", width=2)
            draw.text((px0 - 65, y - 13), str(tick), font=font(22), fill=GREY)
        budgets = sorted(
            {key[1] for key in grouped if key[0] == dataset},
            key=lambda b: examples[(dataset, b)],
        )
        xs = [px0 + i * (px1 - px0) / max(len(budgets) - 1, 1) for i in range(len(budgets))]
        for b, x in zip(budgets, xs):
            label = "all" if b == "all" else b[1:]
            draw.text((x - 18, py1 + 18), label, font=font(22), fill=GREY)
        for variant, colour, label in [("ann_sew_ce", BLUE, "ANN-SEW"), ("sew_clif_ce", RED, "SEW-CLIF")]:
            points = []
            for b, x in zip(budgets, xs):
                acc = 100 * mean(grouped[(dataset, b, variant)])
                y = py1 - (acc - 40) / 50 * (py1 - py0)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colour, width=6)
            for x, y in points:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colour)
            draw.text((px0 + 10, py0 + (0 if variant == "ann_sew_ce" else 34)), label, font=font(22, True), fill=colour)
        draw.text((px0 + 100, py1 + 55), "examples per class", font=font(22), fill=INK)
    draw.text((25, 410), "Accuracy (%)", font=font(24, True), fill=INK)
    draw.text(
        (80, 900),
        "Frozen supervised representations; this isolates decoder label efficiency and is not an end-to-end sample-efficiency test.",
        font=font(25),
        fill=GREY,
    )
    image.save(output, dpi=(300, 300))


def effects(output: Path) -> None:
    rows = [
        ("BCI2a E28\nwithin-session OOF", 10.352, 0.0, 0.0, "25/27 paired runs positive"),
        ("OpenBMI E29\nS1 -> S2", 5.530, 3.896, 7.204, "42/54 subjects positive"),
        ("BCI IV-2b E30\nproject-blind", -0.058, -1.446, 1.181, "5/9 subjects positive"),
    ]
    image = Image.new("RGB", (1900, 820), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Matched SEW-CLIF minus ANN-SEW accuracy", font=font(44, True), fill=INK)
    x0, x1 = 590, 1780
    y_positions = [220, 430, 640]
    vmin, vmax = -3.0, 12.5
    sx = lambda v: x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
    draw.line((sx(0), 145, sx(0), 710), fill="#98A2B3", width=4)
    for tick in range(-2, 13, 2):
        x = sx(tick)
        draw.line((x, 710, x, 724), fill=INK, width=2)
        draw.text((x - 18, 735), str(tick), font=font(22), fill=GREY)
    draw.text((1070, 770), "accuracy difference (percentage points)", font=font(24), fill=INK)
    for (label, value, low, high, note), y in zip(rows, y_positions):
        draw.multiline_text((70, y - 45), label, font=font(27, True), fill=INK, spacing=5)
        colour = RED if value > 0 else BLUE
        if high > low:
            draw.line((sx(low), y, sx(high), y), fill=colour, width=7)
            draw.line((sx(low), y - 13, sx(low), y + 13), fill=colour, width=4)
            draw.line((sx(high), y - 13, sx(high), y + 13), fill=colour, width=4)
        draw.ellipse((sx(value) - 13, y - 13, sx(value) + 13, y + 13), fill=colour)
        draw.text((sx(value) + 20, y - 20), f"{value:+.2f} pp", font=font(25, True), fill=colour)
        draw.text((70, y + 48), note, font=font(23), fill=GREY)
    image.save(output, dpi=(300, 300))


def baseline_comparison(evidence: Path, output: Path) -> None:
    sources = [
        (
            evidence / "paper_figures" / "bci2a" / "baseline_accuracy.png",
            "(a) BCI Competition IV-2a: Session T -> E",
        ),
        (
            evidence / "paper_figures" / "openbmi" / "baseline_accuracy.png",
            "(b) OpenBMI: Session S1 -> S2",
        ),
    ]
    panels: list[Image.Image] = []
    panel_width, panel_height = 1320, 900
    for path, label in sources:
        source = Image.open(path).convert("RGB")
        source.thumbnail((panel_width - 70, panel_height - 120), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((35, 25), label, font=font(32, True), fill=INK)
        framed = ImageOps.expand(source, border=2, fill="#D0D5DD")
        panel.paste(framed, ((panel_width - framed.width) // 2, 82))
        panels.append(panel)

    canvas = Image.new("RGB", (panel_width * 2, panel_height + 105), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 18), "Strong baselines and fixed heterogeneous fusion", font=font(44, True), fill=INK)
    canvas.paste(panels[0], (0, 105))
    canvas.paste(panels[1], (panel_width, 105))
    draw.line((panel_width, 130, panel_width, panel_height + 80), fill="#E4E7EC", width=3)
    canvas.save(output, dpi=(300, 300))


def perturbation_comparison(evidence: Path, output: Path) -> None:
    sources = [
        (
            evidence / "paper_figures" / "bci2a" / "frequency_region_occlusion.png",
            "(a) BCI Competition IV-2a",
        ),
        (
            evidence / "paper_figures" / "openbmi" / "frequency_region_occlusion.png",
            "(b) OpenBMI",
        ),
    ]
    panels: list[Image.Image] = []
    panel_width, panel_height = 1320, 940
    for path, label in sources:
        source = Image.open(path).convert("RGB")
        source.thumbnail((panel_width - 70, panel_height - 120), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((35, 25), label, font=font(32, True), fill=INK)
        framed = ImageOps.expand(source, border=2, fill="#D0D5DD")
        panel.paste(framed, ((panel_width - framed.width) // 2, 82))
        panels.append(panel)

    canvas = Image.new("RGB", (panel_width * 2, panel_height + 105), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 18), "Frequency-band and regional perturbation dependence", font=font(44, True), fill=INK)
    canvas.paste(panels[0], (0, 105))
    canvas.paste(panels[1], (panel_width, 105))
    draw.line((panel_width, 130, panel_width, panel_height + 80), fill="#E4E7EC", width=3)
    canvas.save(output, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = Path(args.evidence_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    architecture(output / "figure_1_architecture.png")
    baseline_comparison(evidence, output / "figure_2_baseline_accuracy.png")
    learning_curve(evidence / "tables" / "E31_variant_metrics.csv", output / "figure_4_learning_curve.png")
    effects(output / "figure_3_matched_decoder_effects.png")
    perturbation_comparison(evidence, output / "figure_5_perturbation_dependence.png")
    print(output)


if __name__ == "__main__":
    main()
