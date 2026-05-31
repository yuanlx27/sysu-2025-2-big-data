import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vqa import DEFAULT_MODEL, ViltVQA, VQAResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a list of VQA test cases.")
    parser.add_argument("--cases", default="data/cases.json", help="JSON file containing image/question test cases.")
    parser.add_argument("--output", default="report/result.csv", help="CSV output path.")
    parser.add_argument("--figures", default="report/figures", help="Directory for PNG test case cards.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local model path.")
    parser.add_argument("--device", default=None, help="Device to run on, for example cpu, cuda, or mps.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached Hugging Face files.")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("cases.json must contain a list of cases.")
    for index, case in enumerate(cases, start=1):
        if "image" not in case or "question" not in case:
            raise ValueError(f"Case {index} must contain image and question fields.")
    return cases


def write_results(path: Path, results: list[VQAResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "question", "answer", "confidence", "model"])
        for result in results:
            writer.writerow(
                [
                    result.image,
                    result.question,
                    result.answer,
                    f"{result.confidence:.4f}",
                    result.model,
                ]
            )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> None:
    x, y = position
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            line = candidate
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)


def write_case_card(path: Path, result: VQAResult, index: int) -> None:
    card = Image.new("RGB", (1000, 620), "#f7f8fb")
    draw = ImageDraw.Draw(card)
    title_font = load_font(32, bold=True)
    heading_font = load_font(24, bold=True)
    body_font = load_font(24)
    answer_font = load_font(42, bold=True)
    meta_font = load_font(20)

    draw.rounded_rectangle((32, 32, 968, 588), radius=8, fill="#ffffff", outline="#d7dce5", width=2)
    draw.text((64, 74), f"Test Case {index}", font=title_font, fill="#18212f")

    source = Image.open(result.image).convert("RGB")
    source.thumbnail((460, 320))
    image_box = Image.new("RGB", (460, 320), "#eef2f7")
    image_box.paste(source, ((460 - source.width) // 2, (320 - source.height) // 2))
    card.paste(image_box, (64, 120))

    draw.text((560, 145), "Question", font=heading_font, fill="#18212f")
    draw_wrapped_text(draw, (560, 185), result.question, body_font, "#263238", 360)
    draw.text((560, 300), "Answer", font=heading_font, fill="#18212f")
    draw.text((560, 345), result.answer, font=answer_font, fill="#0f766e")
    draw.text((560, 405), f"Confidence: {result.confidence:.4f}", font=body_font, fill="#455a64")
    draw.text((64, 500), f"Model: {result.model}", font=meta_font, fill="#455a64")

    card.save(path)


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases))
    predictor = ViltVQA(args.model, args.device, args.local_files_only)

    results: list[VQAResult] = []
    for index, case in enumerate(cases, start=1):
        result = predictor.answer(case["image"], case["question"])
        results.append(result)
        print(f"[{index}] {result.question} -> {result.answer} ({result.confidence:.4f})")

    write_results(Path(args.output), results)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results, start=1):
        write_case_card(figures_dir / f"case_{index}.png", result, index)


if __name__ == "__main__":
    main()
