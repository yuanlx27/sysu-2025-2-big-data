import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor


DEFAULT_MODEL = "dandelin/vilt-b32-finetuned-vqa"


@dataclass
class VQAResult:
    image: str
    question: str
    answer: str
    confidence: float
    model: str


class ViltVQA:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = ViltProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = ViltForQuestionAnswering.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_safetensors=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def answer(self, image_path: str | Path, question: str) -> VQAResult:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            score, predicted_id = probabilities.max(dim=-1)

        answer = self.model.config.id2label[predicted_id.item()]
        return VQAResult(
            image=str(image_path),
            question=question,
            answer=answer,
            confidence=float(score.item()),
            model=self.model_name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual question answering with a pretrained ViLT model.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local model path.")
    parser.add_argument("--device", default=None, help="Device to run on, for example cpu, cuda, or mps.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached Hugging Face files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = ViltVQA(args.model, args.device, args.local_files_only)
    result = predictor.answer(args.image, args.question)
    print(f"Image: {result.image}")
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Model: {result.model}")


if __name__ == "__main__":
    main()
