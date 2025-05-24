import argparse
import os
import sys
import io
import pathlib
import requests
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TextStreamer,
)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(("http://", "https://")):
        resp = requests.get(image_source, stream=True, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        path = pathlib.Path(image_source).expanduser().resolve()
        if not path.is_file():
            sys.exit(f"[Wrong] Can't find fig：{path}")
        return Image.open(str(path)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Terminal visual question answering with LLaVA‑1.5‑7B"
    )
    parser.add_argument(
        "-i", "--image", metavar="IMAGE",
        help="Local image path or URL"
    )
    parser.add_argument(
        "-q", "--question", metavar="TEXT",
        help="Question"
    )
    args = parser.parse_args()
    
    image_src = args.image or input("Fig Path/URL：").strip()
    question  = args.question or input("Question：").strip()

    
    print("⏳ Load Model…")
    tokenizer        = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    image_processor  = AutoImageProcessor.from_pretrained(MODEL_ID)
    processor        = LlavaProcessor(image_processor=image_processor,
                                      tokenizer=tokenizer)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )

    
    image = load_image(image_src)

    
    prompt = f"USER: <image>\n{question} ASSISTANT:"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True
    ).to(model.device, DTYPE)

    print(f"\nQuestion: {question}\nAnswer:")
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=True)

    model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        streamer=streamer
    )


if __name__ == "__main__":
    main()