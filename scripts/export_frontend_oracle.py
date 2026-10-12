# /// script
# requires-python = ">=3.10"
# dependencies = ["irodori-tts", "dacvae", "silentcipher", "torch==2.10.0", "torchaudio==2.10.0", "transformers==5.12.1"]
# [tool.uv.sources]
# irodori-tts = { git = "https://github.com/Aratako/Irodori-TTS", rev = "89f9d8fbd4d51ea019867ee1197725ede1df13c5" }
# dacvae = { git = "https://github.com/facebookresearch/dacvae", rev = "414c20785fc3a28373073ea8ef7a1316eeeaca6e" }
# silentcipher = { git = "https://github.com/SesameAILabs/silentcipher.git", rev = "d46d7d0893a583d8968ab3a6626e2289faec9152" }
# ///
"""Export normalization/token goldens from the pinned official tokenizer.

The cases combine speech.cpp's MIT-licensed text_cases.py edge cases with
twenty short Japanese sentences. No GPU or model weights are needed.
"""

import argparse
import ast
import json
from pathlib import Path

from irodori_tts.duration import ALLOWED_ANNOTATION_EMOJIS
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        f"{word}を確認します。"
        for word in [
            "天気",
            "明日の予定",
            "音声合成",
            "東京の気温",
            "電車の時刻",
            "日本語の発音",
            "声の高さ",
            "文章の長さ",
            "漢字の読み方",
            "数字",
            "記号",
            "句読点",
            "疑問文",
            "感情表現",
            "録音",
            "話す速度",
            "音量",
            "朝の挨拶",
            "名前",
            "次のニュース",
        ]
    ]
    tree = ast.parse(
        (args.speech_root / "reference/irodori-tts/text_cases.py").read_text()
    )
    for node in tree.body:
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "texts"
            and isinstance(node.value, ast.List)
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.value.elts
            )
        ):
            cases.extend(ast.literal_eval(node.value))
    emojis = sorted(ALLOWED_ANNOTATION_EMOJIS)
    cases.extend(emojis)
    cases.extend(f"えっと{e}、そうなんだ{e}。" for e in emojis)
    cases.append("".join(emojis))
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        str(args.tokenizer), add_bos=True, local_files_only=True
    )
    records = []
    for text in cases:
        normalized = normalize_text(text).strip()
        records.append(
            {
                "input": text,
                "normalized": normalized,
                "ids": tokenizer.encode(normalized).tolist() if normalized else [],
            }
        )
    with args.output.open("x") as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(len(records), "cases")


if __name__ == "__main__":
    main()
