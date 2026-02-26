import os

from sentence_transformers import SentenceTransformer

from app.core.constants import EMBEDDING_MODEL


def main() -> None:
    cache_dir = os.path.expanduser("~/.cache")
    model_path = os.path.join(cache_dir, EMBEDDING_MODEL)

    if not os.path.exists(model_path):
        SentenceTransformer(EMBEDDING_MODEL, device="cpu")


if __name__ == "__main__":
    main()
