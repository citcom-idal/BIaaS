import logging

from huggingface_hub import try_to_load_from_cache
from sentence_transformers import SentenceTransformer

from app.core.constants import EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    filepath = try_to_load_from_cache(repo_id=EMBEDDING_MODEL, filename="modules.json")

    if isinstance(filepath, str):
        logger.info("Model '%s' is already cached at: %s", EMBEDDING_MODEL, filepath)
        return

    logger.info("Model '%s' is not cached. Downloading...", EMBEDDING_MODEL)

    try:
        SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        logger.info("Model '%s' has been loaded and cached.", EMBEDDING_MODEL)
    except Exception:
        logger.exception("Failed to load model '%s': %s", EMBEDDING_MODEL)
        raise


if __name__ == "__main__":
    main()
