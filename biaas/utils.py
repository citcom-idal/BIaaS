import re
from typing import Any

from biaas.llm.models import get_llm_model


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def make_llm_call(prompt_text: str, is_json_output: bool = False) -> str | Any:
    try:
        llm_model = get_llm_model()

        return llm_model.get_response(prompt_text, json_output=is_json_output)
    except Exception as e:
        return e
