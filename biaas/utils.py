import re
from typing import Any

from biaas.llm.models import get_llm_provider


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def make_llm_call(llm_provider: str, prompt_text: str, is_json_output: bool = False) -> str | Any:
    try:
        llm_model = get_llm_provider(llm_provider)

        return llm_model.get_response(prompt_text, json_output=is_json_output)
    except Exception as e:
        return e
