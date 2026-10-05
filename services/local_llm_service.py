import json
import logging
import os
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SECONDS = 1
OLLAMA_GENERATION_TIMEOUT_SECONDS = 90


def get_ollama_model():
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def _get_installed_models():
    request = Request(f"{OLLAMA_BASE_URL}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        LOGGER.debug("Local Ollama status check failed (%s).", error.__class__.__name__)
        return None

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None
    return {
        name
        for model in models
        if isinstance(model, dict)
        for name in (model.get("name"), model.get("model"))
        if isinstance(name, str)
    }


def get_local_llm_status():
    models = _get_installed_models()
    model = get_ollama_model()
    running = models is not None
    model_available = running and model in models
    return {
        "ollama_installed": bool(shutil.which("ollama")) or running,
        "ollama_running": running,
        "local_llm_available": bool(model_available),
        "model": model,
    }


def generate_local_response(prompt):
    status = get_local_llm_status()
    if not status["local_llm_available"]:
        return None

    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps({
            "model": status["model"],
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_GENERATION_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        LOGGER.warning("Local Ollama generation failed (%s).", error.__class__.__name__)
        return None

    answer = payload.get("response") if isinstance(payload, dict) else None
    return answer.strip() if isinstance(answer, str) and answer.strip() else None
