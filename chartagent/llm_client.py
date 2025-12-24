import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Protocol, cast

from PIL import Image

def _maybe_load_dotenv() -> Optional[str]:
    """Best-effort `.env` loader (useful for non-interactive runs like sbatch).

    Returns the resolved `.env` path if loaded.
    """

    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return None

    dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return str(dotenv_path)
    return None


_DOTENV_PATH = _maybe_load_dotenv()


class LLMClient(Protocol):
    def generate(self, *, prompt: str, images: Optional[List[Image.Image]] = None) -> str: ...


def _data_url_from_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = __import__("base64").b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64,{}".format(b64)


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), resample=Image.BICUBIC)


class OpenAIResponsesClient(object):
    """Minimal wrapper around OpenAI Responses API (vision)."""

    def __init__(self, *, model: str, max_side: int = 1024) -> None:
        self.model = str(model)
        self.max_side = int(max_side)

        if not os.environ.get("OPENAI_API_KEY"):
            hint = ""
            if _DOTENV_PATH:
                hint = f' (loaded "{_DOTENV_PATH}" but key is still missing)'
            raise RuntimeError(
                "OPENAI_API_KEY is not set{}. Set it in the environment, or create a `.env` with "
                "OPENAI_API_KEY=... and install python-dotenv (pip install python-dotenv).".format(hint)
            )

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI Python SDK (v1+) is required. Install/upgrade with: pip install --upgrade openai"
            ) from e

        self._client = OpenAI()

    def generate(self, *, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": str(prompt)}]
        if images:
            for im in images:
                im2 = _resize_max_side(im, self.max_side)
                content.append({"type": "input_image", "image_url": _data_url_from_image(im2)})

        resp = self._client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        out_text = getattr(resp, "output_text", None)
        if out_text is None:
            try:
                out_text = str(resp.output[0].content[0].text)  # type: ignore
            except Exception:
                out_text = ""
        return str(out_text or "").strip()


def _looks_like_openai_model(model: str) -> bool:
    m = str(model or "").strip()
    if not m:
        return False
    if m.startswith("openai:"):
        return True
    ml = m.lower()
    # Common OpenAI model prefixes (Responses API).
    return bool(
        ml.startswith("gpt-")
        or ml.startswith("o1")
        or ml.startswith("o3")
        or ml.startswith("o4")
        or ml.startswith("chatgpt-")
    )


class Qwen3VLClient(object):
    """Local HF Qwen3-VL client (Transformers)."""

    def __init__(self, *, model: str, max_side: int = 1024) -> None:
        self.model_id = str(model)
        self.max_side = int(max_side)

        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Qwen3-VL requires torch + transformers. Install with: pip install torch transformers"
            ) from e

        attn_impl = os.environ.get("QWEN3_VL_ATTN_IMPL")
        kwargs: Dict[str, Any] = {"dtype": "auto", "device_map": "auto"}
        if isinstance(attn_impl, str) and attn_impl.strip():
            kwargs["attn_implementation"] = str(attn_impl).strip()

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._torch = torch

        try:
            self._max_new_tokens = int(os.environ.get("CHARTAGENT_MAX_NEW_TOKENS") or "1024")
        except Exception:
            self._max_new_tokens = 1024

    def generate(self, *, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        content: List[Dict[str, Any]] = []
        if images:
            for im in images:
                im2 = _resize_max_side(im, self.max_side)
                content.append({"type": "image", "image": im2})
        content.append({"type": "text", "text": str(prompt)})

        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model.device)
        elif isinstance(inputs, dict):
            inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        generated_ids = self._model.generate(**inputs, max_new_tokens=int(self._max_new_tokens), do_sample=False)
        # Trim the prompt tokens.
        in_ids = getattr(inputs, "input_ids", None)
        if in_ids is None and isinstance(inputs, dict):
            in_ids = inputs.get("input_ids")
        if in_ids is None:
            raise RuntimeError("Qwen3VLClient: processor output is missing input_ids")
        generated_ids_trimmed = [out_ids[len(in0) :] for in0, out_ids in zip(in_ids, generated_ids)]
        out_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return str(out_text[0] if out_text else "").strip()


class DeepSeekVLV2Client(object):
    """Local DeepSeek-VL2 client (deepseek_vl + transformers)."""

    def __init__(self, *, model: str, max_side: int = 1024) -> None:
        self.model_id = str(model)
        self.max_side = int(max_side)

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError("DeepSeek-VL2 requires torch + transformers.") from e

        try:
            from deepseek_vl2.models import DeepseekVLV2Processor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "DeepSeek-VL2 requires the `deepseek_vl2` package. Install it from the vendored source:\n\n"
                "  pip install -e DeepSeek-VL2\n\n"
                "or install from PyPI if available, and see `deepseek_vl2_usage.md`."
            ) from e

        self._processor = DeepseekVLV2Processor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer

        # Optional: clamp image candidate resolutions to avoid very large tiling/tokenization.
        # This is often the root cause of GPU OOM (image tokens -> huge KV cache).
        max_candidate_side = 0
        try:
            max_candidate_side = int(os.environ.get("CHARTAGENT_DEEPSEEK_MAX_CANDIDATE_SIDE") or "0")
        except Exception:
            max_candidate_side = 0
        if max_candidate_side > 0:
            try:
                cand = getattr(self._processor, "candidate_resolutions", None)
                if isinstance(cand, (list, tuple)):
                    filtered: List[tuple[int, int]] = []
                    for item in cand:
                        if not (isinstance(item, (list, tuple)) and len(item) == 2):
                            continue
                        try:
                            w = int(item[0])
                            h = int(item[1])
                        except Exception:
                            continue
                        if max(w, h) <= max_candidate_side:
                            filtered.append((w, h))
                    if filtered:
                        self._processor.candidate_resolutions = tuple(filtered)  # type: ignore[assignment]
                        try:
                            self._processor.image_size = int(filtered[0][0])  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                pass

        # Prefer GPU when available.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Optional: shard the model across multiple GPUs via HF `device_map`.
        # NOTE: requesting more GPUs in Slurm does nothing unless the model is actually sharded.
        device_map_env = str(os.environ.get("CHARTAGENT_DEEPSEEK_DEVICE_MAP") or "").strip()
        device_map: Optional[str] = device_map_env or None
        if device == "cpu":
            device_map = None

        # Reduce peak CPU memory during load; on Slurm this can avoid cgroup OOM kills
        # when `--mem` is modest.
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs.update({"torch_dtype": dtype, "low_cpu_mem_usage": True})
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except ImportError:
            # Most common when `device_map` is used without `accelerate` installed.
            if "device_map" in model_kwargs:
                model_kwargs.pop("device_map", None)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            else:
                raise
        except TypeError:
            # Back-compat for older transformers that don't accept some kwargs.
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
            if device == "cuda":
                self._model = self._model.to(dtype=dtype)
        if device_map is None:
            self._model = self._model.to(device)
        self._model = self._model.eval()
        self._torch = torch

        try:
            self._max_new_tokens = int(os.environ.get("CHARTAGENT_MAX_NEW_TOKENS") or "1024")
        except Exception:
            self._max_new_tokens = 1024

    def generate(self, *, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        images = images or []
        pil_images: List[Image.Image] = [_resize_max_side(im, self.max_side) for im in images]

        # DeepSeek-VL2 uses "<image>" placeholders in the user content.
        img_prefix = "".join(["<image>\n" for _ in pil_images])
        user_content = img_prefix + str(prompt)

        # The processor expects a conversation with an assistant turn.
        conversation: List[Dict[str, Any]] = [
            {"role": "<|User|>", "content": user_content, "images": [""] * len(pil_images)},
            {"role": "<|Assistant|>", "content": ""},
        ]

        use_cache = True
        use_cache_env = str(os.environ.get("CHARTAGENT_DEEPSEEK_USE_CACHE") or "").strip().lower()
        if use_cache_env in {"0", "false", "no", "off"}:
            use_cache = False

        def _is_oom(err: BaseException) -> bool:
            return "out of memory" in str(err).lower()

        with self._torch.inference_mode():
            prepare_inputs = self._processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt="",
            ).to(self._model.device)

            inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

            def _generate_once(*, use_cache_flag: bool) -> object:
                # Prefer generating from the outer VLM model (matches DeepSeek-VL2 serve code).
                # Some versions crash if `input_ids` is omitted even when `inputs_embeds` is provided, so pass both.
                gen_kwargs: Dict[str, Any] = {
                    "inputs_embeds": inputs_embeds,
                    "input_ids": prepare_inputs.input_ids,
                    "attention_mask": prepare_inputs.attention_mask,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "bos_token_id": self._tokenizer.bos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                    "max_new_tokens": int(self._max_new_tokens),
                    "do_sample": False,
                    "use_cache": bool(use_cache_flag),
                }

                last_exc: Optional[BaseException] = None
                outputs = None

                # 1) Try multimodal generate on the outer model (preferred).
                if hasattr(self._model, "generate"):
                    try:
                        outputs = self._model.generate(  # type: ignore[call-arg]
                            **gen_kwargs,
                            images=prepare_inputs.images,
                            images_seq_mask=prepare_inputs.images_seq_mask,
                            images_spatial_crop=prepare_inputs.images_spatial_crop,
                        )
                    except TypeError as e:
                        last_exc = e
                        outputs = None
                    except Exception as e:
                        last_exc = e
                        if _is_oom(e):
                            raise
                        outputs = None

                # 2) Fall back to outer model generate without multimodal kwargs
                # (still multimodal if `inputs_embeds` already contains image features).
                if outputs is None and hasattr(self._model, "generate"):
                    try:
                        outputs = self._model.generate(**gen_kwargs)  # type: ignore[call-arg]
                    except Exception as e:
                        last_exc = e
                        if _is_oom(e):
                            raise
                        outputs = None

                # 3) Last resort: call the underlying language model if present.
                if outputs is None:
                    candidates: List[object] = []
                    if hasattr(self._model, "language_model"):
                        candidates.append(getattr(self._model, "language_model"))
                    if hasattr(self._model, "language"):
                        candidates.append(getattr(self._model, "language"))
                    lm = next((c for c in candidates if hasattr(c, "generate")), None)
                    if lm is None:
                        raise RuntimeError(
                            "DeepSeek-VL2 model does not expose a `generate` method via `.generate`, `.language_model`, or `.language`. "
                            "Please check your `deepseek_vl2`/transformers versions."
                        ) from last_exc
                    try:
                        outputs = lm.generate(**gen_kwargs)  # type: ignore[call-arg]
                    except Exception as e:
                        last_exc = e
                        if _is_oom(e):
                            raise
                        outputs = None

                if outputs is None:
                    raise RuntimeError("DeepSeek-VL2 generate failed") from last_exc
                return outputs

            try:
                outputs = _generate_once(use_cache_flag=use_cache)
            except Exception as e:
                if use_cache and _is_oom(e) and self._torch.cuda.is_available():
                    # Best-effort retry with KV cache disabled.
                    try:
                        self._torch.cuda.empty_cache()
                    except Exception:
                        pass
                    outputs = _generate_once(use_cache_flag=False)
                else:
                    raise

        decoded = self._tokenizer.decode(outputs[0].detach().to("cpu").tolist(), skip_special_tokens=True)
        # Best-effort: strip the prompt prefix if present.
        try:
            sft = prepare_inputs.get("sft_format")  # type: ignore[attr-defined]
            if isinstance(sft, (list, tuple)) and sft and isinstance(sft[0], str):
                p0 = str(sft[0])
                if decoded.startswith(p0):
                    decoded = decoded[len(p0) :]
        except Exception:
            pass
        return str(decoded).strip()


def create_llm_client(*, model: str, max_side: int = 1024) -> LLMClient:
    """
    Factory for ChartAgent LLM backends.

    - OpenAI (Responses API): pass model like "gpt-5-nano" (requires OPENAI_API_KEY)
    - Qwen3-VL: pass model like "Qwen/Qwen3-VL-4B-Instruct" (local HF)
    - DeepSeek-VL2: pass model like "deepseek-ai/deepseek-vl2-small" (local HF + deepseek_vl)
    """

    m = str(model or "").strip()
    ml = m.lower()

    if _looks_like_openai_model(m):
        # Strip optional prefix.
        if m.startswith("openai:"):
            m = m[len("openai:") :]
        return OpenAIResponsesClient(model=m, max_side=int(max_side))

    if "qwen3-vl" in ml:
        return Qwen3VLClient(model=m, max_side=int(max_side))

    if "deepseek-vl2" in ml:
        return DeepSeekVLV2Client(model=m, max_side=int(max_side))

    raise RuntimeError(
        "Unknown LLM model backend for {!r}. Use an OpenAI model (e.g., gpt-5-nano), "
        "Qwen3-VL (e.g., Qwen/Qwen3-VL-4B-Instruct), or DeepSeek-VL2 (e.g., deepseek-ai/deepseek-vl2-small).".format(
            m
        )
    )
