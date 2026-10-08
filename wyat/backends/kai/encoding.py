"""The maintained tokenizer policy and exact segmented Kai request encoding."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Protocol, cast

from wyat.core import ModelInputError, Request
from wyat.wire import read_object

from .artifact import KaiArtifact, positive_integer

SUFFIX = (
    "\n\nSelect the single option best supported by the context and instructions."
    "\nDecision:"
)


class EncodedRequest(NamedTuple):
    ids: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    query_position: int


class TokenizerProtocol(Protocol):
    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def load_tokenizer(root: Path) -> TokenizerProtocol:
    from transformers import AutoTokenizer

    config = read_object(root / "tokenizer_config.json")
    if config.get("fast_tokenizer_files"):
        raise ValueError(
            "Alternate tokenizer files require an explicit export contract"
        )
    # HF owns Qwen normalization, configured special tokens and legacy overrides.
    # Direct tokenizer.json loading silently skips those parts of the recipe.
    try:
        return cast(
            TokenizerProtocol,
            AutoTokenizer.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
    except Exception as exc:
        raise ValueError(f"Cannot load local tokenizer: {exc}") from exc


class KaiEncoder:
    """Exact segmented tokens; cached values never contain contextual vectors."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        *,
        max_length: int,
        max_options: int,
        token_cache_size: int = 0,
    ) -> None:
        self.max_length = positive_integer(max_length, "max_length")
        if type(token_cache_size) is not int or token_cache_size < 0:
            raise ValueError("token_cache_size must be a nonnegative integer")
        self.tokenizer = tokenizer
        self.max_options = max_options
        self._option_tokens = lru_cache(maxsize=token_cache_size)(self._tokens)
        self._suffix = self._tokens(SUFFIX)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        self.pad_id = pad_id

    @classmethod
    def from_artifact(
        cls, artifact: KaiArtifact, *, max_length: int, token_cache_size: int = 0
    ) -> KaiEncoder:
        return cls(
            load_tokenizer(artifact.root),
            max_length=max_length,
            max_options=artifact.max_options,
            token_cache_size=token_cache_size,
        )

    def _tokens(self, text: str) -> tuple[int, ...]:
        if len(text.encode("utf-8")) > self.max_length * 32:
            raise ModelInputError(
                "Input segment exceeds tokenization byte limit", reason="input-segment"
            )
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids or len(ids) > self.max_length:
            raise ModelInputError(
                "Input segment is empty or exceeds max_length; no truncation",
                reason="input-segment",
            )
        return ids

    def encode_request(self, request: Request) -> EncodedRequest:
        if request.task != "choice":
            raise ModelInputError(
                "Kai currently supports choice requests only", reason="unsupported-task"
            )
        if not 2 <= len(request.candidates) <= self.max_options:
            raise ModelInputError(
                f"Kai requires 2..{self.max_options} candidates",
                reason="candidate-count",
            )
        prefix = (
            f"Context:\n{request.context}\n\n"
            f"Task type: choice\nQuestion:\n{request.query}\nOptions:"
        )
        ids = list(self._tokens(prefix))
        positions: list[int] = []
        for candidate in request.candidates:
            payload = json.dumps(
                {"key": candidate.id, "description": candidate.text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            ids.extend(self._option_tokens("\n<option>\n" + payload + "\n</option>"))
            positions.append(len(ids) - 1)
        ids.extend(self._suffix)
        if len(ids) > self.max_length:
            raise ModelInputError(
                f"{len(ids)} tokens exceeds max_length={self.max_length}; "
                "no truncation",
                reason="input-length",
            )
        return EncodedRequest(tuple(ids), tuple(positions), len(ids) - 1)
