"""Regression checks for real chat-output shapes and Gemma inference-only suffixes."""

import pytest

from rlm.v100.training import chat_token_ids, encode_record


class GemmaTemplate:
    def __init__(self, shape, header="<|turn>model\n", marker="<|channel>thought\n<channel|>"):
        self.shape, self.header, self.marker = shape, header, marker

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        text = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            text += f"<|turn>{role}\n{message['content']}<turn|>\n"
        if add_generation_prompt:
            text += self.header + self.marker
        ids = list(text.encode())
        if self.shape == "mapping":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.shape == "batched-mapping":
            return {"input_ids": [ids]}
        return ids


MESSAGES = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


@pytest.mark.parametrize("shape", ["list", "mapping", "batched-mapping"])
def test_gemma_suffix_removal_masks_every_prompt_token_and_preserves_answer(shape):
    tokenizer = GemmaTemplate(shape)
    encoded = encode_record({"messages": MESSAGES}, tokenizer, 512)
    answer = list(b"4<turn|>\n")
    assert encoded["input_ids"][-len(answer) :] == answer
    assert encoded["labels"] == [-100] * (len(encoded["input_ids"]) - len(answer)) + answer
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])
    assert b"<|channel>thought" not in bytes(encoded["input_ids"])
    with pytest.raises(ValueError, match="shorten"):
        encode_record({"messages": MESSAGES}, tokenizer, len(encoded["input_ids"]) - 1)


@pytest.mark.parametrize(
    "header,marker",
    [
        ("<|turn>user\n", "<|channel>thought\n<channel|>"),
        ("<|turn>model\n", "<|channel>thought\nprivate reasoning<channel|>"),
    ],
)
def test_prefix_mismatch_is_never_repaired_by_guessing(header, marker):
    tokenizer = GemmaTemplate("mapping", header, marker)
    with pytest.raises(ValueError, match="prefix mismatch"):
        encode_record({"messages": MESSAGES}, tokenizer, 512)


@pytest.mark.parametrize("value", [[], [[1], [2]], [True], ["1"], object()])
def test_token_id_shape_is_validated(value):
    with pytest.raises(ValueError, match="token-ID"):
        chat_token_ids(value)
