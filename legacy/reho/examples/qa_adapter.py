"""Wire an existing provider to CHISEL; no requests run at import time."""
from chisel.adapters.qa import solve


class OriginalInput:
    """Replace with the original harness adapter when integrating a benchmark."""

    @staticmethod
    def truncate_context(context):
        return context

    @staticmethod
    def build_user(item):
        return f"Question: {item['question']}\nSource: {item['context']}"


def answer_with_receipts(*, item, original_prompt, call):
    """The caller supplies a provider returning responses and saved usage receipts."""
    return solve(
        item=item,
        arm="interleave",
        branch="original",
        prompts={"original": original_prompt},
        config={"interleave": {
            "backend": "qwen_typed_decision_proxy",
            "maximum_drafts": 2,
            "draft_max_tokens": 192,
            "judge_max_tokens": 96,
        }},
        legacy=OriginalInput(),
        call=call,
    )
