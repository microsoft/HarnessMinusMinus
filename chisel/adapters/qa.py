"""QA request orchestration with stage-by-stage provider usage accounting."""
from chisel.harness import jev_interleave as logic

def solve(item: dict, arm: str, branch: str, prompts: dict, config: dict, legacy, call) -> dict:
    context = legacy.truncate_context(item["context"])
    question = item["question"]
    settings = config["interleave"]
    records, stages = [], []
    draft_count, judge_count = 0, 0

    def request(messages, stage, maximum, schema=None):
        record = call(messages, phase="jev_" + stage, key=f"{item['id']}:{arm}:{len(records)}",
                      maximum=maximum, response_schema=schema)
        records.append(record)
        stages.append(stage)
        return record["raw_response"]["choices"][0]["message"]["content"]

    def fallback():
        return request([{"role": "system", "content": prompts[branch]},
                        {"role": "user", "content": legacy.build_user(item)}], "fallback", 512)

    def generate(previous, decisions):
        nonlocal draft_count
        draft_count += 1
        return request(logic.generator_messages(question, context, previous, decisions), "draft",
                       settings["draft_max_tokens"], logic.DRAFT_SCHEMA)

    def judge(draft):
        nonlocal judge_count
        judge_count += 1
        return request(logic.verifier_messages(question, context, draft), "proxy_judge",
                       settings["judge_max_tokens"], logic.DECISION_SCHEMA)

    if arm in {"original", "no_skill", "skillopt", "autosaddler"}:
        system = prompts[branch] if arm == "original" else prompts[arm]
        result = {"answer": request([{"role": "system", "content": system},
                                     {"role": "user", "content": legacy.build_user(item)}], "baseline", 512),
                  "trace": [], "fallback": False, "drafts": 0}
    elif arm == "cot_only":
        raw = generate(None, None)
        try:
            draft = logic.parse_draft(raw, context)
            result = {"answer": f"<answer>{draft.answer}</answer>", "trace": [], "fallback": False, "drafts": 1}
        except (TypeError, ValueError) as error:
            result = {"answer": fallback(), "trace": [{"status": "INVALID_DRAFT", "error": str(error)}],
                      "fallback": True, "drafts": 1}
    elif arm == "interleave":
        result = logic.interleave(context=context, generate=generate, judge=judge, fallback=fallback,
                                  maximum_drafts=settings["maximum_drafts"])
    else:
        raise ValueError("Unknown fixed experiment condition")
    usage = {field: sum(record["usage"][field] for record in records)
             for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if not records or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("Every online stage must be accounted")
    return {**result, "id": item["id"], "arm": arm, "route": branch,
            "usage": usage, "calls": [record["call_file"] for record in records], "stages": stages,
            "draft_calls": draft_count, "judge_calls": judge_count,
            "verifier_identity": settings["backend"], "true_jev_used": False,
            "calibration_claimed": False}
