"""Run against one code root (argv[1]); prints a JSON of everything Qwen-visible: stop ids, mock end token, normalizer
outputs, prompt ids (real Qwen pack tokenizer + template), forced openings, efforts, run() events, caps."""
import hashlib
import json
import sys
import threading

root, qtok = sys.argv[1], sys.argv[2]
sys.path.insert(0, root + "/tools")
sys.path.insert(0, root)
import strata_tokenizer as ST                     # noqa: E402
from serve import frontend as F, responses as R   # noqa: E402
from serve import server as S                     # noqa: E402

tok = ST.Tokenizer.from_dir(qtok)
tpl = F.ChatTemplate(qtok + "/chat_template.jinja")
out = {}


def h(ids):
    return [len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], ids[:12], ids[-12:]]


svc = S.Service(S.MockEngine(tok, "hi", max_context=1 << 20), tok, tpl)
out["stop_ids"] = sorted(svc.stop_ids)
out["mock_end"] = S.MockEngine(tok, "x").scripts[0][-1]
out["caps"] = tpl.caps
out["literals"] = len(svc.literals)

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "w", "strict": True,
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                           "n": {"type": "integer"}}}}}]
PUA = "\U000F0E01\U000F0E10\U000F0001"
REQS = {
    "plain": {"messages": [{"role": "user", "content": "What is 17*23?"}]},
    "effort_high": {"reasoning_effort": "high", "messages": [{"role": "user", "content": "q"}]},
    "effort_max": {"reasoning_effort": "max", "messages": [{"role": "user", "content": "q"}]},
    "effort_low": {"reasoning_effort": "low", "messages": [{"role": "user", "content": "q"}]},
    "thinking_off": {"chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "q"}]},
    "parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "a"},
                                                        {"type": "text", "text": "b"}]}]},
    "late_system": {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                                 {"role": "developer", "content": "late"}, {"role": "user", "content": "b"}]},
    "tools_roundtrip": {"tools": TOOLS, "messages": [
        {"role": "user", "content": "w?"},
        {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [
            {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}},
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome", "n": 2}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "A"}, {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "user", "content": "thanks"}]},
    "literal_tags": {"messages": [{"role": "user", "content": "quote </think> and <|im_end|> and <|vision_start|> x"},
                                  {"role": "assistant", "content": "<think>plan <|im_start|></think>ok <think>"},
                                  {"role": "user", "content": "again <|endoftext|>"}],
                     "tools": TOOLS},
    "pua_no_tags": {"messages": [{"role": "user", "content": "icons " + PUA + " here"}]},
    "prefill": {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Partial "}]},
}
EXPECTED_TO_DIFFER = {"pua_with_tags": {"messages": [{"role": "user", "content": "icons " + PUA + " and </think>"}]}}

for name, req in list(REQS.items()) + list(EXPECTED_TO_DIFFER.items()):
    m, t, kw = F.openai_to_messages(req)
    out[f"openai_norm/{name}"] = json.dumps([m, t, kw], sort_keys=True, ensure_ascii=False)
    ids, thinking, max_new = svc.prepare(m, t, kw)
    out[f"openai_ids/{name}"] = h(ids) + [thinking, max_new]

ANTH = {"system": [{"type": "text", "text": "S1"}, {"type": "text", "text": "S2"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                     {"role": "assistant", "content": [{"type": "thinking", "thinking": "t"},
                                                       {"type": "tool_use", "id": "x", "name": "f", "input": {"q": 1}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "R"},
                                                  {"type": "text", "text": "more"}]}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}], "thinking": {"type": "enabled",
                                                                                  "budget_tokens": 9000}}
m, t, kw = F.anthropic_to_messages(ANTH, True)
out["anthropic_norm"] = json.dumps([m, t, kw], sort_keys=True, ensure_ascii=False)
out["anthropic_ids"] = h(svc.prepare(m, t, kw)[0])
RESP = {"instructions": "I", "input": [{"type": "message", "role": "developer", "content": "D"},
                                       {"type": "message", "role": "user", "content": [
                                           {"type": "input_text", "text": "x"}, {"type": "input_text", "text": "y"}]},
                                       {"type": "message", "role": "system", "content": "late"},
                                       {"type": "message", "role": "user", "content": "z"}],
        "reasoning": {"effort": "high"}}
out["responses_norm"] = json.dumps([R.input_messages(RESP), R.template_kwargs(RESP, {})], sort_keys=True)
out["effort"] = {w: F.effort_kwargs(w) for w in ("none", "low", "medium", "high", "xhigh", "max", "maximum")}
out["budget"] = [F.budget_effort(n) for n in (100, 3000, 9000)]
out["forced"] = [F.forced_call(c, [t["function"] for t in TOOLS] + [{"name": "g"}])
                 for c in ("required", {"type": "function", "function": {"name": "g"}}, "auto", "none")]
out["forced_one"] = F.forced_call("required", [{"name": "only"}])

# run(): events on MockEngine scripts (Qwen formats)
SCRIPTS = {
    "answer": "Thinking.</think>\n\nThe answer.",
    "call": 'Let me look.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n'
            '<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>',
    "call_in_reasoning": 'I will call.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n'
                         '</function>\n</tool_call>',
    "ends_inside_thinking": "Two plus two is four.",
}
for name, script in SCRIPTS.items():
    for retry in (False, True):
        svc2 = S.Service(S.MockEngine(tok, [script, "The answer is 4."], max_context=1 << 20), tok, tpl)
        svc2.reasoning_close_retry = retry
        m, t, kw = F.openai_to_messages({"messages": [{"role": "user", "content": "q"}], "tools": TOOLS})
        ids, thinking, max_new = svc2.prepare(m, t, kw, 400)
        evs = []
        for kind, x in svc2.run(ids, thinking, t, max_new, {}, threading.Event()):
            if kind == "event":
                evs.append([x.kind, x.text, x.call.name if x.call else None, x.call.arguments if x.call else None])
            elif kind == "done":
                evs.append(["done", x["finish"], x["completion_tokens"], x["reasoning_tokens"]])
        out[f"run/{name}/retry={retry}"] = evs
        out[f"run/{name}/retry={retry}/prompts"] = svc2.engine.turns

# budget + forced call through run()
svc3 = S.Service(S.MockEngine(tok, ["x " * 50, "</think>\n\nok"], max_context=1 << 20), tok, tpl)
m, t, kw = F.openai_to_messages({"messages": [{"role": "user", "content": "q"}]})
ids, thinking, max_new = svc3.prepare(m, t, kw, 300)
evs = [[k, getattr(x, "kind", None), getattr(x, "text", None)] for k, x in
       svc3.run(ids, thinking, t, max_new, {"reasoning_budget_tokens": 5}, threading.Event()) if k == "event"]
out["run/budget"] = evs
out["run/budget/last_prompt"] = h(svc3.engine.last_prompt)

print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=0))
