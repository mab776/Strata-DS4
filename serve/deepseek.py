"""serve/deepseek.py - mab776 M4: the DeepSeek-V4-Flash-0731 (DS4) family for Strata's server.

The model's own chat template (the GGUF's, exported with the tokenizer) renders the prompt; this module gives it what
llama.cpp gives it, so prompts are byte-equal to stack 73's (llama.cpp b11429) - and reads the reply's DSML tool calls.
The server's hooks read the family from the template object (`getattr(template, name, default)`), so a plain
ChatTemplate - Qwen - takes exactly the code path it always did.

    DeepSeekTemplate        render: bos_token, thinking on by default, effort high/max, tools as type=function, tool
                            results in call order, text parts joined, a trailing assistant continued (prefill)
    DeepSeekOutputParser    reasoning_content / content / DSML tool calls, streamed; malformed DSML is text
    detect, tokenizer_problems, config_problems     selection and the config checks main() runs before the engine

Adapted from (MIT):
  - Aleks-Che/Strata, branch dev, serve/deepseek.py (efbeb163, 444f442c): the template wrapper (bos, thinking default,
    tools wrapped as type=function, images refused), the DSML grammar (START/END/INVOKE/PARAM, calls_of), malformed
    DSML as text; https://github.com/Aleks-Che/Strata
  - Jackwwg83/Strata-DS, branch main, serve/deepseek.py (2a9342a2, d468e6d5): the parser interface for Strata 0.1.4x
    (finish(reason), pending / rescued / refused), reasoning_wrap_up, the config's args passed through to the engine;
    https://github.com/Jackwwg83/Strata-DS
  - ggml-org/llama.cpp common/parsers/deepseek.cpp, common/chat.cpp, tools/server/server-common.cpp: V4 tool-result
    order, the reasoning / tool-call boundary, the prefill continuation, concat_content_parts, tool re-serialization.
"""
from __future__ import annotations

import json
import re

from serve.frontend import (RCALL_MAX, ChatTemplate, Event, OutputParser, TemplateRequestError, ToolCall,
                            _join_texts, images_of)

FAMILY = "deepseek4"
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_START, THINK_END = "<think>", "</think>"
DSML = "｜DSML｜"
START, END = f"<{DSML}tool_calls>", f"</{DSML}tool_calls>"
INVOKE_OPEN, INVOKE_CLOSE = f'<{DSML}invoke name="', f"</{DSML}invoke>"
PARAM_OPEN, PARAM_CLOSE = f'<{DSML}parameter name="', f"</{DSML}parameter>"
SEP = "\n\n"                     # what the template writes between the text and a call block (llama.cpp TC_SEPARATOR)

# The client's words -> the template's.  The template knows only "high" and "max" (a preamble before the system
# prompt); low and medium think with no preamble, as the default does.  llama.cpp hands the word over as it is, so
# "xhigh" gets nothing there: here it is "high" (by design; the c06b capture differs).
DS_EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
             "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "max", "maximum": "max"}
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer." + THINK_END


# ------------------------------------------------------------------------------------------------ selection
def detect(source: str) -> bool:
    """The chat template is DeepSeek V4's: llama.cpp's own test (common/chat.cpp - `dsml_token` and DSML, and a
    `tool_calls` or `function_calls` block).  ValueError for V3.2's template (`function_calls`): its DSML differs."""
    if "dsml_token" not in source or "DSML" not in source or not ("tool_calls" in source or
                                                                  "function_calls" in source):
        return False
    if "function_calls" in source:
        raise ValueError("this chat template is DeepSeek V3.2's (DSML function_calls); this server reads V4's "
                         "(tool_calls) only")
    return True


def tokenizer_problems(tok) -> list[str]:
    """Why `tok` cannot serve DS4 ([] = it can): it must name bos and eos (tokenizer.json special_ids) and encode each
    marker the template writes - and the stop token - as ONE token, or the prompt and the stop rule are wrong."""
    special = getattr(tok, "special_ids", None) or {}
    out = [f"the tokenizer names no {k} (tokenizer.json special_ids)" for k in
           ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id") if k not in special]
    for text in (BOS, EOS, USER, ASSISTANT, THINK_START, THINK_END, DSML):
        try:
            ids = tok.encode(text, parse_special=True)
        except Exception as e:  # noqa: BLE001 - reported, never raised
            ids = [f"error: {e}"]
        if len(ids) != 1:
            out.append(f"{text!r} is not one token in this tokenizer ({ids[:8]})")
    for key, text in (("tokenizer.ggml.bos_token_id", BOS), ("tokenizer.ggml.eos_token_id", EOS)):
        if key in special and tok.encode(text, parse_special=True) != [special[key]]:
            out.append(f"{key} = {special[key]} is not {text!r}")
    return out


def config_problems(cfg: dict, slot_save_path=None) -> tuple[list[str], list[str]]:
    """(errors, notes) for a run config serving DS4.  Errors stop the start (before the minutes of loading): each one
    would make the server use something the DS4 engine does not have.  Notes: Qwen-only settings that are ignored."""
    errors, notes = [], []
    if cfg.get("vision"):
        errors.append('"vision": DeepSeek-V4-Flash reads text only (no image encoder) - remove the section')
    if cfg.get("parallel") not in (None, False, 0, 1):
        errors.append('"parallel": the DeepSeek engine runs one request at a time (no batch slots) - remove it')
    if cfg.get("vram_elastic") is True:
        errors.append('"vram_elastic": the DeepSeek engine has no VRAM command - remove it')
    if cfg.get("expert_profile_save"):
        errors.append('"expert_profile_save": not a DeepSeek engine option - remove it')
    if slot_save_path or cfg.get("slot_save_path"):
        errors.append('"slot_save_path" / --slot-save-path: the DeepSeek engine has no SAVE/RESTORE (the server ends '
                      "an engine that answers SAVE with an error) - remove it")
    if cfg.get("effort_position", "start") != "start":
        notes.append('"effort_position": "end" ignored for DeepSeek (it writes a ChatML turn); the effort stays at '
                     "the top of the prompt")
    if cfg.get("tool_call_recovery") is True:
        notes.append('"tool_call_recovery" ignored for DeepSeek (it reads Qwen\'s call forms; DSML is parsed '
                     "strictly)")
    if cfg.get("reasoning_loop_recovery") in (True, "recover"):
        notes.append('"reasoning_loop_recovery": "recover" swaps ChatML effort sentences and does nothing for '
                     'DeepSeek ("stop" works)')
    return errors, notes


# ------------------------------------------------------------------------------------------------ render helpers
def _from_json(value):
    """The template's `from_json` filter (string tool-call arguments).  Bad JSON is the client's 400."""
    try:
        return json.loads(value)
    except (TypeError, ValueError) as e:
        raise TemplateRequestError(f"tool call arguments are not valid JSON: {e}") from None


def normalize_tools(tools):
    """Tools as llama.cpp hands them to the template (common_chat_tools_parse_oaicompat + _to_json_oaicompat):
    {"type": "function", "function": {"name", "description" ("" if none), "parameters" ({} if none)}}, in that key
    order - a client's extra keys (strict, ...) are left out.  Takes the wrapped or the unwrapped form."""
    if not tools:
        return None
    out = []
    for t in tools:
        f = t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
        if not isinstance(f, dict):
            continue
        params = f.get("parameters")
        out.append({"type": "function", "function": {"name": f.get("name"), "description": f.get("description") or "",
                                                     "parameters": {} if params is None else params}})
    return out or None


def sort_tool_results(messages: list[dict]) -> list[dict]:
    """llama.cpp's deepseek_v4_sort_tool_results: each run of user/tool messages (they render as one user block) has
    its tool results in the order of the calls of the last assistant turn with calls, matched by tool_call_id (an
    unknown id counts as 0; a stable sort; user messages stay where they are).  DeepSeek's reference encoder does so."""
    out, order, i = list(messages), {}, 0
    while i < len(out):
        m = out[i]
        role = m.get("role") if isinstance(m, dict) else None
        if role == "assistant" and isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
            order = {c["id"]: k for k, c in enumerate(m["tool_calls"])
                     if isinstance(c, dict) and isinstance(c.get("id"), str) and c["id"]}
            i += 1
            continue
        if role not in ("user", "tool"):
            i += 1
            continue
        end, pos = i, []
        while end < len(out) and isinstance(out[end], dict) and out[end].get("role") in ("user", "tool"):
            if out[end].get("role") == "tool":
                pos.append(end)
            end += 1
        if len(pos) > 1 and order:
            results = sorted((out[p] for p in pos), key=lambda r: order.get(r.get("tool_call_id") or "", 0))
            for p, r in zip(pos, results):
                out[p] = r
        i = end
    return out


def _flat(m):
    """A message whose content is a list of text items (a tool picture turned into a note) -> its text, joined the
    llama.cpp way."""
    content = m.get("content") if isinstance(m, dict) else None
    if not isinstance(content, list):
        return m
    return {**m, "content": _join_texts([p.get("text", "") for p in content if isinstance(p, dict)
                                         and p.get("type") in ("text", "input_text", None)], "\n")}


def _continued(messages) -> dict | None:
    """The trailing assistant message the reply continues (llama.cpp's prefill_assistant, AUTO): one without calls."""
    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") == "assistant" and not last.get("tool_calls"):
        return last
    return None


# ------------------------------------------------------------------------------------------------ template
class DeepSeekTemplate(ChatTemplate):
    """The GGUF's chat template, rendered with what llama.cpp passes it.  The class attributes are what the server's
    hooks read (frontend._family, stop_ids_of, Service.prepare / run / vram, main)."""
    family = FAMILY
    stop_token_keys = ("tokenizer.ggml.eos_token_id",)   # {1}: llama.cpp's end-of-generation set for this vocab
    effort_table = DS_EFFORT
    part_joiner = "\n"               # llama.cpp concat_content_parts
    late_system_to_user = False      # the template gathers every system message at the top (llama.cpp parity)
    keep_tool_ids = True             # sort_tool_results needs them
    reasoning_close = THINK_END      # #1053: the template's form is reasoning</think>answer
    reasoning_wrap_up = REASONING_WRAP_UP
    vram_command = False             # the DS4 engine has no VRAM command: POST /v1/vram is refused, nothing is sent
    output_parser = None             # DeepSeekOutputParser (set below its class)

    def __init__(self, path, tokenizer=None):
        special = getattr(tokenizer, "special_ids", None) or {}

        def text_of(key, default):
            try:
                return tokenizer.decode([special[key]]) if key in special else default
            except Exception:  # noqa: BLE001 - the literal is the right answer for every DS4 vocabulary
                return default
        # before the base class compiles the template and runs its capability probes through render()
        self.bos_token = text_of("tokenizer.ggml.bos_token_id", BOS)
        self.eos_token = text_of("tokenizer.ggml.eos_token_id", EOS)
        super().__init__(path)

    def extra_filters(self) -> dict:
        return {"from_json": _from_json}

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs) -> str:
        if images_of(messages):
            raise ValueError("this model reads text only (DeepSeek-V4-Flash has no image encoder): send the request "
                             "without images")
        kw = dict(kwargs)
        thinking = kw.pop("enable_thinking", True) is not False      # llama.cpp's default: thinking on
        effort = kw.pop("reasoning_effort", None)
        kw["enable_thinking"] = thinking
        if thinking and effort:
            kw["reasoning_effort"] = {"xhigh": "high"}.get(effort, effort)
        kw.setdefault("bos_token", self.bos_token)
        kw.setdefault("eos_token", self.eos_token)
        tools = normalize_tools(tools)
        messages = sort_tool_results([_flat(m) for m in messages])
        last = _continued(messages) if add_generation_prompt else None
        if last is None:
            return super().render(messages, tools, add_generation_prompt, **kw)
        # llama.cpp (deepseek.cpp, V4): the rest without a generation prompt, then the assistant turn left open
        head = super().render(messages[:-1], tools, False, **kw)
        reasoning, content = last.get("reasoning_content") or "", last.get("content") or ""
        if not reasoning:
            return head + ASSISTANT + THINK_END + content
        return head + ASSISTANT + THINK_START + reasoning + (THINK_END + content if content else "")

    def thinking_of(self, messages, kwargs) -> bool:
        """Does the prompt end inside the thinking (the parser starts in the reasoning)?  A continued assistant turn:
        only when it has reasoning and no content; otherwise unless the request turned thinking off."""
        last = _continued(messages)
        if last is not None:
            content = last.get("content")
            text = content if isinstance(content, str) else _flat(last).get("content") if content else ""
            return bool(last.get("reasoning_content")) and not text
        return kwargs.get("enable_thinking", True) is not False

    def call_opening(self, name: str | None) -> str:
        """forced_call's text for this format: the template's blank line, the block and the invoke, named or not."""
        return SEP + START + "\n" + INVOKE_OPEN + (name + '">\n' if name else "")

    def _detect_caps(self) -> dict[str, bool]:
        """The base class's probes, reading DSML: tools, calls, parallel calls and the system role hold; earlier
        reasoning is dropped when no tools are present (preserve_reasoning False) - stack 73's /props, exactly."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception:  # noqa: BLE001 - a hint only
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}} for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "", "tool_calls": [{"function": c} for c in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            prompt = render(messages + [{"role": "tool", "content": r} for r in replies] + [user], tools)
            found = []
            for m in re.finditer(re.escape(START) + "(.*?)" + re.escape(END), prompt, re.S):
                if INVOKE_OPEN + "strata_caps_call_" in m.group(1):   # not the instructions' $TOOL_NAME example
                    try:
                        found += calls_of(m.group(1))
                    except ValueError:
                        return False
            return [{"name": c.name, "arguments": c.arguments} for c in found] == calls \
                and all(r in prompt for r in replies)

        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        return {"supports_tools": all(s in tool_prompt for s in ("strata_caps_call_0", "strata_caps_description",
                                                                  START)),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}


# ------------------------------------------------------------------------------------------------ DSML
NAME_MAX = 256                                       # longest tool / parameter name taken
_NAME = re.compile('([^"<>]{1,%d})">' % NAME_MAX)
_KEY = re.compile('([^"<>]{1,%d})" string="(true|false)">' % NAME_MAX)
_NAME_SPAN = NAME_MAX + len('">')
_KEY_SPAN = NAME_MAX + len('" string="false">')
_WS = " \t\r\n"


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return i


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")             # NaN / Infinity: json.loads takes them, clients do not


def _value(raw: str, string: bool):
    """A parameter's value: string="true" is the text as written; "false" is JSON - or, when it is not, the text
    (v0.1.41's rule for Qwen's untyped values: one bad value does not cost the call)."""
    if string:
        return raw
    try:
        return json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return raw


class _Walker:
    """Reads one DSML block's body from `pos` (just after START) as the text grows; `step(text)` returns None (more
    text needed), True (the block ended: `end` is just past END, `calls` the calls) or False (malformed: `bad` is
    where the element that failed starts).  Each decision depends only on the text, never on how it arrived, and
    only new text is scanned.

    A value ends at the first PARAM_CLOSE followed (after whitespace) by the next PARAM_OPEN or INVOKE_CLOSE - the
    form the template writes (#210's rule for Qwen) - so a value may hold END, INVOKE_CLOSE or a lone PARAM_CLOSE:
    a file that documents DSML (or this parser) can be written through a call."""

    def __init__(self, pos: int):
        self.pos, self.state, self.calls = pos, "invoke", []
        self.elem, self.end, self.bad = pos, -1, -1
        self.name, self.args, self.key, self.string = None, None, None, True
        self.vstart = self.vscan = 0

    def _fail(self, at: int) -> bool:
        self.bad = at
        return False

    def step(self, text: str):
        while True:
            if self.state == "invoke":                       # an invoke, or (after one) the block's end
                p = self.pos = _skip_ws(text, self.pos)
                if text.startswith(INVOKE_OPEN, p):
                    self.elem, self.pos, self.state = p, p + len(INVOKE_OPEN), "name"
                    continue
                if self.calls and text.startswith(END, p):
                    self.end = p + len(END)
                    return True
                tail = text[p:p + len(INVOKE_OPEN)]
                if INVOKE_OPEN.startswith(tail) or (self.calls and END.startswith(tail)):
                    return None
                return self._fail(p)
            if self.state == "name":
                m = _NAME.match(text, self.pos)
                if m is None:
                    window = text[self.pos:self.pos + _NAME_SPAN]
                    return self._fail(self.elem) if ">" in window or len(window) >= _NAME_SPAN else None
                self.name, self.args, self.pos, self.state = m.group(1), {}, m.end(), "params"
                continue
            if self.state == "params":                       # a parameter, or the invoke's end
                p = self.pos = _skip_ws(text, self.pos)
                if text.startswith(PARAM_OPEN, p):
                    self.elem, self.pos, self.state = p, p + len(PARAM_OPEN), "key"
                    continue
                if text.startswith(INVOKE_CLOSE, p):
                    self.calls.append(ToolCall(self.name, self.args))
                    self.pos, self.state = p + len(INVOKE_CLOSE), "invoke"
                    continue
                tail = text[p:p + len(PARAM_OPEN)]
                if PARAM_OPEN.startswith(tail) or INVOKE_CLOSE.startswith(tail):
                    return None
                return self._fail(p)
            if self.state == "key":
                m = _KEY.match(text, self.pos)
                if m is None:
                    window = text[self.pos:self.pos + _KEY_SPAN]
                    return self._fail(self.elem) if ">" in window or len(window) >= _KEY_SPAN else None
                if m.group(1) in self.args:                  # a duplicate parameter (aleksche)
                    return self._fail(self.elem)
                self.key, self.string = m.group(1), m.group(2) == "true"
                self.pos = self.vstart = self.vscan = m.end()
                self.state = "value"
                continue
            # value
            while True:
                c = text.find(PARAM_CLOSE, self.vscan)
                if c < 0:
                    self.vscan = max(self.vstart, len(text) - len(PARAM_CLOSE) + 1)
                    return None
                after = _skip_ws(text, c + len(PARAM_CLOSE))
                if text.startswith(PARAM_OPEN, after) or text.startswith(INVOKE_CLOSE, after):
                    self.args[self.key] = _value(text[self.vstart:c], self.string)
                    self.pos, self.state = c + len(PARAM_CLOSE), "params"
                    break
                tail = text[after:after + len(PARAM_OPEN)]
                if PARAM_OPEN.startswith(tail) or INVOKE_CLOSE.startswith(tail):
                    self.vscan = c                           # its follower is still arriving: look again then
                    return None
                self.vscan = c + 1                           # part of the value


def calls_of(body: str) -> list[ToolCall]:
    """The calls of a whole DSML block's body (between START and END).  ValueError when it is not exactly one valid
    block: malformed or truncated DSML is text, never a partial call (aleksche)."""
    w = _Walker(0)
    text = body + END
    if w.step(text) is not True or w.end != len(text):
        raise ValueError("malformed DSML tool call block")
    return w.calls


def _split_sep(text: str) -> tuple[str, str]:
    """Text before a block -> (kept, separator): exactly one "\\n\\n" goes with the block (llama.cpp's
    until_one_of({"\\n\\n" + START, START}))."""
    return (text[:-len(SEP)], SEP) if text.endswith(SEP) else (text, "")


class DeepSeekOutputParser:
    """The reply's text -> reasoning, content and tool-call events, fed in pieces (OutputParser's interface, which
    Service.run uses: feed, finish(reason), state, buf, pending, rescued, refused).  States: "reasoning" ->
    ("</think>") "content"; "block" = a DSML block in the answer; "rcall" = one inside the reasoning.  DSML is read
    only when the request declares tools (llama.cpp builds no tool parser otherwise).

    * A block's calls are sent whole once it is complete and valid ("tool_call" events, no tool_start/tool_args), so
      malformed DSML can always go out as the text it is: from its separator to the element that failed, and the rest
      is read again.  Undeclared names are delivered, as upstream's Qwen parser does.  `holding` is True meanwhile
      (Service.run sends keep-alives).
    * A block inside the reasoning counts only when the output ends with it (llama.cpp's reasoning_with_tc): its calls
      wait in `pending` with the state kept at "rcall" (so #1053's close-and-retry never fires on them), whitespace or
      another block keeps them, and finish("stop") delivers them (`rescued`).  Anything else after them - text,
      "</think>" (unlike upstream's Qwen rule), a cut by max tokens - makes all of it reasoning text (`refused`).  So
      does an undeclared name, malformed DSML, "</think>" inside the block, or a block over RCALL_MAX characters.
    * A partial marker is held, with up to two newlines in front of it, so streamed and whole outputs agree."""
    _hold = OutputParser._hold

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False                # drop the newlines that follow "</think>" or a block
        self.names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
        self.dsml = bool(self.names)
        self.pending: list[list] = []    # [raw text, [ToolCall]] of complete blocks inside the reasoning
        self.rescued = 0                 # calls delivered from the reasoning
        self.refused = 0                 # parsed calls kept as reasoning text
        self.walker: _Walker | None = None
        self.holding = False             # a block is being read: nothing of it has gone out yet

    def _held(self, text: str, tags) -> int:
        """How much of the end of `text` to keep back: a partial marker and up to two newlines before it."""
        k = self._hold(text, tags)
        j, n = len(text) - k, 0
        while n < 2 and j - n > 0 and text[j - n - 1] == "\n":
            n += 1
        return k + n

    def _release(self, deliver: bool) -> list[Event]:
        out = []
        for raw, calls in self.pending:
            if deliver:
                out += [Event("tool_call", call=c) for c in calls]
                self.rescued += len(calls)
            else:
                self.refused += len(calls)
                out.append(Event("reasoning", raw))
        self.pending = []
        return out

    def _open(self, state: str, at: int, kept: str, sep: str, out: list):
        """A block starts at self.buf[at] (START): `kept` goes out, the separator stays with the block."""
        if kept:
            out.append(Event("reasoning" if state == "rcall" else "content", kept))
        self.buf = sep + self.buf[at:]
        self.walker = _Walker(len(sep) + len(START))
        self.state = state

    def _back(self, upto: int, kind: str, out: list):
        """self.buf[:upto] goes out as `kind` text; the rest is read again in the reasoning or the answer."""
        if upto:
            out.append(Event(kind, self.buf[:upto]))
        self.buf = self.buf[upto:]
        self.walker = None
        self.state = kind

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while self._step(out):
            pass
        self.holding = self.walker is not None
        return out

    def _step(self, out: list) -> bool:
        """One move of the state machine: True to go on, False when it needs more text."""
        if self.state == "reasoning":
            i = self.buf.find(THINK_END)
            s = self.buf.find(START) if self.dsml else -1
            if s >= 0 and (i < 0 or s < i):
                kept, sep = _split_sep(self.buf[:s])
                self._open("rcall", s, kept, sep, out)
                return True
            if i < 0:
                keep = self._held(self.buf, (THINK_END, START)) if self.dsml else self._hold(self.buf, (THINK_END,))
                if len(self.buf) > keep:
                    out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                return False
            if i:
                out.append(Event("reasoning", self.buf[:i]))
            self.buf = self.buf[i + len(THINK_END):]
            self.state, self.lead = "content", True
            return True
        if self.state == "content":
            if self.lead:
                stripped = self.buf.lstrip("\n")
                if not stripped:
                    self.buf = ""
                    return False
                self.buf, self.lead = stripped, False
            s = self.buf.find(START) if self.dsml else -1
            if s < 0:
                keep = self._held(self.buf, (START,)) if self.dsml else 0
                if len(self.buf) > keep:
                    out.append(Event("content", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                return False
            kept, sep = _split_sep(self.buf[:s])
            self._open("block", s, kept, sep, out)
            return True
        if self.state == "block":
            r = self.walker.step(self.buf)
            if r is None:
                return False
            if r:
                out += [Event("tool_call", call=c) for c in self.walker.calls]
                self.buf = self.buf[self.walker.end:]
                self.walker, self.state, self.lead = None, "content", True
            else:
                self._back(self.walker.bad, "content", out)
            return True
        # rcall
        if self.walker is None:                              # complete blocks wait: what comes after them?
            stripped = self.buf.lstrip(_WS)
            if len(stripped) < len(self.buf):
                self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                self.buf = stripped
            if not self.buf or (START.startswith(self.buf) and not self.buf.startswith(START)):
                return False
            if self.buf.startswith(START):                   # another block
                self.walker = _Walker(len(START))
                return True
            out += self._release(False)                      # text or "</think>": they were not acts
            self.state = "reasoning"
            return True
        # a block inside the reasoning: only its first RCALL_MAX characters are looked at (and the end marker
        # straddling them), so the decision is the same however the text arrives
        r = self.walker.step(self.buf[:RCALL_MAX])
        limit = self.walker.end if r is True else self.walker.bad if r is False else \
            min(len(self.buf), RCALL_MAX + len(THINK_END) - 1)
        t = self.buf.find(THINK_END, len(START), limit)
        if t >= 0:                                           # the thinking ended inside it: never a call
            out += self._release(False)
            self._back(t, "reasoning", out)
            return True
        if r is None:
            if len(self.buf) < RCALL_MAX + len(THINK_END) - 1:
                return False
            out += self._release(False)                      # too long to be held: reasoning text
            self._back(RCALL_MAX, "reasoning", out)
            return True
        if r is False:
            out += self._release(False)
            self._back(self.walker.bad, "reasoning", out)
            return True
        calls = self.walker.calls
        if all(c.name in self.names for c in calls):
            self.pending.append([self.buf[:self.walker.end], calls])
            self.buf = self.buf[self.walker.end:]
            self.walker = None                               # waiting (state stays "rcall")
        else:                                                # an undeclared name inside the thinking: a quote
            out += self._release(False)
            self._back(self.walker.end, "reasoning", out)
        return True

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of the output.  Calls waiting from the reasoning are delivered only on a natural stop (None / "stop")
        with nothing after them; an unfinished block is its text (content, or reasoning inside the thinking)."""
        out = self.feed("")
        if self.state == "rcall" and self.walker is None:
            deliver = reason in (None, "stop") and not self.buf.strip(_WS)
            out += self._release(deliver)
            if self.buf:
                out.append(Event("reasoning", self.buf))
        elif self.state == "rcall":
            out += self._release(False)
            if self.buf:
                out.append(Event("reasoning", self.buf))
        elif self.buf:
            out.append(Event("content" if self.state == "block" else self.state, self.buf))
        self.buf, self.walker, self.holding = "", None, False
        if self.state in ("rcall", "block"):
            self.state = "reasoning" if self.state == "rcall" else "content"
        return out


DeepSeekTemplate.output_parser = DeepSeekOutputParser
