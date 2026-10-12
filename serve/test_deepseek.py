"""mab776 M4: the DeepSeek-V4-Flash-0731 family (serve/deepseek.py) and the server hooks it uses.

    python -m unittest serve.test_deepseek -v

CPU only, no pack: a byte tokenizer with DeepSeek's specials stands in for the vocabulary.  STRATA_DS4_TOKENIZER (a DS4
pack's tokenizer directory, e.g. bench/mab776-m2/work/ds4tok/tokenizer) turns on the tests on the real one.
Parser shapes adapted from Jackwwg83/Strata-DS serve/test_deepseek.py (main, MIT) and Aleks-Che/Strata
serve/test_deepseek.py (dev, MIT): every split point, malformed DSML as text, the parser interface, the budget pass."""
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from serve import deepseek, frontend as F, server  # noqa: E402
from serve.deepseek import (ASSISTANT, BOS, DSML, END, EOS, INVOKE_CLOSE, INVOKE_OPEN, PARAM_CLOSE,  # noqa: E402
                            PARAM_OPEN, START, THINK_END, THINK_START, USER, DeepSeekOutputParser, DeepSeekTemplate)
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, DeepSeekEngine, MockEngine, Service, StrataEngine, serve  # noqa: E402

FIXTURE = ROOT / "serve/fixtures/ds4/chat_template.jinja"     # the GGUF's chat template (Unsloth, Apache-2.0)
QWEN = ROOT / "serve/chat_template.jinja"
M2 = ROOT / "bench/mab776-m2"                                 # stack 73's captured renders (llama.cpp b11429)
DS4_TOKENIZER = os.environ.get("STRATA_DS4_TOKENIZER")
CTX = 8192


class DSByteTokenizer(ByteTokenizer):
    """One id per byte; DeepSeek-V4's specials from 256.  <think>, </think> and ｜DSML｜ are matched without
    parse_special (GGUF type 4), the others are control tokens (type 3); special_ids name bos and eos."""
    SPECIALS = [BOS, EOS, USER, ASSISTANT, THINK_START, THINK_END, DSML, "<|EOT|>", "<｜place▁holder▁no▁7｜>"]
    ALWAYS = (THINK_START, THINK_END, DSML)
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}


TOK = DSByteTokenizer()
TPL = DeepSeekTemplate(FIXTURE, TOK)
EOS_ID = 257
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                            "parameters": {"type": "object",
                                                           "properties": {"city": {"type": "string"},
                                                                          "unit": {"type": "string",
                                                                                   "enum": ["c", "f"]}},
                                                           "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search_files", "description": "Search files by glob and regex",
                                           "parameters": {"type": "object",
                                                          "properties": {"glob": {"type": "string"},
                                                                         "pattern": {"type": "string"},
                                                                         "max_results": {"type": "integer"},
                                                                         "flags": {"type": "array",
                                                                                   "items": {"type": "string"}}},
                                                          "required": ["glob", "pattern"]}}}
WRITE = {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}}}}}
OPENAI_TOOLS = [WEATHER, SEARCH, WRITE]
TOOLS = [t["function"] for t in OPENAI_TOOLS]                 # the normalizers' (unwrapped) form


def invoke(name, **args):
    """One invoke exactly as the template writes it (strings raw, the rest JSON)."""
    out = f'{INVOKE_OPEN}{name}">\n'
    for k, v in args.items():
        if isinstance(v, str):
            out += f'{PARAM_OPEN}{k}" string="true">{v}{PARAM_CLOSE}\n'
        else:
            out += f'{PARAM_OPEN}{k}" string="false">{json.dumps(v, ensure_ascii=False)}{PARAM_CLOSE}\n'
    return out + INVOKE_CLOSE + "\n"


def block(*invokes):
    return START + "\n" + "".join(invokes) + END


def summary(events):
    """The events with adjacent text of one kind merged: [("reasoning" | "content", text) | ("call", (name, args))]."""
    out = []
    for ev in events:
        if ev.kind in ("reasoning", "content"):
            if not ev.text:
                continue
            if out and out[-1][0] == ev.kind:
                out[-1] = (ev.kind, out[-1][1] + ev.text)
            else:
                out.append((ev.kind, ev.text))
        elif ev.kind == "tool_call":
            out.append(("call", (ev.call.name, ev.call.arguments)))
        else:
            out.append((ev.kind, None))                       # never expected: whole calls only
    return out


def parse(parts, thinking=True, tools=TOOLS, reason="stop"):
    """-> (summary, (rescued, refused)) of `parts` fed in order (a string: fed whole)."""
    p = DeepSeekOutputParser(thinking=thinking, tools=tools, stream_tools=True)
    events = []
    for part in [parts] if isinstance(parts, str) else parts:
        events += p.feed(part)
    events += p.finish(reason)
    return summary(events), (p.rescued, p.refused)


def chunkings(text, rng, n=4):
    """Random ways to cut `text` into pieces (and char by char)."""
    yield list(text)
    for _ in range(n):
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 12)))) if len(text) > 1 else []
        yield [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]


# ------------------------------------------------------------------------------------------------ renders
class CapturedRenders(unittest.TestCase):
    """The 14 conversations captured from stack 73 (llama.cpp b11429, the same GGUF template): the request bodies go
    through the OpenAI normalizer and the template; 13 are byte-equal, c06b (xhigh) differs by design."""

    def cases(self):
        for conv in sorted((M2 / "conv").glob("*.json")):
            want = json.loads((M2 / "renders" / (conv.stem + ".render.json")).read_text(encoding="utf-8"))["prompt"]
            yield conv.stem, json.loads(conv.read_text(encoding="utf-8")), want

    def test_the_fixture_is_the_ggufs_template(self):
        props = json.loads((M2 / "renders/props.json").read_text(encoding="utf-8"))
        self.assertEqual(FIXTURE.read_text(encoding="utf-8"), props["chat_template"])

    def test_14_captures_byte_equal_but_xhigh(self):
        cases = list(self.cases())
        self.assertEqual(len(cases), 14)
        rendered = {}
        for name, req, want in cases:
            messages, tools, kw = F.openai_to_messages(req, TPL)
            got = TPL.render(messages, tools, **kw)
            rendered[name] = got
            with self.subTest(name):
                if name.startswith("c06b"):
                    self.assertNotEqual(got, want)             # llama.cpp passes "xhigh" on: no preamble there
                else:
                    self.assertEqual(got, want)
        # by design: xhigh renders as high
        self.assertEqual(rendered["c06b-effort-xhigh"], rendered["c05-effort-high"])

    def test_the_servers_prompt_is_the_capture(self):
        """Service.encode_prompt (literal tags, encode with specials) decodes back to the capture."""
        svc = Service(MockEngine(TOK, "x", max_context=1 << 20), TOK, TPL)
        for name, req, want in self.cases():
            if name.startswith("c06b"):
                continue
            messages, tools, kw = F.openai_to_messages(req, TPL)
            with self.subTest(name):
                ids = svc.encode_prompt(messages, tools, kw)
                self.assertEqual(TOK.decode(ids), want)
                self.assertEqual(ids, TOK.encode(want, parse_special=True))
                self.assertEqual(ids[:1], [256])                # bos from special_ids

    @unittest.skipUnless(DS4_TOKENIZER, "set STRATA_DS4_TOKENIZER to a DS4 pack's tokenizer directory")
    def test_on_the_real_vocabulary(self):
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(DS4_TOKENIZER)
        tpl = DeepSeekTemplate(FIXTURE, tok)
        svc = Service(MockEngine(tok, "x", max_context=1 << 20), tok, tpl)
        self.assertEqual(svc.stop_ids, {1})
        for name, req, want in self.cases():
            if name.startswith("c06b"):
                continue
            messages, tools, kw = F.openai_to_messages(req, tpl)
            with self.subTest(name):
                ids = svc.encode_prompt(messages, tools, kw)
                self.assertEqual(ids, tok.encode(want, parse_special=True))
        c01 = json.loads((M2 / "conv/c01-plain.json").read_text())
        self.assertEqual(svc.encode_prompt(*F.openai_to_messages(c01, tpl))[:2], [0, 128803])
        self.assertEqual(tok.decode([128825]), DSML)              # a special decodes as its text (M3)


# ------------------------------------------------------------------------------------------------ normalizers
PREAMBLE_HIGH = "Reasoning Effort: Absolute maximum"
PREAMBLE_MAX = "Reasoning Effort: Beyond maximum"


def preamble(prompt):
    return "high" if PREAMBLE_HIGH in prompt else "max" if PREAMBLE_MAX in prompt else None


class Normalizers(unittest.TestCase):
    USER = [{"role": "user", "content": "q"}]

    def render(self, req, api="openai"):
        if api == "openai":
            m, t, kw = F.openai_to_messages({"messages": self.USER, **req}, TPL)
        elif api == "anthropic":
            m, t, kw = F.anthropic_to_messages({"messages": self.USER, **req}, True, TPL)
        else:
            from serve import responses
            m = responses.input_messages({"input": "q", **req}, TPL)
            t, kw = None, responses.template_kwargs(req, {}, TPL)
        return TPL.render(m, t, **kw)

    def test_effort_through_every_api(self):
        want = {None: (None, True), "none": (None, False), "low": (None, True), "medium": (None, True),
                "high": ("high", True), "xhigh": ("high", True), "max": ("max", True), "maximum": ("max", True)}
        for word, (pre, thinks) in want.items():
            for api, req in (("openai", {"reasoning_effort": word} if word else {}),
                             ("openai", {"chat_template_kwargs": {"reasoning_effort": word}} if word else {}),
                             ("anthropic", {"output_config": {"effort": word}} if word else {}),
                             ("responses", {"reasoning": {"effort": word}} if word else {})):
                with self.subTest(api=api, effort=word, req=req):
                    prompt = self.render(req, api)
                    self.assertEqual(preamble(prompt), pre)
                    self.assertTrue(prompt.endswith(THINK_START if thinks else THINK_END), prompt[-40:])

    def test_anthropic_budget_and_disabled(self):
        for budget, pre in ((1024, None), (4096, None), (16000, "high")):
            self.assertEqual(preamble(self.render({"thinking": {"type": "enabled", "budget_tokens": budget}},
                                                  "anthropic")), pre)
        self.assertTrue(self.render({"thinking": {"type": "disabled"}}, "anthropic").endswith(ASSISTANT + THINK_END))

    def test_shared_settings_effort(self):
        svc = Service(MockEngine(TOK, "x", max_context=CTX), TOK, TPL)
        svc.set_shared({"reasoning_effort": "high"})
        req = svc.with_shared({"messages": self.USER}, "openai")
        m, t, kw = F.openai_to_messages(req, TPL)
        self.assertEqual(preamble(TPL.render(m, t, **kw)), "high")

    def test_qwen_keeps_its_own_map(self):
        for word in ("high", "xhigh", "max"):
            self.assertEqual(F.openai_to_messages({"messages": self.USER, "reasoning_effort": word})[2],
                             {"reasoning_effort": "xhigh"})
        self.assertEqual(F.budget_effort(16000), {"reasoning_effort": "xhigh"})

    def test_text_parts_are_joined_the_llama_cpp_way(self):
        parts = lambda *t: [{"type": "text", "text": x} for x in t]  # noqa: E731
        for given, want in ((parts("a", "b"), "a\nb"), (parts("", "b"), "b"), (parts("a", "", "b"), "a\n\nb")):
            m, _, _ = F.openai_to_messages({"messages": [{"role": "user", "content": given}]}, TPL)
            self.assertEqual(m[0]["content"], want)
            self.assertEqual(F.openai_to_messages({"messages": [{"role": "user", "content": given}]})[0][0]["content"],
                             "".join(p["text"] for p in given))     # Qwen: unchanged
        m, _, _ = F.anthropic_to_messages({"messages": [{"role": "user", "content": parts("x", "y")}]}, True, TPL)
        self.assertEqual(m[0]["content"], "x\ny")

    def test_late_system_messages_stay_system(self):
        req = {"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"},
                            {"role": "user", "content": "b"}]}
        self.assertEqual(F.openai_to_messages(req, TPL)[0][1]["role"], "system")
        self.assertEqual(F.openai_to_messages(req)[0][1]["role"], "user")          # Qwen: as before
        prompt = TPL.render(*F.openai_to_messages(req, TPL)[:2])
        self.assertTrue(prompt.startswith(BOS + "late" + USER + "a"), prompt)      # hoisted by the template

    def test_tool_results_follow_the_call_order(self):
        calls = [{"id": "a", "type": "function", "function": {"name": "search_files", "arguments": "{}"}},
                 {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
        req = {"messages": [{"role": "user", "content": "go"},
                            {"role": "assistant", "content": "", "tool_calls": calls},
                            {"role": "tool", "tool_call_id": "b", "content": "B"},
                            {"role": "tool", "tool_call_id": "a", "content": "A"}], "tools": [WEATHER, SEARCH]}
        prompt = TPL.render(*F.openai_to_messages(req, TPL)[:2])
        self.assertIn("<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>", prompt)
        anth = {"messages": [{"role": "user", "content": "go"},
                             {"role": "assistant", "content": [
                                 {"type": "tool_use", "id": "a", "name": "search_files", "input": {}},
                                 {"type": "tool_use", "id": "b", "name": "get_weather", "input": {}}]},
                             {"role": "user", "content": [
                                 {"type": "tool_result", "tool_use_id": "b", "content": "B"},
                                 {"type": "tool_result", "tool_use_id": "a", "content": "A"}]}]}
        prompt = TPL.render(*F.anthropic_to_messages(anth, True, TPL)[:2])
        self.assertIn("<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>", prompt)
        # without ids (the MCP loop, Qwen's normalizer): the order as given
        unordered = [{"role": "tool", "content": "B"}, {"role": "tool", "content": "A"}]
        self.assertEqual(deepseek.sort_tool_results(unordered), unordered)

    def test_tools_are_re_serialized_like_llama_cpp(self):
        messy = {"type": "function", "function": {"strict": True, "parameters": {"type": "object"}, "name": "f"}}
        prompt = TPL.render(self.USER, [messy["function"]])
        self.assertIn('\n{"name": "f", "description": "", "parameters": {"type": "object"}}\n', prompt)
        self.assertIn('{"name": "g", "description": "", "parameters": {}}', TPL.render(self.USER, [{"name": "g"}]))

    def test_from_json_and_dict_arguments_render_alike(self):
        def history(args):
            return [{"role": "user", "content": "go"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "f", "arguments": args}}]},
                    {"role": "tool", "content": "ok"}]
        tools = [{"name": "f", "parameters": {}}]
        self.assertEqual(TPL.render(history('{"a": "x", "n": 5}'), tools),
                         TPL.render(history({"a": "x", "n": 5}), tools))
        with self.assertRaises(ValueError):                       # a 400, not a dropped connection
            TPL.render(history("{not json"), tools)

    def test_prefill_continuations(self):
        head = [{"role": "user", "content": "q"}]
        for last, tail, thinks in (({"content": "Part"}, ASSISTANT + THINK_END + "Part", False),
                                   ({"content": "", "reasoning_content": "R"}, ASSISTANT + THINK_START + "R", True),
                                   ({"content": "C", "reasoning_content": "R"},
                                    ASSISTANT + THINK_START + "R" + THINK_END + "C", False),
                                   ({"content": ""}, ASSISTANT + THINK_END, False)):
            messages = head + [{"role": "assistant", **last}]
            with self.subTest(last=last):
                prompt = TPL.render(messages)
                self.assertTrue(prompt.endswith(USER + "q" + tail), prompt[-60:])
                self.assertEqual(TPL.thinking_of(messages, {}), thinks)
        # thinking off by request, no continuation: not thinking
        self.assertFalse(TPL.thinking_of(head, {"enable_thinking": False}))
        self.assertTrue(TPL.thinking_of(head, {}))

    def test_images_are_refused_and_tool_pictures_become_notes(self):
        svc = Service(MockEngine(TOK, "x", max_context=CTX), TOK, TPL)
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        m, t, kw = F.openai_to_messages({"messages": [{"role": "user", "content": [img]}]}, TPL)
        with self.assertRaises(ValueError):
            svc.prepare(m, t, kw)
        tool = {"role": "tool", "content": [img, {"type": "text", "text": "t"}]}
        m, t, kw = F.openai_to_messages({"messages": [{"role": "user", "content": "look"}, tool]}, TPL)
        ids, thinking, _ = svc.prepare(m, t, kw)
        self.assertIn("<tool_result>[image omitted: this server has no image encoder]\nt</tool_result>",
                      TOK.decode(ids))

    def test_caps_match_stack_73(self):
        props = json.loads((M2 / "renders/props.json").read_text(encoding="utf-8"))["chat_template_caps"]
        for k, v in TPL.caps.items():
            self.assertEqual(v, props[k], k)


# ------------------------------------------------------------------------------------------------ parser
class Parser(unittest.TestCase):
    def assert_stable(self, text, thinking=True, tools=TOOLS, reason="stop", seed=0):
        """Whole == every two-piece split == char by char == random pieces; returns the whole result."""
        whole = parse(text, thinking, tools, reason)
        for i in range(len(text) + 1):
            self.assertEqual(parse([text[:i], text[i:]], thinking, tools, reason), whole, f"split at {i}")
        for pieces in chunkings(text, random.Random(seed)):
            self.assertEqual(parse(pieces, thinking, tools, reason), whole, pieces)
        return whole

    def test_reasoning_content_and_two_typed_calls(self):
        calls = block(invoke("search_files", glob="*.py", pattern="TODO", max_results=5, flags=["-i"]),
                      invoke("get_weather", city="Montréal"))
        text = "Two lookups.</think>Checking both.\n\n" + calls
        got, counts = self.assert_stable(text)
        self.assertEqual(got, [("reasoning", "Two lookups."), ("content", "Checking both."),
                               ("call", ("search_files", {"glob": "*.py", "pattern": "TODO", "max_results": 5,
                                                          "flags": ["-i"]})),
                               ("call", ("get_weather", {"city": "Montréal"}))])
        self.assertEqual(counts, (0, 0))

    def test_empty_arguments_and_json_values(self):
        body = (f'{INVOKE_OPEN}write">\n{INVOKE_CLOSE}\n' +
                f'{INVOKE_OPEN}write">\n{PARAM_OPEN}a" string="false">not json{PARAM_CLOSE}\n'
                f'{PARAM_OPEN}b" string="false">NaN{PARAM_CLOSE}\n'
                f'{PARAM_OPEN}c" string="false">{{"x": [1, null, true]}}{PARAM_CLOSE}\n'
                f'{PARAM_OPEN}d" string="true"> spaced \n{PARAM_CLOSE}\n{INVOKE_CLOSE}\n')
        got, _ = self.assert_stable(START + "\n" + body + END, thinking=False)
        self.assertEqual(got, [("call", ("write", {})),
                               ("call", ("write", {"a": "not json", "b": "NaN", "c": {"x": [1, None, True]},
                                                   "d": " spaced \n"}))])

    def test_values_may_hold_dsml_markup(self):
        value = f"doc: {END} and {INVOKE_CLOSE}, a lone {PARAM_CLOSE} here, {START} too\n{INVOKE_OPEN}x\">"
        got, _ = self.assert_stable("</think>" + block(invoke("write", path="a.md", content=value)))
        self.assertEqual(got, [("call", ("write", {"path": "a.md", "content": value}))])

    def test_malformed_dsml_is_the_text_it_is(self):
        bad = [START + "\n" + END,                                                   # no invoke
               block(f'{INVOKE_OPEN}write">\njunk{INVOKE_CLOSE}\n'),                 # stray text in an invoke
               block(invoke("write", a="1") + "junk\n" + invoke("write", b="2")),   # stray text between invokes
               block(f'{INVOKE_OPEN}write">\n{PARAM_OPEN}a">x{PARAM_CLOSE}\n{INVOKE_CLOSE}\n'),   # no string=
               block(f'{INVOKE_OPEN}write">\n{PARAM_OPEN}a" string="true">1{PARAM_CLOSE}\n'
                     f'{PARAM_OPEN}a" string="true">2{PARAM_CLOSE}\n{INVOKE_CLOSE}\n'),          # duplicate parameter
               block(f'{INVOKE_OPEN}">\n{INVOKE_CLOSE}\n'),                         # empty name
               block(f'{INVOKE_OPEN}a<b">\n{INVOKE_CLOSE}\n'),                      # bad name
               START + "\n" + invoke("write", a="1"),                                # cut before its end
               START + " is how I call tools",                                       # prose naming the opener
               START + "\n" + START + "\n" + invoke("write", a="1")[:-1]]            # two openers, unfinished
        for b in bad:
            for lead in ("", "Hi.", "Hi.\n\n"):
                text = lead + b
                with self.subTest(text=text):
                    got, _ = self.assert_stable(text, thinking=False)
                    self.assertEqual(got, [("content", text)])

    def test_a_bad_block_then_a_good_one(self):
        text = START + " oops\n\n" + block(invoke("write", a="1"))
        got, _ = self.assert_stable(text, thinking=False)
        self.assertEqual(got, [("content", START + " oops"), ("call", ("write", {"a": "1"}))])

    def test_separator_and_bare_opener(self):
        call = block(invoke("write", a="1"))
        for lead, kept in (("abc", "abc"), ("abc\n", "abc\n"), ("abc\n\n", "abc"), ("abc\n\n\n", "abc\n"),
                           ("say ", "say ")):
            got, _ = self.assert_stable(lead + call, thinking=False)
            self.assertEqual(got, [("content", kept), ("call", ("write", {"a": "1"}))], lead)
        self.assertEqual(parse("hello\n\n", thinking=False)[0], [("content", "hello\n\n")])   # held, then kept
        self.assertEqual(parse("hello\n", thinking=False)[0], [("content", "hello\n")])
        got, _ = self.assert_stable(call + "\n\nAfter.", thinking=False)                   # text after a call
        self.assertEqual(got, [("call", ("write", {"a": "1"})), ("content", "After.")])

    def test_no_tools_means_text(self):
        text = "R</think>A\n\n" + block(invoke("write", a="1"))
        got, _ = self.assert_stable(text, tools=None)
        self.assertEqual(got, [("reasoning", "R"), ("content", text[len("R</think>"):])])

    def test_undeclared_names(self):
        text = "R</think>" + block(invoke("nope", a="1"))            # in the answer: delivered (as upstream's Qwen)
        self.assertEqual(self.assert_stable(text)[0], [("reasoning", "R"), ("call", ("nope", {"a": "1"}))])
        text = "R\n\n" + block(invoke("nope", a="1"))                # in the reasoning: a quote
        self.assertEqual(self.assert_stable(text)[0], [("reasoning", text)])

    def test_a_call_inside_the_reasoning(self):
        call = block(invoke("get_weather", city="Oslo"))
        think = "I need the weather.\n\n" + call
        want = ("call", ("get_weather", {"city": "Oslo"}))
        self.assertEqual(self.assert_stable(think), ([("reasoning", "I need the weather."), want], (1, 0)))
        self.assertEqual(self.assert_stable(think + "\n \n"), ([("reasoning", "I need the weather."), want], (1, 0)))
        two = think + "\n" + block(invoke("get_weather", city="Bergen"))
        self.assertEqual(self.assert_stable(two)[0][-1], ("call", ("get_weather", {"city": "Bergen"})))
        self.assertEqual(self.assert_stable(two)[1], (2, 0))
        # what comes after decides: the end of the turn delivers; text, </think> or a cut keeps it reasoning
        self.assertEqual(self.assert_stable(think + "\n\nActually no."), ([("reasoning", think + "\n\nActually no.")],
                                                                           (0, 1)))
        self.assertEqual(self.assert_stable(think + "</think>Done."), ([("reasoning", think), ("content", "Done.")],
                                                                        (0, 1)))
        self.assertEqual(self.assert_stable(think, reason="length"), ([("reasoning", think)], (0, 1)))
        self.assertEqual(self.assert_stable(think, reason="cancel"), ([("reasoning", think)], (0, 1)))

    def test_the_thinking_ending_inside_a_block(self):
        text = "x\n\n" + START + "\n" + INVOKE_OPEN + 'write">\n' + "</think>Answer."
        self.assertEqual(self.assert_stable(text)[0],
                         [("reasoning", text.split("</think>")[0]), ("content", "Answer.")])

    def test_a_block_too_long_to_hold_in_the_reasoning(self):
        with mock.patch.object(deepseek, "RCALL_MAX", 200):
            text = "R\n\n" + block(invoke("write", content="y" * 400)) + "\n</think>Answer."
            got, counts = self.assert_stable(text, seed=3)
            self.assertEqual(got, [("reasoning", text.split("</think>")[0]), ("content", "Answer.")])
            short = "R\n\n" + block(invoke("write", content="y" * 40))
            self.assertEqual(self.assert_stable(short)[0][-1], ("call", ("write", {"content": "y" * 40})))

    def test_the_server_parser_interface(self):
        """Service.run calls finish(reason) and reads state / buf / pending / rescued / refused (jackwwg83)."""
        for reason in (None, "stop", "length", "cancel", "error"):
            p = DeepSeekOutputParser(thinking=False)
            evs = p.feed("hi") + p.finish(reason)
            self.assertEqual([(e.kind, e.text) for e in evs], [("content", "hi")])
            self.assertEqual((p.pending, p.rescued, p.refused, p.buf, p.state), ([], 0, 0, "", "content"))
        p = DeepSeekOutputParser(thinking=True, tools=TOOLS)
        p.feed("thinking " + START + "\n" + INVOKE_OPEN)
        self.assertTrue(p.holding)
        self.assertEqual(p.state, "rcall")
        p.feed('write">\n' + INVOKE_CLOSE + "\n" + END)
        self.assertFalse(p.holding)
        self.assertEqual(p.state, "rcall")                       # waiting: #1053 must not fire on it
        self.assertEqual(len(p.pending), 1)

    def test_fuzz_against_whole_text(self):
        """Random texts of markers, partial markers, newlines and words: every way of cutting them gives the whole
        text's events; with no call delivered, no character is lost or added."""
        frags = ["abc", " ", "\n", "\n\n", THINK_END, THINK_START, START, END, START[:6], END[:7], DSML, "<", ">", '"',
                 INVOKE_OPEN + 'write">', INVOKE_OPEN + 'nope">', INVOKE_OPEN, PARAM_OPEN + 'a" string="true">',
                 PARAM_OPEN + 'n" string="false">', PARAM_OPEN + 'a"', "5", "x y", PARAM_CLOSE, INVOKE_CLOSE,
                 invoke("write", a="v"), block(invoke("write", a="1")), "é", "😀"]
        rng = random.Random(1234)
        for _ in range(300):
            text = "".join(rng.choice(frags) for _ in range(rng.randint(1, 18)))
            for thinking in (True, False):
                for tools in (TOOLS, None):
                    whole = parse(text, thinking, tools)
                    for pieces in chunkings(text, rng, 3):
                        self.assertEqual(parse(pieces, thinking, tools), whole, (text, pieces))
                    events = whole[0]
                    if any(k == "call" for k, _ in events):
                        continue
                    reasoning = "".join(t for k, t in events if k == "reasoning")
                    content = "".join(t for k, t in events if k == "content")
                    if not thinking:
                        self.assertEqual(content, text)
                    elif THINK_END in text:
                        i = text.index(THINK_END)
                        self.assertEqual((reasoning, content), (text[:i], text[i + len(THINK_END):].lstrip("\n")))
                    else:
                        self.assertEqual((reasoning, content), (text, ""))

    @unittest.skipUnless(DS4_TOKENIZER, "set STRATA_DS4_TOKENIZER to a DS4 pack's tokenizer directory")
    def test_real_token_boundaries(self):
        """The model's own tokens: ｜DSML｜ is one token, <｜DSML｜tool_calls> six (<, ｜DSML｜, tool, _c, alls, >)."""
        import strata_tokenizer as ST
        from serve.server import Detokenizer
        tok = ST.Tokenizer.from_dir(DS4_TOKENIZER)
        self.assertEqual(tok.encode(START, parse_special=True), [30, 128825, 72461, 4941, 12548, 32])
        texts = ["Let me check both.</think>Checking 台北.\n\n" + block(
                     invoke("get_weather", city="台北 \"true\"\n🦊"), invoke("search_files", glob="*.py", pattern="T",
                                                                         max_results=3, flags=["-i", "x"])),
                 "Need it.\n\n" + block(invoke("get_weather", city="Oslo")),
                 "Bad one.</think>" + START + "\n" + INVOKE_OPEN + 'write">\njunk' + INVOKE_CLOSE + "\n" + END]
        for text in texts:
            ids = tok.encode(text, parse_special=True)
            d = Detokenizer(tok)
            pieces = [d.push(i) for i in ids]
            self.assertEqual("".join(pieces), text)
            whole = parse(text)
            self.assertEqual(parse(pieces), whole)
            for i in range(len(pieces) + 1):
                self.assertEqual(parse(["".join(pieces[:i]), "".join(pieces[i:])]), whole)


# ------------------------------------------------------------------------------------------------ service
class DSThinkingEngine(MockEngine):
    """#1053's reply: a sentence of reasoning, then eos, no </think>; a prompt ending the thinking gets the answer."""
    THOUGHT = "Let me think: two plus two is four."
    ANSWER = "The answer is 4."

    def __init__(self, tok):
        super().__init__(tok, "x", max_context=CTX)
        self.prompts = []

    def text_for(self, prompt):
        return self.ANSWER if prompt.endswith(THINK_END) else self.THOUGHT

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        for t in (self.tok.encode(self.text_for(self.tok.decode(ids)), parse_special=True) + [EOS_ID])[:max_new]:
            if cancel.is_set():
                return
            yield t


class DSCallingEngine(DSThinkingEngine):
    """Thinks, then answers - unless its prompt opens a call (a forced tool_choice): then it finishes that call."""
    THOUGHT = "I could look this up. "
    ARGS = f'{PARAM_OPEN}q" string="true">2+2{PARAM_CLOSE}\n{INVOKE_CLOSE}\n{END}'

    def text_for(self, prompt):
        if prompt.endswith(INVOKE_OPEN):
            return 'search">\n' + self.ARGS
        if prompt.endswith(INVOKE_OPEN + 'search">\n'):
            return self.ARGS
        if prompt.endswith(THINK_END):
            return self.ANSWER
        return self.THOUGHT + THINK_END + self.ANSWER


class HttpCase(unittest.TestCase):
    def start(self, engine, template=TPL, tok=TOK):
        self.svc = Service(engine, tok, template)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body, raw=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode()
                return r.status, (text if raw or body.get("stream") else json.loads(text))
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 600, **extra}
        return self.post("/v1/chat/completions", body)

    def reply(self, code, b, stream):
        """-> (finish_reason, reasoning, content, [(name, args)]) of a whole or streamed chat reply."""
        self.assertEqual(code, 200, b)
        if not stream:
            c = b["choices"][0]
            m = c["message"]
            calls = [(x["function"]["name"], json.loads(x["function"]["arguments"])) for x in m.get("tool_calls") or []]
            return c["finish_reason"], m.get("reasoning_content") or "", m.get("content") or "", calls
        chunks = [json.loads(line[6:]) for line in b.splitlines() if line.startswith("data: {")]
        reasoning, content, names, args, finish = "", "", {}, {}, None
        for c in chunks:
            d = c["choices"][0]["delta"]
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                names.setdefault(tc["index"], tc["function"].get("name"))
                args[tc["index"]] = args.get(tc["index"], "") + tc["function"].get("arguments", "")
            finish = c["choices"][0]["finish_reason"] or finish
        return finish, reasoning, content, [(names[i], json.loads(args[i])) for i in sorted(names)]


TWO_CALLS = ("Two lookups.</think>Checking both.\n\n" +
             block(invoke("search_files", glob="*.py", pattern="TODO", max_results=5, flags=["-i"]),
                   invoke("get_weather", city="Paris")))


class ServiceCalls(HttpCase):
    def test_stop_ids_come_from_special_ids(self):
        svc = Service(MockEngine(TOK, "x"), TOK, TPL)
        self.assertEqual(svc.stop_ids, {EOS_ID})
        self.assertEqual(MockEngine(TOK, "hi").scripts[0][-1], EOS_ID)
        # a ChatTemplate over the same vocabulary: never the pieces of <|im_end|> (it would stop at the first '<')
        self.assertEqual(Service(MockEngine(TOK, "x"), TOK, ChatTemplate(QWEN)).stop_ids, {EOS_ID})
        self.assertEqual(Service(MockEngine(ByteTokenizer(), "x"), ByteTokenizer(), ChatTemplate(QWEN)).stop_ids,
                         {257, 258})                             # Qwen: as before
        with self.assertRaises(ValueError):                       # a family's stop token the tokenizer does not name
            Service(MockEngine(ByteTokenizer(), "x"), ByteTokenizer(), TPL)

    def test_two_calls_whole_and_streamed(self):
        self.start(MockEngine(TOK, TWO_CALLS, max_context=CTX))
        want = ("tool_calls", "Two lookups.", "Checking both.",
                [("search_files", {"glob": "*.py", "pattern": "TODO", "max_results": 5, "flags": ["-i"]}),
                 ("get_weather", {"city": "Paris"})])
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self.reply(*self.chat(tools=OPENAI_TOOLS, stream=stream), stream), want)

    @unittest.skipUnless(DS4_TOKENIZER, "set STRATA_DS4_TOKENIZER to a DS4 pack's tokenizer directory")
    def test_two_calls_on_the_real_vocabulary(self):
        """The model's own ids through Service.run: the detokenizer gives ｜DSML｜ (128825) as its text, eos is 1."""
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(DS4_TOKENIZER)
        engine = MockEngine(tok, TWO_CALLS, max_context=CTX)
        self.assertEqual(engine.scripts[0][-1], 1)
        self.assertIn(128825, engine.scripts[0])
        self.start(engine, DeepSeekTemplate(FIXTURE, tok), tok)
        want = ("tool_calls", "Two lookups.", "Checking both.",
                [("search_files", {"glob": "*.py", "pattern": "TODO", "max_results": 5, "flags": ["-i"]}),
                 ("get_weather", {"city": "Paris"})])
        for stream in (False, True):
            self.assertEqual(self.reply(*self.chat(tools=OPENAI_TOOLS, stream=stream), stream), want)

    def test_a_call_inside_the_reasoning_is_delivered_in_one_pass(self):
        engine = MockEngine(TOK, "I need the weather.\n\n" + block(invoke("get_weather", city="Oslo")), max_context=CTX)
        prompts = []
        real = engine.generate
        engine.generate = lambda ids, *a, **k: (prompts.append(ids), real(ids, *a, **k))[1]
        self.start(engine)
        self.svc.reasoning_close_retry = True
        for stream in (False, True):
            prompts.clear()
            got = self.reply(*self.chat(tools=OPENAI_TOOLS, stream=stream), stream)
            self.assertEqual(got, ("tool_calls", "I need the weather.", "", [("get_weather", {"city": "Oslo"})]))
            self.assertEqual(len(prompts), 1)                    # not closed-and-retried by #1053

    def test_reasoning_close_retry(self):
        engine = DSThinkingEngine(TOK)
        self.start(engine)
        _, b = self.chat()
        self.assertEqual(len(engine.prompts), 1)                 # off by default
        self.assertFalse(b["choices"][0]["message"].get("content"))
        self.svc.reasoning_close_retry = True
        engine.prompts.clear()
        finish, reasoning, content, calls = self.reply(*self.chat(), False)
        self.assertEqual(len(engine.prompts), 2)
        first, second = engine.prompts
        self.assertEqual(second, first + TOK.encode(DSThinkingEngine.THOUGHT, parse_special=True) +
                         TOK.encode(THINK_END, parse_special=True))
        self.assertEqual(second[-1], 261)                         # the </think> special id
        self.assertEqual((finish, content), ("stop", DSThinkingEngine.ANSWER))
        self.assertIn("two plus two is four", reasoning)
        engine.prompts.clear()
        engine.THOUGHT = "ok" + THINK_END + "Fine."               # an answered reply is left alone
        self.assertEqual(self.reply(*self.chat(), False)[2], "Fine.")
        self.assertEqual(len(engine.prompts), 1)

    def test_the_thinking_budget_wraps_up_in_this_format(self):
        engine = DSThinkingEngine(TOK)
        engine.THOUGHT = "Thinking more and more. " * 10
        self.start(engine)
        finish, reasoning, content, _ = self.reply(*self.chat(reasoning_budget_tokens=12), False)
        self.assertEqual(len(engine.prompts), 2)
        last = TOK.decode(engine.prompts[1])
        self.assertTrue(last.endswith("time to give my answer." + THINK_END), last[-80:])
        self.assertNotIn("<|im_", last)
        self.assertEqual((finish, content), ("stop", DSThinkingEngine.ANSWER))

    def test_forced_tool_choice(self):
        engine = DSCallingEngine(TOK)
        self.start(engine)
        tools = [{"type": "function", "function": {"name": "search", "description": "search the web", "parameters": {
            "type": "object", "properties": {"q": {"type": "string"}}}}},
                 {"type": "function", "function": {"name": "done", "parameters": {"type": "object", "properties": {}}}}]
        named = {"type": "function", "function": {"name": "search"}}
        for choice in ("required", named):
            opening = START + "\n" + INVOKE_OPEN + ('search">\n' if choice == named else "")
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream, thinking=True):
                    engine.prompts.clear()
                    finish, reasoning, _, calls = self.reply(*self.chat(tools=tools, tool_choice=choice,
                                                                        stream=stream), stream)
                    self.assertEqual((finish, calls), ("tool_calls", [("search", {"q": "2+2"})]))
                    self.assertEqual(reasoning, DSCallingEngine.THOUGHT)
                    first, second = engine.prompts
                    self.assertEqual(second, first + TOK.encode(DSCallingEngine.THOUGHT + THINK_END + "\n\n" + opening,
                                                                parse_special=True))
                with self.subTest(choice=choice, stream=stream, thinking=False):
                    engine.prompts.clear()
                    finish, reasoning, _, calls = self.reply(*self.chat(
                        tools=tools, tool_choice=choice, stream=stream, reasoning_effort="none"), stream)
                    self.assertEqual((finish, reasoning, calls), ("tool_calls", "", [("search", {"q": "2+2"})]))
                    self.assertEqual(len(engine.prompts), 1)
                    self.assertTrue(TOK.decode(engine.prompts[0]).endswith(ASSISTANT + THINK_END + "\n\n" + opening))

    def test_the_budget_wrap_up_opens_the_call(self):
        engine = DSCallingEngine(TOK)
        engine.THOUGHT = "I could look this up. " * 8
        self.start(engine)
        tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}}}}} for n in ("search", "done")]
        finish, reasoning, _, calls = self.reply(*self.chat(tools=tools, tool_choice="required",
                                                            reasoning_budget_tokens=20), False)
        self.assertEqual((finish, calls), ("tool_calls", [("search", {"q": "2+2"})]))
        first, second = engine.prompts
        self.assertEqual(second, first + TOK.encode(engine.THOUGHT[:20] + deepseek.REASONING_WRAP_UP + "\n\n" + START +
                                                    "\n" + INVOKE_OPEN, parse_special=True))
        self.assertEqual(reasoning, engine.THOUGHT[:20] + deepseek.REASONING_WRAP_UP.split(THINK_END)[0])

    def test_anthropic_and_responses(self):
        self.start(MockEngine(TOK, TWO_CALLS, max_context=CTX))
        anth_tools = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                      for t in TOOLS]
        for stream in (False, True):
            code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 600, "stream": stream, "tools": anth_tools,
                                                 "messages": [{"role": "user", "content": "go"}]})
            self.assertEqual(code, 200, b)
            if stream:
                evs = [json.loads(line[6:]) for line in b.splitlines() if line.startswith("data: {")]
                stop = [e for e in evs if e["type"] == "message_delta"][0]["delta"]["stop_reason"]
                uses = [e["content_block"]["name"] for e in evs if e["type"] == "content_block_start"
                        and e["content_block"]["type"] == "tool_use"]
            else:
                stop, uses = b["stop_reason"], [c["name"] for c in b["content"] if c["type"] == "tool_use"]
                self.assertEqual([c["type"] for c in b["content"]], ["thinking", "text", "tool_use", "tool_use"])
                self.assertEqual(b["content"][3]["input"], {"city": "Paris"})
            self.assertEqual((stop, uses), ("tool_use", ["search_files", "get_weather"]))
        code, b = self.post("/v1/responses", {"model": "m", "input": "go", "max_output_tokens": 600,
                                              "tools": [{"type": "function", **t} for t in TOOLS]})
        self.assertEqual(code, 200, b)
        calls = [(o["name"], json.loads(o["arguments"])) for o in b["output"] if o["type"] == "function_call"]
        self.assertEqual(calls, [("search_files", {"glob": "*.py", "pattern": "TODO", "max_results": 5,
                                                   "flags": ["-i"]}), ("get_weather", {"city": "Paris"})])

    def test_keep_alive_while_a_block_is_held(self):
        script = "</think>" + block(invoke("write", path="a", content="z" * 200))
        self.start(MockEngine(TOK, script, max_context=CTX, delay_s=0.002))
        with mock.patch.object(server, "HOLD_PING_S", 0.05):
            code, raw = self.chat(tools=OPENAI_TOOLS, stream=True)
        self.assertEqual(code, 200)
        head = raw.split('"tool_calls"')[0]
        self.assertGreater(head.count(": keep-alive"), 0)
        self.assertEqual(self.reply(code, raw, True)[3], [("write", {"path": "a", "content": "z" * 200})])

    def test_vram_and_count_tokens(self):
        self.start(MockEngine(TOK, "x", max_context=CTX))
        code, b = self.post("/v1/vram", {"reserve_mib": 100})
        self.assertEqual(code, 400, b)                            # nothing is sent to the engine
        code, b = self.post("/v1/messages/count_tokens",
                            {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((code, b["input_tokens"]), (200, len(TOK.encode(
            BOS + USER + "hi" + ASSISTANT + THINK_START, parse_special=True))))

    def test_props(self):
        self.start(MockEngine(TOK, "x", max_context=CTX))
        with urllib.request.urlopen(self.base + "/props", timeout=30) as r:
            props = json.loads(r.read())
        self.assertEqual((props["bos_token"], props["eos_token"]), (BOS, EOS))
        self.assertTrue(props["chat_template_caps"]["supports_tool_calls"])
        self.assertIn("dsml_token", props["chat_template"])


# ------------------------------------------------------------------------------------------------ literal tags
def legacy_encode(tok, template, tags, messages, tools, kwargs):
    """v0.1.41's encode_prompt (fb58e0db frontend.py 222-290): every literal, fixed marks, an alternation unmark."""
    def mark(text):
        for tag, m in tags.items():
            text = text.replace(tag, m)
        return text

    def deep(v):
        return mark(v) if isinstance(v, str) else {k: deep(x) for k, x in v.items()} if isinstance(v, dict) else \
            [deep(x) for x in v] if isinstance(v, list) else v

    def has(v):
        return any(t in v for t in tags) if isinstance(v, str) else any(has(x) for x in v.values()) \
            if isinstance(v, dict) else any(has(x) for x in v) if isinstance(v, list) else False
    if not has(messages) and not has(tools):
        return tok.encode(template.render(messages, tools=tools, **kwargs), parse_special=True)
    out = []
    for m in messages:
        m, content = dict(m), m.get("content")
        for k, v in m.items():
            if k != "role":
                m[k] = deep(v)
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = content[:i] + mark(content[i:j]) + "</think>" + mark(content[j + len("</think>"):])
        out.append(m)
    prompt = template.render(out, tools=deep(tools), **kwargs)
    marks = {v: k for k, v in tags.items()}
    rx = re.compile("|".join(map(re.escape, marks)))
    parts, spans, pos, n = [], [], 0, 0
    for m in rx.finditer(prompt):
        parts.append(prompt[pos:m.start()])
        n += m.start() - pos
        parts.append(marks[m.group(0)])
        spans.append((n, n + len(marks[m.group(0)])))
        n += len(marks[m.group(0)])
        pos = m.end()
    parts.append(prompt[pos:])
    return tok.encode("".join(parts), parse_special=True, plain=spans)


class ThinkTokenizer(ByteTokenizer):
    ALWAYS = ("<think>", "</think>")
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]


class LiteralTags(unittest.TestCase):
    """#537/#554 with M4's fixes: the same ids as v0.1.41 for any request without private-use text, fast on DS4's
    1,283 control literals, and a private-use character of the request never becomes a literal's text."""

    def random_requests(self, literals, n, seed=7):
        rng = random.Random(seed)
        words = ["plain", " ", "\n", "x<y", "<", ">", "|", "think", "é"] + list(literals)
        say = lambda: "".join(rng.choice(words) for _ in range(rng.randint(0, 9)))  # noqa: E731
        for _ in range(n):
            msgs = [{"role": "system", "content": say()}, {"role": "user", "content": say()}]
            if rng.random() < 0.5:
                msgs.append({"role": "assistant", "content": rng.choice(["", "<think>" + say() + "</think>"]) + say(),
                             "reasoning_content": say(),
                             "tool_calls": [{"function": {"name": "write", "arguments": {"text": say()}}}]})
                msgs.append({"role": "tool", "content": say()})
                msgs.append({"role": "user", "content": say()})
            tools = [{"name": "write", "description": say(), "parameters": {}}] if rng.random() < 0.5 else None
            yield msgs, tools

    def check_identical(self, tok, template, n=150):
        svc = Service(MockEngine(tok, "x", max_context=1 << 20), tok, template)
        for msgs, tools in self.random_requests(tok.control_tokens + ["<think>", "</think>", "<|vision_start|>"], n):
            for kw in ({}, {"enable_thinking": False}):
                self.assertEqual(svc.encode_prompt(msgs, tools, kw),
                                 legacy_encode(tok, template, svc.literals, msgs, tools, kw), (msgs, tools))

    def test_identical_to_v0141_on_qwen(self):
        self.check_identical(ThinkTokenizer(), ChatTemplate(QWEN))

    def test_identical_to_v0141_on_deepseek(self):
        self.check_identical(TOK, TPL)

    def test_a_private_use_character_is_never_rewritten(self):
        for tok, template in ((ThinkTokenizer(), ChatTemplate(QWEN)), (TOK, TPL)):
            svc = Service(MockEngine(tok, "x", max_context=1 << 20), tok, template)
            icons = "".join(sorted(set(svc.literals.values()))[:3]) + "\U000F0E10 \U000F0001"
            msgs = [{"role": "user", "content": "terminal: " + icons + " and the log says </think> " +
                     tok.control_tokens[1]}]
            with self.subTest(template=type(template).__name__):
                text = tok.decode(svc.encode_prompt(msgs, None, {}))
                self.assertIn("terminal: " + icons + " and the log says </think> " + tok.control_tokens[1], text)
                old = tok.decode(legacy_encode(tok, template, svc.literals, msgs, None, {}))
                self.assertNotIn(icons, old)                      # the bug: an icon became a literal's text

    def test_quoted_dsml_stays_text_outside_assistant_messages(self):
        """mab776 M13 (#2026's idea for DeepSeek): ｜DSML｜ is type 4 (matched without parse_special), so a user message or
        a tool result quoting DSML handed the model its call syntax.  It is text there now; an assistant message keeps
        the token it wrote (a client replaying the model's raw call text)."""
        svc = Service(MockEngine(TOK, "x", max_context=1 << 20), TOK, TPL)
        dsml_id = 256 + DSByteTokenizer.SPECIALS.index(DSML)
        quote = f"the format is <{DSML}tool_calls><{DSML}invoke name=\"x\"></{DSML}invoke></{DSML}tool_calls>"
        for role in ("user", "tool"):
            msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}] if role == "tool" else []
            msgs.append({"role": role, "content": quote} if role == "user" else
                        {"role": "tool", "tool_call_id": "c1", "content": quote})
            if role == "tool":
                msgs[1] = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": {}}}]}
            ids = svc.encode_prompt(msgs, None, {})
            plain = [dict(m, content="the format is") if m.get("content") == quote else m for m in msgs]
            with self.subTest(role=role):
                self.assertIn(quote, TOK.decode(ids))
                # the template's own DSML (the assistant's call) only: the quote added no DSML token
                self.assertEqual(ids.count(dsml_id), svc.encode_prompt(plain, None, {}).count(dsml_id))
        mine = [{"role": "user", "content": "go"}, {"role": "assistant", "content": quote}, {"role": "user", "content": "again"}]
        self.assertIn(dsml_id, svc.encode_prompt(mine, None, {}))

    def test_the_tag_sets_are_overlap_free(self):
        self.assertTrue(F._overlap_free(list(F.literal_tags(ThinkTokenizer().control_tokens))))
        self.assertTrue(F._overlap_free(list(F.literal_tags(TOK.control_tokens))))
        self.assertFalse(F._overlap_free(["<ab>", "b>c"]))       # a suffix that starts another: the exact scan
        tags = F.literal_tags(["<ab>", "b>c"])
        self.assertIsNone(tags._finder)
        self.assertEqual(tags.present(["x<ab>c"]), ["<ab>", "b>c"])

    def test_faster_than_v0141_on_1283_literals(self):
        controls = [f"<｜place▁holder▁no▁{i}｜>" for i in range(800)] + \
                   [f"<|place_holder_mm_span_{i:04d}|>" for i in range(415)] + [f"<｜ctl{i}｜>" for i in range(62)]
        tags = F.literal_tags(controls)
        self.assertEqual(len(tags), 1286)                         # DS4: 1,277 control tokens + the 9 literal tags (M13: </tool_call>, <｜DSML｜, </｜DSML｜)
        text = (Path(F.__file__).read_text(encoding="utf-8") * 6)[:400_000]
        msgs = [{"role": "user", "content": text[:200_000] + " <｜place▁holder▁no▁7｜> and </think> " + text[200_000:]}]

        class Echo:                                               # a template that writes the message as it is
            def render(self, messages, tools=None, **kw):
                return "<s>" + messages[0]["content"]

        class Plain:
            def encode(self, text, parse_special=False, plain=()):
                return [text, tuple(plain)]

        def new():
            marked, _, marks = F.mark_think_literals(msgs, None, tags)
            prompt, plain = F.unmark_think_literals(Echo().render(marked), marks)
            return Plain().encode(prompt, parse_special=True, plain=plain)

        def best(fn):
            t, out = 1e9, None
            for _ in range(3):
                t0 = time.perf_counter()
                out = fn()
                t = min(t, time.perf_counter() - t0)
            return out, t
        got, t_new = best(new)
        want, t_old = best(lambda: legacy_encode(Plain(), Echo(), tags, msgs, None, {}))
        self.assertEqual(got, want)
        self.assertLess(t_new * 3, t_old, f"new {t_new * 1000:.0f} ms vs v0.1.41 {t_old * 1000:.0f} ms")


# ------------------------------------------------------------------------------------------------ selection, config
def tiny_tokenizer(d: Path, template: Path = FIXTURE) -> Path:
    """A from_dir tokenizer directory with DeepSeek-V4's specials: 256 byte tokens, one merge, the template."""
    import strata_tokenizer as ST
    b2u = ST.bytes_to_unicode()
    tokens = [b2u[b] for b in range(256)] + ["ĠĠ", BOS, EOS, USER, ASSISTANT, THINK_START, THINK_END, DSML]
    types = [1] * 257 + [3, 3, 3, 3, 4, 4, 4]
    t = d / "tokenizer"
    t.mkdir()
    (t / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(tokens)}, ensure_ascii=False),
                                  encoding="utf-8")
    (t / "merges.txt").write_text("Ġ Ġ", encoding="utf-8")
    (t / "token_type.json").write_text(json.dumps(types))
    (t / "tokenizer.json").write_text(json.dumps({"pre": "joyai-llm", "special_ids": {
        "tokenizer.ggml.bos_token_id": 257, "tokenizer.ggml.eos_token_id": 258}}))
    (t / "chat_template.jinja").write_bytes(template.read_bytes())
    return t


class Selection(unittest.TestCase):
    def test_detect(self):
        self.assertTrue(deepseek.detect(FIXTURE.read_text(encoding="utf-8")))
        self.assertFalse(deepseek.detect(QWEN.read_text(encoding="utf-8")))
        with self.assertRaises(ValueError):                      # V3.2's DSML is another format
            deepseek.detect("{%- set dsml_token = '｜DSML｜' -%}<{{ dsml_token }}function_calls>")

    def test_load_template(self):
        self.assertIsInstance(server.load_template(FIXTURE, TOK, {}), DeepSeekTemplate)
        self.assertIsInstance(server.load_template(FIXTURE, TOK, {"architecture": "deepseek4"}), DeepSeekTemplate)
        qwen = server.load_template(QWEN, ByteTokenizer(), {})
        self.assertIs(type(qwen), ChatTemplate)
        for args in ((FIXTURE, ByteTokenizer(), {}),                       # the tokenizer does not fit the template
                     (QWEN, ByteTokenizer(), {"architecture": "deepseek4"}),
                     (FIXTURE, TOK, {"architecture": "qwen"}),
                     (FIXTURE, TOK, {"vision": {"exe": "v"}}),
                     (FIXTURE, TOK, {"parallel": 2}),
                     (FIXTURE, TOK, {"slot_save_path": "slots"})):
            with self.subTest(args=args[2]), self.assertRaises(SystemExit):
                server.load_template(*args)
        with self.assertRaises(SystemExit):
            server.load_template(FIXTURE, TOK, {}, slot_save_path="slots")

    def test_config_problems(self):
        errors, notes = deepseek.config_problems({"vision": {"exe": "v"}, "parallel": 4, "vram_elastic": True,
                                                  "expert_profile_save": "p", "slot_save_path": "s",
                                                  "effort_position": "end", "tool_call_recovery": True,
                                                  "reasoning_loop_recovery": "recover"})
        self.assertEqual(len(errors), 5)
        self.assertEqual(len(notes), 3)
        self.assertEqual(deepseek.config_problems({"parallel": 1, "effort_position": "start",
                                                   "reasoning_loop_recovery": "stop"}), ([], []))
        example = json.loads((ROOT / "tools/ds4/ds4-server.json.example").read_text(encoding="utf-8"))
        self.assertEqual(deepseek.config_problems(example), ([], []))
        self.assertIn("ds4-lcpp", example["allowed_hosts"])
        self.assertTrue(example["reasoning_close_retry"])

    def test_args_pass_through_and_slots_are_not_batch_slots(self):
        args = ["-m", "/m/x.gguf", "--slots", "auto", "--ctx", "8192"]
        cfg = {"args": args, "gpu": [0, 1]}
        self.assertEqual(server.engine_args(cfg, True), args)
        self.assertIn("--layer-split", server.engine_args(cfg))   # the Strata engine: as before
        fake = ('import sys\nprint("READY 8192 stop", flush=True)\n'
                'for line in sys.stdin:\n    if line.startswith("QUIT"):\n        break\n')
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "fake.py"
            script.write_text(fake, encoding="utf-8")
            real, started = server.subprocess.Popen, []

            def popen(cmd, **kw):
                started.append(real([sys.executable, str(script), *cmd[1:]], **kw))
                return started[-1]
            with mock.patch.object(server.subprocess, "Popen", popen):
                eng = DeepSeekEngine("ds4_serve", args)
                try:
                    self.assertEqual((eng.max_context, eng.batch), (8192, 0))
                finally:
                    eng.close()
                try:
                    with self.assertRaises(ValueError):           # read as batch slots: int("auto")
                        StrataEngine("strata", args)
                finally:
                    for p in started[1:]:
                        p.kill()
                        p.wait(10)
                        for f in (p.stdin, p.stdout):
                            f.close()


FAKE_DS4 = r'''import json, sys
# a DeepSeek engine speaking Strata's line protocol: replays --script's id lists (the last repeats) for each GEN
a = sys.argv
script = json.load(open(a[a.index("--script") + 1]))
log = open(a[a.index("--gens") + 1], "a")
print("INFO engine=fake-ds4 context=4096", flush=True)
print("READY 4096 stop", flush=True)
k = 0
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if not line.startswith("GEN "):
        continue                                   # a stale STOP
    log.write(line)
    log.flush()
    ids = script[min(k, len(script) - 1)]
    k += 1
    for t in ids:
        print(f"T {t}", flush=True)
    n = len(line.split()[-1].split(","))
    print(f"DONE {len(ids)} {n} 1.0 1.0 stop 0 0 0", flush=True)
'''


class MainEndToEnd(unittest.TestCase):
    """serve/server.py main() from a config, as the container runs it: the template is detected, the fake engine
    started with its args as they are, stop ids {eos}, DSML calls back."""

    def test_main_with_a_deepseek_config(self):
        import strata_tokenizer as ST
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tdir = tiny_tokenizer(d)
            tok = ST.Tokenizer.from_dir(tdir)
            reply = "Need it.</think>On it.\n\n" + block(invoke("get_weather", city="Oslo"))
            (d / "script.json").write_text(json.dumps([tok.encode(reply, parse_special=True) + [258]]))
            (d / "fake.py").write_text(FAKE_DS4, encoding="utf-8")
            exe = d / "ds4_serve"
            exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{d / "fake.py"}" "$@"\n', encoding="utf-8")
            exe.chmod(0o755)
            cfg = {"architecture": "deepseek4", "exe": str(exe), "tokenizer": str(tdir), "log": str(d / "engine.log"),
                   "args": ["--slots", "auto", "--script", str(d / "script.json"), "--gens", str(d / "gens.log")],
                   "gpu": [0, 1], "reasoning_close_retry": True, "effort_position": "end"}
            (d / "cfg.json").write_text(json.dumps(cfg))
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config",
                                     str(d / "cfg.json"), "--port", str(port)], cwd=str(ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = []
            reader = threading.Thread(target=lambda: out.extend(proc.stdout), daemon=True)
            reader.start()
            base = f"http://127.0.0.1:{port}"
            try:
                deadline = time.time() + 60
                while True:
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2):
                            break
                    except OSError:
                        if time.time() > deadline or proc.poll() is not None:
                            self.fail("the server did not start:\n" + "".join(out))
                        time.sleep(0.2)
                body = {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "weather?"}],
                        "tools": [WEATHER]}
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    choice = json.loads(r.read())["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                self.assertEqual((choice["message"]["reasoning_content"], choice["message"]["content"]),
                                 ("Need it.", "On it."))
                self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
                                 {"city": "Oslo"})
                with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
                    self.assertEqual(json.loads(r.read())["data"][0]["id"], "deepseek-v4-flash")
                gen = (d / "gens.log").read_text().splitlines()[0]
                prompt = tok.decode([int(x) for x in gen.split()[-1].split(",")])
                self.assertTrue(prompt.startswith(BOS + "## Tools"), prompt[:80])
                self.assertTrue(prompt.endswith(USER + "weather?" + ASSISTANT + THINK_START), prompt[-80:])
                self.assertNotIn("<|im_", prompt)                 # no ChatML effort turn ("end" was ignored)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(10)
                reader.join(10)
                proc.stdout.close()
            log = "".join(out)
            self.assertIn("DeepSeek-V4 chat format", log)
            self.assertIn("stop ids [258]", log)
            self.assertIn('"effort_position": "end" ignored', log)
            self.assertIn("--serve --slots auto --script", (d / "engine.log").read_text())   # args as they are


if __name__ == "__main__":
    unittest.main()
