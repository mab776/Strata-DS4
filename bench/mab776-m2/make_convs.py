#!/usr/bin/env python3
"""mab776 M2/M4: the 10 chat requests whose stack-73 (llama.cpp b11429) /apply-template renders M4 must reproduce
byte for byte.  Writes bench/mab776-m2/conv/<name>.json (request bodies for /apply-template)."""
import json, pathlib

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "search_files", "description": "Search files by glob and regex",
          "parameters": {"type": "object", "properties": {"glob": {"type": "string"}, "pattern": {"type": "string"},
                         "max_results": {"type": "integer"}, "flags": {"type": "array", "items": {"type": "string"}}},
                         "required": ["glob", "pattern"]}}}]
U = [{"role": "user", "content": "What is 17*23?"}]
C = {
 "c01-plain": {"messages": U},
 "c02-system": {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three primes."}]},
 "c03-multiturn-reasoning": {"messages": [{"role": "user", "content": "Pick a number."},
     {"role": "assistant", "content": "7.", "reasoning_content": "Any small number works."},
     {"role": "user", "content": "Double it."}]},
 "c04-thinking-off": {"reasoning_effort": "none", "messages": U},
 "c05-effort-high": {"reasoning_effort": "high", "messages": U},
 "c06-effort-max": {"reasoning_effort": "max", "messages": U},
 "c06b-effort-xhigh": {"reasoning_effort": "xhigh", "messages": U},
 "c06c-effort-medium": {"reasoning_effort": "medium", "messages": U},
 "c07-tools": {"tools": TOOLS, "messages": [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Montréal?"}]},
 "c08-tool-roundtrip": {"tools": TOOLS, "messages": [{"role": "user", "content": "Weather in Montréal?"},
     {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function":
         {"name": "get_weather", "arguments": "{\"city\": \"Montréal\", \"unit\": \"c\"}"}}]},
     {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": -3, \"sky\": \"snow\"}"}]},
 "c09-parallel-calls-history": {"tools": TOOLS, "messages": [{"role": "user", "content": "TODOs in *.py and Paris weather?"},
     {"role": "assistant", "content": "", "reasoning_content": "Two independent calls.", "tool_calls": [
         {"id": "a", "type": "function", "function": {"name": "search_files", "arguments": "{\"glob\": \"*.py\", \"pattern\": \"TODO\", \"max_results\": 5, \"flags\": [\"-i\"]}"}},
         {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
     {"role": "tool", "tool_call_id": "b", "content": "12C cloudy"},
     {"role": "tool", "tool_call_id": "a", "content": "main.py:3: # TODO"},
     {"role": "assistant", "content": "One TODO in main.py; Paris 12C, cloudy.", "reasoning_content": "Summarize."},
     {"role": "user", "content": "Thanks. Now fix it."}]},
 "c10-history-developer-parts-prefill": {"messages": [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"},
     {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
     {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
     {"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Q2"},
     {"role": "assistant", "content": "Partial answer: "}]},
 "c11-response-format": {"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema":
     {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}, "messages": U},
 "c12-multiturn-noreasoning-tools": {"tools": TOOLS, "messages": [{"role": "system", "content": "Be useful."},
     {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?", "reasoning_content": "Greet."},
     {"role": "user", "content": "Weather in Oslo, in f?"}]},
}
out = pathlib.Path(__file__).resolve().parent / "conv"
out.mkdir(exist_ok=True)
for k, v in C.items():
    (out / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(C), "conversations ->", out)
