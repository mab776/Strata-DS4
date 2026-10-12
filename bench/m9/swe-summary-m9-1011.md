# SWE-bench Verified-mini, DeepSeek-V4-Flash IQ2_M - m9-1011: Strata-DS4 (stack 74, all 50) vs llama.cpp (stack 73, first 10)

Issues run under every condition: 10

| condition | resolved | mean steps (solved / failed) | median steps | output tokens (median / total) | total minutes | runs > 60 steps | hit 250-step / 45-min cap |
|---|---|---|---|---|---|---|---|
| lcpp-t1.0 | 8/10 | 44.3 (32.1 / 93.0) | 32 | 8,724 / 139,515 | 77 | 2 | 0 |
| strata-t1.0 | 8/10 | 58.1 (38.5 / 136.5) | 48 | 10,670 / 210,583 | 60 | 4 | 0 |

Per issue: resolved, steps, output tokens, minutes

| issue | lcpp-t1.0 | strata-t1.0 |
|---|---|---|
| django__django-11815 | ✅ 27 st, 6,885 tok, 4 min | ✅ 42 st, 9,307 tok, 3 min |
| django__django-11880 | ✅ 14 st, 1,907 tok, 1 min | ✅ 11 st, 1,557 tok, 0 min |
| django__django-12276 | ✅ 21 st, 2,310 tok, 1 min | ✅ 27 st, 4,116 tok, 1 min |
| django__django-12713 | ✅ 28 st, 5,190 tok, 3 min | ✅ 13 st, 2,455 tok, 1 min |
| sphinx-doc__sphinx-7757 | ✅ 29 st, 10,562 tok, 5 min | ✅ 19 st, 6,670 tok, 2 min |
| sphinx-doc__sphinx-8035 | ✅ 53 st, 11,071 tok, 6 min | ✅ 54 st, 12,033 tok, 4 min |
| sphinx-doc__sphinx-8056 | ✅ 51 st, 18,630 tok, 9 min | ✅ 77 st, 34,919 tok, 9 min |
| sphinx-doc__sphinx-8638 | ❌ 71 st, 31,285 tok, 16 min | ❌ 123 st, 53,218 tok, 15 min |
| sphinx-doc__sphinx-9229 | ❌ 115 st, 45,473 tok, 29 min | ❌ 150 st, 64,378 tok, 18 min |
| sphinx-doc__sphinx-9230 | ✅ 34 st, 6,202 tok, 3 min | ✅ 65 st, 21,930 tok, 6 min |

Solved at lcpp-t1.0 but missed at another temperature: 0 []

Solved at strata-t1.0 but missed at another temperature: 0 []
