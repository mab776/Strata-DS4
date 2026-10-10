# mab776: encode/decode token ids with the GGUF tokenizer.  python3 mab776-tok.py enc|dec <gguf> "<text>"|<ids,comma>
# needs `regex` -> run in python:3.12-slim with this repo + the gguf mounted (pip install regex). BOS = id 0, not parsed.
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "tools"))
from strata_tokenizer import Tokenizer
t = Tokenizer.from_gguf(sys.argv[2]) if hasattr(Tokenizer, "from_gguf") else Tokenizer(sys.argv[2])
if sys.argv[1] == "enc":
    print(",".join(str(i) for i in t.encode(sys.argv[3])))
else:
    print(t.decode([int(x) for x in sys.argv[3].split(",")]))
