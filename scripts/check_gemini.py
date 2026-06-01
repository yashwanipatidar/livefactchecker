import sys
import os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from app import verifier

try:
    out = verifier.call_gemini('Return JSON {"verdict":"TRUE","confidence":100,"explanation":"ok"}')
    print('OK')
    print(out[:200])
except Exception as e:
    print(type(e).__name__, str(e))
