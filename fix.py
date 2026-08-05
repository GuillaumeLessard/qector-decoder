with open('python/qector_decoder_v3/__init__.py', 'r', encoding='utf-8') as f:
    c = f.read()
import re
c = re.sub(r'@property\s+def precision\(self\):\s+.*?\s+return self\._inner\.precision', '', c, flags=re.DOTALL)
with open('python/qector_decoder_v3/__init__.py', 'w', encoding='utf-8') as f:
    f.write(c)
