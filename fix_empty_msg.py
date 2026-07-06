import os
import shutil

target_file = r"C:\Users\Admin\Desktop\qector-decoder-v3-0.5.0-frozen\python\qector_decoder_v3\__init__.py"

with open(target_file, "r", encoding="utf-8", newline="") as f:
    content = f.read()

# Replace the specific empty check message
content = content.replace('raise ValueError(f"Check {i} is empty")', 'raise ValueError("All checks must be non-empty")')

with open(target_file, "w", encoding="utf-8", newline="") as f:
    f.write(content)

# Copy to site-packages
shutil.copy(target_file, r".venv\Lib\site-packages\qector_decoder_v3\__init__.py")

print("Fixed empty check message and copied successfully!")
