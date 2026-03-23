import re
import os

path = r"d:\SkinAi\backend\app\services\ml_model.py"
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove 'Updated upstream' to '======='
text = re.sub(r'<<<<<<< Updated upstream.*?=======\r?\n', '', text, flags=re.DOTALL)
# Remove 'Stashed changes'
text = re.sub(r'>>>>>>> Stashed changes\r?\n?', '', text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Git merge conflict markers successfully removed from ml_model.py!")
