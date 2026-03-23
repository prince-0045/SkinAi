import glob
import re

files = glob.glob('src/pages/*.jsx') + ['src/components/auth/AuthLayout.jsx']

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    def replacer(match):
        cls_str = match.group(1)
        # Remove old backgrounds
        cls_str = re.sub(r'\bbg-(white|gray-50|slate-50|medical-50|slate-950)\b', '', cls_str)
        cls_str = re.sub(r'\bdark:bg-slate-[0-9]{3}\b', '', cls_str)
        # Clean up spaces
        cls_str = re.sub(r'\s+', ' ', cls_str).strip()
        
        # Add new backgrounds if not already there
        if 'bg-[#0d1117]' not in cls_str:
            cls_str = f'bg-[#0d1117] bg-navy-mesh {cls_str}'
            
        return f'className="{cls_str}"'

    new_content = content
    
    # Replace lines with min-h-screen
    new_content = re.sub(r'className="([^"]*?min-h-screen[^"]*?)"', replacer, new_content)

    # For AuthLayout, also replace the split panels
    if 'AuthLayout.jsx' in file:
        new_content = re.sub(r'className="([^"]*?lg:w-1/2[^"]*?)"', replacer, new_content)
        new_content = re.sub(r'className="([^"]*?hidden lg:flex w-1/2[^"]*?)"', replacer, new_content)

    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {file}')
