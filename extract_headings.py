# extract_headings.py
import re

# Path to your README.md
readme_file = "README.md"

# Regex to match headings ##, ###, ####
heading_pattern = re.compile(r'^(#{2,4})\s+(.*)')

headings = []

with open(readme_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = heading_pattern.match(line.strip())
        if match:
            level = len(match.group(1))  # 2, 3, or 4
            title = match.group(2).strip()
            if level in [2, 3]:  # Only keep ## and ###
                headings.append((level, title))

# Print headings in order
for level, title in headings:
    print(f"{'#'*level} {title}")