"""Delete get_model_config_for_expert_location from a model file, in place.

Reproduces upstream's real state for a MoE model class that never declared the
hook (HYV4 is such a class), without needing that model's weights.
"""

import sys

path = sys.argv[1]
lines = open(path).read().split("\n")

start = next(
    i
    for i, l in enumerate(lines)
    if l.strip().startswith("def get_model_config_for_expert_location")
)
assert lines[start - 1].strip() == "@classmethod", lines[start - 1]
end = start + 1
while end < len(lines) and (lines[end].strip() == "" or lines[end].startswith(" " * 8)):
    end += 1
# Keep one blank separator, drop the decorator line too.
removed = lines[start - 1 : end]
del lines[start - 1 : end]
open(path, "w").write("\n".join(lines))
print(f"{path}: removed {len(removed)} lines")
