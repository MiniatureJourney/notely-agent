import re
import json

test_str = r'{"content": "Here is an equation: \sigma and \u002 and \uABCD and \n and \"."}'

# The regex
fixed_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', test_str)

print("Original:", test_str)
print("Fixed:", fixed_str)
try:
    print(json.loads(fixed_str))
except Exception as e:
    print("Error:", e)
