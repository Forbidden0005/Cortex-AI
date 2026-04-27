from core.task_classifier import classify_task_type

tests = [
    ("Hi Cortex, how are you?", "general"),
    ("read the file config.txt", "file"),
    ("write a python script", "coding"),
    ("search the web for news", "web"),
    ("plan my project steps", "planning"),
    ("what time is it?", "general"),
    ("remember my name is Tyler", "memory"),
]

all_ok = True
for text, expected in tests:
    got = classify_task_type(text)
    ok = got == expected
    print(f"  [{'OK' if ok else 'FAIL'}] {text!r} -> {got} (expected {expected})")
    all_ok = all_ok and ok

print()
print("[PASS] Classifier correct!" if all_ok else "[FAIL] Some mismatches")
