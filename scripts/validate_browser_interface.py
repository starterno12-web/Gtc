# -*- coding: utf-8 -*-
"""Validate the final browser-interface answer; reject prose-only and pseudo-code."""
import re
from pathlib import Path

source = Path("hard_test_answer.txt").read_text(encoding="utf-8")
marker = "MODEL OUTPUT (RAW):"
answer = source.split(marker, 1)[1] if marker in source else source
# Permit a model to wrap its complete HTML in a Markdown fence, but validate the code itself.
answer = re.sub(r"```(?:html)?", "", answer, flags=re.IGNORECASE)
checks = {
    "html_document": bool(re.search(r"<!doctype\\s+html|<html\\b", answer, re.I)),
    "css": bool(re.search(r"<style\\b|style\\s*=", answer, re.I)),
    "javascript": bool(re.search(r"<script\\b", answer, re.I)),
    "visible_status_word": "شغال" in answer,
    "ten_distinct_hex_colors": len(set(re.findall(r"#[0-9a-fA-F]{6}\\b", answer))) >= 10,
    "real_dom_event_binding": "addEventListener" in answer,
    "dom_update_or_clipboard": bool(re.search(r"textContent|style\\.|navigator\\.clipboard", answer)),
}
passed = all(checks.values())
if passed:
    report = "نجاح: اجتاز الناتج فحص الواجهة الأساسي.\n" + "\n".join(f"{k}: {v}" for k, v in checks.items())
else:
    report = "هذا غير واجهة. اكتب صفحة فعلية باستخدام HTML وCSS وJavaScript، ثم أعد الاختبار.\n" + "\n".join(f"{k}: {v}" for k, v in checks.items())
Path("hard_test_validation.txt").write_text(report + "\n", encoding="utf-8")
print(report)
