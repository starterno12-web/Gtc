from pathlib import Path

source = Path("النموذج_نسخة_محدثة (3).kt").read_text(encoding="utf-8")
start = source.index("class NMem {")
end = source.index("class NlmSession(", start)
engine = source[start:end]
engine = engine.replace(
    "android.util.Base64.decode(EMBEDDED_WEIGHTS_B64, android.util.Base64.DEFAULT)",
    "java.util.Base64.getDecoder().decode(EMBEDDED_WEIGHTS_B64)"
)
if "android." in engine:
    raise SystemExit("Unexpected Android dependency remains in extracted NLM2 engine")
if "class Nlm private constructor" not in engine or "fun gradCheck(" not in engine:
    raise SystemExit("NLM2 extraction markers not found; refusing to test a partial engine")

header = """package ai.nuwaa

import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder

"""
out = Path(".gradle-check/src/main/kotlin/ai/nuwaa/Engine.kt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(header + engine, encoding="utf-8")
print(f"Extracted exact NMem + Nlm source ({len(engine)} chars) from Android source.")
