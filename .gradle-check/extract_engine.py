from pathlib import Path

source = Path("النموذج_نسخة_محدثة (4).kt").read_text(encoding="utf-8")
start = source.index("class NMem {")
end = source.index("class SimWorld(", start)
engine = source[start:end]
engine = engine.replace(
    "android.util.Base64.decode(EMBEDDED_WEIGHTS_B64, android.util.Base64.DEFAULT)",
    "java.util.Base64.getDecoder().decode(EMBEDDED_WEIGHTS_B64)"
)
if "android." in engine:
    raise SystemExit("Unexpected Android dependency remains in extracted model/session source")
for marker in ("class NMem {", "class Nlm private constructor", "fun gradCheck(", "class NlmSession("):
    if marker not in engine:
        raise SystemExit(f"Required NLM2 source marker missing: {marker}")

header = """package ai.nuwaa

import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder

private fun partialSane(s: String): Boolean {
    var round = 0
    var square = 0
    var curly = 0
    for (c in s) when (c) {
        '(' -> round++
        ')' -> { round--; if (round < 0) return false }
        '[' -> square++
        ']' -> { square--; if (square < 0) return false }
        '{' -> curly++
        '}' -> { curly--; if (curly < 0) return false }
    }
    return true
}

"""
out = Path(".gradle-check/src/main/kotlin/ai/nuwaa/Engine.kt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(header + engine, encoding="utf-8")
print(f"Extracted exact NMem + Nlm + NlmSession source ({len(engine)} chars).")
