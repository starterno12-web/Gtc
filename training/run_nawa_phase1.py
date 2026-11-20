#!/usr/bin/env python3
from pathlib import Path
src = Path('النموذج_نسخة_محدثة__1_ (1).kt').read_text(encoding='utf-8')
start = src.index('class Nlm private')
end = src.index('/** One conversation.', start)
model = src[start:end]
header = '''package ai.nuwaa
import java.io.File
import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
class NMem { fun add(text: String) {} }
'''
tail = '''
fun main() {
 val story = File('../training/nawa_phase1_story.txt').readText(Charsets.UTF_8).trim()
 require(story.length > 500)
 val f = File('../training/model/nlm2_user.bin')
 val m = if (f.exists()) Nlm.load(f.readBytes()).let { old -> if (old.chars.containsAll(story.map { it.toString() }.distinct())) old else Nlm.fresh(story,32,64) } else Nlm.fresh(story,32,64)
 val ids = m.encode(story)
 val before = m.bpc(ids)
 val chunks = story.chunked(120).filter { it.length >= 8 }
 for (pass in 1..50) {
   for (c in chunks) m.trainWindow(c,0.00012f)
   println('Pass ' + pass + '/50 complete')
 }
 val after = m.bpc(ids)
 require(after.isFinite() && after < before) { 'Loss gate failed; weights not saved.' }
 File('../training/model').mkdirs()
 val tmp = File('../training/model/nlm2_user.bin.tmp'); tmp.writeBytes(m.toBytes())
 if (!tmp.renameTo(f)) { f.writeBytes(tmp.readBytes()); tmp.delete() }
 val report = 'Phase 1 story training\nPasses: 50\nBefore bits/char: ' + before + '\nAfter bits/char: ' + after + '\nWeights: ' + m.paramCount() + '\nSaved: true\n'
 File('../training/model/results_phase1.txt').writeText(report,Charsets.UTF_8)
 println(report)
}
'''
Path('runner/src/main/kotlin/ai/nuwaa/ModelRunner.kt').parent.mkdir(parents=True,exist_ok=True)
Path('runner/src/main/kotlin/ai/nuwaa/ModelRunner.kt').write_text(header + model + tail,encoding='utf-8')