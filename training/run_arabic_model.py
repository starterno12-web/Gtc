#!/usr/bin/env python3
from pathlib import Path
src = Path("النموذج_نسخة_محدثة__1_ (1).kt").read_text(encoding="utf-8")
start = src.index("class Nlm private")
end = src.index("/** One conversation.", start)
model = src[start:end]
out = Path("runner/src/main/kotlin/ai/nuwaa/ModelRunner.kt")
out.parent.mkdir(parents=True, exist_ok=True)
header = """package ai.nuwaa
import java.io.File
import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
class NMem { fun add(text: String) {} }
"""
qs = ["ما الفرق بين الجملة الاسمية والجملة الفعلية؟","أعرب الجملة: قرأ الطالب الكتاب.","ما الفرق بين الفاعل ونائب الفاعل؟","متى ترفع الأسماء الخمسة بالواو؟","ما الفرق بين إن وأن؟","استخرج الجذر من كلمة مكتبة.","ما الفرق بين اسم الفاعل واسم المفعول؟","اشتق اسم الفاعل من الفعل استخرج.","ما وزن كلمة مستقبِل؟","حوّل الفعل كتب إلى المضارع والأمر."]
qsrc = ",\n".join('  "'+q.replace('"','\\"')+'"' for q in qs)
tail = """
fun main() {
 val corpus = File("../src/main/assets/arabic_language_training.txt").readText(Charsets.UTF_8)
 val qs = listOf(
""" + qsrc + """
 )
 val modelFile = File("../training/model/nlm2_user.bin")
 val model = if (modelFile.exists()) Nlm.load(modelFile.readBytes()) else Nlm.fresh(corpus + "\\n" + qs.joinToString("\\n") { "س:" + it + "\\nج:" }, 32, 64)
 fun answer(q:String):String {
   val a=FloatArray(model.h); val b=FloatArray(model.h)
   var p=model.stepChar(model.cid('\\n'),a,b)
   val prefix="س:"+q+"\\nج:"
   for(ch in prefix) p=model.stepChar(model.cid(ch),a,b)
   val out=StringBuilder()
   repeat(300) {
     var best=1; var score=Float.NEGATIVE_INFINITY
     for(i in p.indices) if(p[i]>score){score=p[i];best=i}
     val ch=model.chars[best]
     if(ch=="\\n") return out.toString().trim()
     out.append(ch)
     p=model.stepChar(best,a,b)
   }
   return out.toString().trim()
 }
 val before=qs.mapIndexed { i,q -> "السؤال " + (i+1) + ": " + q + "\\nإجابة النموذج قبل التدريب: " + answer(q) }
 val samples = ArrayList<String>()
 samples.add(corpus)
 for(i in qs.indices) {
   val section = corpus.split(Regex("(?m)^\\\\d+\\\\.\\\\s+")).drop(1).getOrNull(i).orEmpty()
   val reference = section.split(Regex("[\\\\n.!؟]")).firstOrNull { it.trim().length>25 }?.trim().orEmpty()
   samples.add("س:" + qs[i] + "\\nج:" + reference + "\\n")
 }
 repeat(20) { epoch ->
   for(sample in samples) model.trainWindow(sample, 0.00012f)
   println("اكتملت دورة التدريب " + (epoch+1) + "/20")
 }
 val after=qs.mapIndexed { i,q -> "السؤال " + (i+1) + ": " + q + "\\nإجابة النموذج بعد التدريب: " + answer(q) }
 File("../training/model").mkdirs()
 val weightTemp = File("../training/model/nlm2_user.bin.tmp")
 weightTemp.writeBytes(model.toBytes())
 if (modelFile.exists()) File("../training/model/nlm2_user.bin.bak").writeBytes(modelFile.readBytes())
 if (!weightTemp.renameTo(modelFile)) { modelFile.writeBytes(weightTemp.readBytes()); weightTemp.delete() }
 File("../training/model/results.txt").writeText((before+after).joinToString("\\n\\n"),Charsets.UTF_8)
 println("تم حفظ أوزان NLM بعد التدريب: " + model.paramCount() + " معامل")
 println((before+after).joinToString("\\n\\n"))
}
"""
out.write_text(header + model + tail, encoding="utf-8")
