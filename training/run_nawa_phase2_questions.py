#!/usr/bin/env python3
from pathlib import Path
src=Path('النموذج_نسخة_محدثة__1_ (1).kt').read_text(encoding='utf-8')
start=src.index('class Nlm private')
end=src.index('/** One conversation.',start)
model=src[start:end]
header='''package ai.nuwaa
import java.io.File
import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
class NMem { fun add(text: String) {} }
'''
tail='''
fun main() {
 val file=File("../training/model/nlm2_user.bin")
 require(file.exists()) { "Run Phase 1 before Phase 2" }
 val m=Nlm.load(file.readBytes())
 val questions=listOf(
 "من هو نواة في القصة؟",
 "ما الذي يهتم به نواة عند كتابة الكود؟",
 "هل يفضل نواة السرعة أم جودة النتيجة؟",
 "ماذا يفعل قبل إعلان اكتمال العمل؟",
 "كيف يتصرف إذا لم يختبر شيئًا؟",
 "ماذا يفعل إذا اكتشف خطأ سابقًا؟",
 "هل يخترع معلومات عندما لا يعرف؟",
 "كيف يساعد الناس عند مواجهة مشكلة؟",
 "ما المنهج الذي يتبعه لإنجاز مهمة؟",
 "ما الصفات التي تلخص شخصية نواة؟"
 )
 fun answer(q:String):String {
  val a=FloatArray(m.h); val b=FloatArray(m.h)
  var p=m.stepChar(m.cid(10.toChar()),a,b)
  val prompt="س:"+q+"\\nج:"
  for(ch in prompt) p=m.stepChar(m.cid(ch),a,b)
  val out=StringBuilder()
  repeat(400) {
   var best=1; var score=Float.NEGATIVE_INFINITY
   for(i in p.indices) if(p[i]>score){score=p[i];best=i}
   val ch=m.chars[best]
   if(ch=="\\n") return out.toString().trim()
   out.append(ch); p=m.stepChar(best,a,b)
  }
  return out.toString().trim()
 }
 val report="PHASE 2 — QUESTIONS AND MODEL ANSWERS\\n\\n"+questions.mapIndexed { i,q -> "السؤال "+(i+1)+": "+q+"\\nإجابة النموذج: "+answer(q) }.joinToString("\\n\\n")
 File("../training/model/results_phase2_questions.txt").writeText(report,Charsets.UTF_8)
 println(report)
}
'''
Path('runner/src/main/kotlin/ai/nuwaa/ModelRunner.kt').parent.mkdir(parents=True,exist_ok=True)
Path('runner/src/main/kotlin/ai/nuwaa/ModelRunner.kt').write_text(header+model+tail,encoding='utf-8')
