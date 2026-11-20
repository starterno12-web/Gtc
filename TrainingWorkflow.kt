package ai.nuwaa

import android.content.Context
import kotlin.concurrent.thread
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Standalone source-grounded training workflow. Uses Brain's existing public hooks;
 * no edits to the engine are required.
 * Read corpus -> train 20 passes -> exact-recall quiz (up to 10 source sentences).
 * Missed questions are repeated unchanged after retraining on the entire corpus,
 * with no retry limit. Weight persistence is invoked after every full pass/retry.
 */
class TrainingWorkflow(private val context: Context, private val brain: Brain) {
    @Volatile var running = false
        private set
    private val stopRequested = AtomicBoolean(false)

    fun stop() { stopRequested.set(true) }

    private fun source(): String = context.assets.open("arabic_language_training.txt")
        .bufferedReader(Charsets.UTF_8).use { it.readText() }

    private fun segments(text: String): List<String> = text
        .replace(Regex("(?m)^[-=]{3,}\\s*$"), " ")
        .split(Regex("[\\n\\r]+|(?<=[.!؟])\\s+"))
        .map { it.trim() }
        .filter { it.length >= 30 && it.any(Char::isLetter) }
        .distinct()

    private fun normalize(s: String): String = s
        .replace(Regex("[\\u064B-\\u065F\\u0670]"), "")
        .replace("ـ", "")
        .replace(Regex("\\s+"), " ")
        .trim()

    private fun persist() { brain.saveTrained() }

    private fun trainWholeSource(model: Nlm, chunks: List<String>) {
        synchronized(brain.modelLock) {
            for (chunk in chunks) {
                if (stopRequested.get()) break
                model.trainWindow(chunk, 0.00012f)
            }
        }
        persist()
    }

    fun start(status: (String) -> Unit) {
        if (running) return
        stopRequested.set(false)
        running = true
        thread(name = "arabic-source-training-workflow") {
            try {
                val model = brain.trainModel() ?: run { status("النموذج غير جاهز."); return@thread }
                val fullText = source()
                val sentences = segments(fullText)
                if (sentences.isEmpty()) { status("ملف التدريب لا يحتوي مقاطع قابلة للتدريب."); return@thread }
                val chunks = sentences.map { it + "\\n" }
                status("قرأت ملف التدريب كاملًا (${fullText.length} حرفًا، ${chunks.size} مقطع). أبدأ 20 دورة.")
                for (round in 1..20) {
                    if (stopRequested.get()) break
                    trainWholeSource(model, chunks)
                    status("الدورة ${round}/20 — حُفظت الأوزان الدائمة.")
                }
                if (stopRequested.get()) {
                    persist()
                    status("أوقف المستخدم سير workflow؛ حُفظت آخر الأوزان.")
                    return@thread
                }

                val quiz = sentences.take(10)
                status("اختبار ${quiz.size} أسئلة من النص الأصلي. ستتكرر الأسئلة الخاطئة بلا حد.")
                quiz.forEachIndexed { index, target ->
                    val question = "أعد كتابة الجملة التالية حرفيًا كما وردت في ملف التدريب، دون شرح أو إضافة:\\n«$target»"
                    var attempt = 0
                    while (!stopRequested.get()) {
                        attempt++
                        val answer = brain.selfAttempt(question).trim()
                        val isCorrect = normalize(answer).contains(normalize(target))
                        status("السؤال ${index + 1}/${quiz.size} — محاولة $attempt\\n$question\\nإجابة النموذج: $answer\\n" + if (isCorrect) "صحيح" else "خطأ — سأعيد السؤال نفسه بعد تدريب النص كاملًا")
                        if (isCorrect) break
                        trainWholeSource(model, chunks)
                    }
                    if (stopRequested.get()) return@forEachIndexed
                }
                persist()
                status(if (stopRequested.get()) "توقف الاختبار؛ حُفظت الأوزان." else "اكتمل الاختبار؛ حُفظت الأوزان الدائمة.")
            } catch (e: Throwable) {
                try { persist() } catch (_: Throwable) {}
                status("تعذر إكمال سير التدريب: ${e.message ?: e.javaClass.simpleName}. تمت محاولة حفظ الأوزان.")
            } finally {
                running = false
                stopRequested.set(false)
            }
        }
    }
}
