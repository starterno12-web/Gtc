package ai.nuwaa

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.Files

private val trainingPairs = listOf(
    "س:ما عاصمة المملكة العربية السعودية؟\nج:الرياض هي عاصمة المملكة العربية السعودية.",
    "س:ما عاصمة مصر؟\nج:القاهرة هي عاصمة مصر.",
    "س:ما عاصمة فرنسا؟\nج:باريس هي عاصمة فرنسا.",
    "س:ما أكبر كواكب المجموعة الشمسية؟\nج:المشتري هو أكبر كواكب المجموعة الشمسية.",
    "س:ما الكوكب المعروف بالكوكب الأحمر؟\nج:المريخ هو الكوكب المعروف بالكوكب الأحمر.",
    "س:ما الغاز الذي تمتصه النباتات في عملية البناء الضوئي؟\nج:تمتص النباتات ثاني أكسيد الكربون وتستخدم ضوء الشمس لصنع غذائها.",
    "س:ما الصيغة الكيميائية للماء؟\nج:الصيغة الكيميائية للماء هي H2O.",
    "س:كم عدد قارات العالم؟\nج:المتعارف عليه سبع قارات.",
    "س:ما اللغة الرسمية في البرازيل؟\nج:اللغة الرسمية في البرازيل هي البرتغالية.",
    "س:من وضع قوانين الحركة الثلاثة؟\nج:صاغ إسحاق نيوتن قوانين الحركة الثلاثة.",
    "س:ما العضو الذي يضخ الدم في جسم الإنسان؟\nج:القلب يضخ الدم إلى أنحاء الجسم.",
    "س:ما وحدة قياس شدة التيار الكهربائي؟\nج:الأمبير هو وحدة قياس شدة التيار الكهربائي.",
    "س:ما ناتج 12 ضرب 8؟\nج:الناتج 96.",
    "س:ما عاصمة اليابان؟\nج:طوكيو هي عاصمة اليابان.",
    "س:ما أكبر محيط على الأرض؟\nج:المحيط الهادئ هو أكبر محيط على الأرض.",
    "س:ما وظيفة الجذور في النبات؟\nج:تمتص الجذور الماء والأملاح وتثبت النبات في التربة."
).joinToString("\n")

private val evaluationQuestions = listOf(
    "ما عاصمة المملكة العربية السعودية؟",
    "ما الكوكب الأحمر؟",
    "ما الصيغة الكيميائية للماء؟",
    "ما أكبر كواكب المجموعة الشمسية؟",
    "ما اللغة الرسمية في البرازيل؟",
    "ما العضو الذي يضخ الدم؟",
    "ما وحدة قياس التيار الكهربائي؟",
    "ما أكبر محيط على الأرض؟"
)

internal fun defaultVocabulary(): List<String> =
    (" \n\t" + """.,!?:;'"()[]{}-_=+*/\\<>@#$%^&|~`""" +
        "0123456789" +
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" +
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰ")
        .toCharArray().distinct().map { it.toString() }

private fun compatibleText(text: String, model: Nlm): String {
    val allowed = model.chars.toHashSet()
    return buildString(text.length) {
        for (ch in text) if (allowed.contains(ch.toString())) append(ch)
    }
}

private fun readUtf8Strict(bytes: ByteArray): String =
    StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes)).toString()

private fun train() {
    val input = File(System.getenv("NUWAA_INPUT_FILE") ?: "/tmp/nuwaa-source.bin")
    require(input.isFile && input.length() in 1..(128L * 1024L * 1024L)) {
        "Drive download is missing, empty, or larger than 128 MiB."
    }
    val bytes = Files.readAllBytes(input.toPath())
    val isNlm2 = bytes.size >= 4 &&
        bytes[0] == 0x4E.toByte() && bytes[1] == 0x4C.toByte() &&
        bytes[2] == 0x4D.toByte() && bytes[3] == 0x32.toByte()

    val textCorpus = if (isNlm2) null else readUtf8Strict(bytes)
    val model = if (isNlm2) {
        println("Input format: NLM2 checkpoint; loading its existing learned weights.")
        Nlm.load(bytes)
    } else {
        println("Input format: UTF-8 text corpus; initializing the repository's NLM2 architecture.")
        Nlm.fresh(e = 88, h = 64, chars = defaultVocabulary(), seed = 20261127L)
    }

    val qa = compatibleText(trainingPairs, model)
    require(qa.length >= 200) {
        "The input model vocabulary cannot represent the Arabic QA training set; refusing unsafe partial training."
    }
    val corpus = if (isNlm2) qa else compatibleText(textCorpus!!.take(250_000), model)
    require(corpus.length >= 200) {
        "The supplied text file has too few characters supported by the model vocabulary."
    }

    val epochs = (System.getenv("NUWAA_EPOCHS") ?: "2").toInt()
    val lr = (System.getenv("NUWAA_LR") ?: "0.0003").toFloat()
    require(epochs in 1..5) { "NUWAA_EPOCHS must be between 1 and 5." }
    require(lr.isFinite() && lr in 0.000001f..0.01f) { "NUWAA_LR is outside the safe configured range." }

    println("Model shape: e=${model.e}, context=${model.h}, vocabulary=${model.v}, parameters=${model.paramCount()}.")
    println("Training corpus characters: ${corpus.length}; QA rehearsal characters: ${qa.length}.")
    val result = model.trainBig(corpus, qa, epochs, lr, bookChunk = 1000, chunk = model.h)
    println(result)
    check(result.contains("قُبلت")) {
        "Training was not accepted by the model's own BPC/rehearsal guard. Existing checkpoint will not be overwritten."
    }

    val report = StringBuilder()
    report.appendLine("Input format: ${if (isNlm2) "NLM2 checkpoint" else "UTF-8 text corpus"}")
    report.appendLine("Model: e=${model.e}, context=${model.h}, vocabulary=${model.v}, parameters=${model.paramCount()}")
    report.appendLine("Epochs: $epochs; learning rate: $lr")
    report.appendLine(result)
    report.appendLine()
    report.appendLine("Held-out general questions (generated by NlmSession after training):")
    for ((index, question) in evaluationQuestions.withIndex()) {
        val q = compatibleText(question, model)
        val session = NlmSession(model, 20261127L + index)
        val answer = session.reply(
            q, maxChars = 140, temp = 0.2f, topk = 4,
            minP = 0.25f, commit = false, memGain = 0f, shortGain = 0f
        ).first
        report.appendLine("Q: $question")
        report.appendLine("A: $answer")
        report.appendLine()
    }

    val outputBytes = model.toBytes()
    val reloaded = Nlm.load(outputBytes)
    check(reloaded.e == model.e && reloaded.h == model.h && reloaded.v == model.v) {
        "Saved NLM2 failed its reload shape check."
    }
    check(reloaded.toBytes().contentEquals(outputBytes)) {
        "Saved NLM2 failed its byte-for-byte serialization round trip."
    }
    val output = File("nlm2.bin")
    val temp = File("nlm2.bin.tmp")
    temp.writeBytes(outputBytes)
    check(temp.renameTo(output)) { "Could not atomically move trained weights to repository root." }
    report.appendLine("Saved root artifact: nlm2.bin (${output.length()} bytes)")
    File("training-report.txt").writeText(report.toString(), Charsets.UTF_8)
    println(report.toString())
}

fun main(args: Array<String>) {
    if (args.firstOrNull() == "gradcheck") runGradCheck() else train()
}
