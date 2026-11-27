package ai.nuwaa

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.io.FileInputStream
import java.io.InputStreamReader

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
    "ما عاصمة مصر؟",
    "ما عاصمة فرنسا؟",
    "ما عاصمة اليابان؟",
    "ما الكوكب الأحمر؟",
    "ما أكبر كواكب المجموعة الشمسية؟",
    "ما الصيغة الكيميائية للماء؟",
    "كم عدد قارات العالم؟",
    "ما اللغة الرسمية في البرازيل؟",
    "من وضع قوانين الحركة الثلاثة؟",
    "ما العضو الذي يضخ الدم في جسم الإنسان؟",
    "ما وحدة قياس شدة التيار الكهربائي؟",
    "ما ناتج 12 ضرب 8؟",
    "ما أكبر محيط على الأرض؟",
    "ما وظيفة جذور النبات؟",
    "ما الغاز الذي تمتصه النباتات في البناء الضوئي؟",
    "ما أقرب كوكب إلى الشمس؟",
    "ما النجم الأقرب إلى الأرض؟",
    "ما درجة تجمد الماء بالسلسيوس؟",
    "ما وحدة قياس القوة؟",
    "ما عاصمة إيطاليا؟",
    "ما عاصمة المملكة المتحدة؟",
    "كم ضلعًا للمثلث؟",
    "ما اسم العملية التي تصنع بها النباتات غذاءها؟",
    "ما أكبر عضو في جسم الإنسان؟",
    "ما المعدن السائل في درجة حرارة الغرفة؟",
    "ما ناتج 15 ضرب 7؟",
    "ما اللغة التي يتحدث بها معظم سكان الأرجنتين؟",
    "ما الجهاز المسؤول عن التنفس عند الإنسان؟",
    "ما الكوكب الذي نعيش عليه؟"
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

private fun readUtf8Prefix(file: File, maxChars: Int): String {
    val decoder = StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
    return InputStreamReader(FileInputStream(file), decoder).use { reader ->
        val out = StringBuilder(maxChars)
        val buffer = CharArray(8192)
        while (out.length < maxChars) {
            val count = reader.read(buffer, 0, minOf(buffer.size, maxChars - out.length))
            if (count < 0) break
            out.append(buffer, 0, count)
        }
        out.toString()
    }
}

private fun train() {
    val input = File(System.getenv("NUWAA_INPUT_FILE") ?: "/tmp/nuwaa-source.bin")
    require(input.isFile && input.length() in 1..2_000_000_000L) {
        "Drive download is missing, empty, or larger than the supported 2 GB input limit."
    }
    val fileSize = input.length()
    val prefix = ByteArray(4)
    val prefixRead = FileInputStream(input).use { it.read(prefix) }
    val isNlm2 = prefixRead == 4 &&
        prefix[0] == 0x4E.toByte() && prefix[1] == 0x4C.toByte() &&
        prefix[2] == 0x4D.toByte() && prefix[3] == 0x32.toByte()
    require(!isNlm2) { "The configured Drive input is a checkpoint, not the expected UTF-8 corpus." }

    val model = Nlm.fresh(e = 88, h = 64, chars = defaultVocabulary(), seed = 20261127L)
    val allowed = model.chars.toHashSet()
    val decoder = StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)

    // First pass: finish reading and validate every character before any training begins.
    println("Pass 1/3: scanning the complete source before training.")
    var totalChars = 0L
    val unsupported = HashMap<Int, Long>()
    InputStreamReader(FileInputStream(input), decoder).buffered(64 * 1024).use { reader ->
        val chars = CharArray(64 * 1024)
        while (true) {
            val n = reader.read(chars)
            if (n < 0) break
            totalChars += n
            for (i in 0 until n) {
                val c = chars[i]
                if (!allowed.contains(c.toString()) && c != '\uFEFF') {
                    unsupported[c.code] = (unsupported[c.code] ?: 0L) + 1L
                }
            }
        }
    }
    require(totalChars >= 200) { "The complete source contains too few characters." }
    require(unsupported.isEmpty()) {
        "The corpus contains characters outside the model vocabulary; refusing to silently skip them: " +
            unsupported.entries.sortedByDescending { it.value }.take(12)
                .joinToString { "U+" + it.key.toString(16).uppercase() + "=" + it.value }
    }
    println("Complete read and UTF-8 validation passed: " + totalChars + " characters, " + fileSize + " bytes.")

    val epochs = (System.getenv("NUWAA_EPOCHS") ?: "1").toInt()
    val lr = (System.getenv("NUWAA_LR") ?: "0.0001").toFloat()
    require(epochs in 1..5) { "NUWAA_EPOCHS must be between 1 and 5." }
    require(lr.isFinite() && lr in 0.000001f..0.01f) { "NUWAA_LR is outside the safe configured range." }

    // Second pass: train every source character using bounded batches and h-character overlap.
    println("Pass 2/3: streaming training across the complete corpus.")
    val report = StringBuilder()
    report.appendLine("Input format: complete UTF-8 text corpus")
    report.appendLine("Input bytes: " + fileSize)
    report.appendLine("Complete source characters: " + totalChars)
    report.appendLine("Model: e=" + model.e + ", context=" + model.h + ", vocabulary=" + model.v + ", parameters=" + model.paramCount())
    report.appendLine("Epochs per batch: " + epochs + "; learning rate: " + lr)
    val batchChars = 100_000
    val overlap = model.h
    val buffer = StringBuilder(batchChars + overlap * 2)
    var batches = 0
    var covered = 0L
    InputStreamReader(FileInputStream(input), StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)).buffered(64 * 1024).use { reader ->
        val chars = CharArray(64 * 1024)
        while (true) {
            val n = reader.read(chars)
            if (n < 0) break
            buffer.append(chars, 0, n)
            while (buffer.length >= batchChars + overlap) {
                val part = buffer.substring(0, batchChars + overlap).replace("\uFEFF", "")
                val result = model.trainBig(part, trainingPairs, epochs, lr, bookChunk = 1000, chunk = model.h)
                check(result.contains("قُبلت")) {
                    "Batch " + (batches + 1) + " was rejected; no incomplete checkpoint will be published. " + result
                }
                batches++
                covered += batchChars
                buffer.delete(0, batchChars)
                if (batches % 10 == 0) println("Accepted batches: " + batches + "; source coverage: " + covered + "/" + totalChars)
            }
        }
    }
    if (buffer.isNotEmpty()) {
        val tail = buffer.toString().replace("\uFEFF", "")
        if (tail.length >= 8) {
            val result = model.trainBig(tail, trainingPairs, epochs, lr, bookChunk = 1000, chunk = model.h)
            check(result.contains("قُبلت")) { "Final batch was rejected; refusing to publish partial weights. " + result }
            batches++
            covered += tail.length
        }
    }
    check(covered >= totalChars - overlap) {
        "Streaming pass did not cover the complete corpus. Covered=" + covered + ", scanned=" + totalChars
    }
    report.appendLine("Accepted training batches: " + batches)
    report.appendLine("Streaming source coverage (overlap included): " + covered)

    // Third pass in the requested sequence: ask exactly 30 questions, then serialize all model parameters.
    println("Pass 3/3: asking 30 questions after the complete corpus pass.")
    report.appendLine()
    report.appendLine("Thirty post-training questions:")
    for ((index, question) in evaluationQuestions.withIndex()) {
        val session = NlmSession(model, 20261127L + index)
        val answer = session.reply(
            question, maxChars = 140, temp = 0.2f, topk = 4,
            minP = 0.25f, commit = false, memGain = 0f, shortGain = 0f
        ).first
        report.appendLine((index + 1).toString() + ". Q: " + question)
        report.appendLine("A: " + answer)
        report.appendLine()
        println("Question " + (index + 1) + "/30 completed.")
    }

    val outputBytes = model.toBytes()
    val reloaded = Nlm.load(outputBytes)
    check(reloaded.e == model.e && reloaded.h == model.h && reloaded.v == model.v) {
        "Saved NLM2 failed its reload shape check."
    }
    check(reloaded.toBytes().contentEquals(outputBytes)) {
        "Saved NLM2 failed its byte-for-byte serialization round trip."
    }
    val workspace = File(System.getenv("GITHUB_WORKSPACE") ?: ".")
    val output = File(workspace, "nlm2.bin")
    val temp = File(workspace, "nlm2.bin.tmp")
    temp.writeBytes(outputBytes)
    check(temp.renameTo(output)) { "Could not atomically move trained weights to repository root." }
    report.appendLine("Saved root artifact: nlm2.bin (" + output.length() + " bytes)")
    File(workspace, "training-report.txt").writeText(report.toString(), Charsets.UTF_8)
    println(report.toString())
}

fun main(args: Array<String>) {
    if (args.firstOrNull() == "gradcheck") runGradCheck() else train()
}
