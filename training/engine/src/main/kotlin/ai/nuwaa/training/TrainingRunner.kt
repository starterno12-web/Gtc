package ai.nuwaa.training

import java.io.File
import java.nio.charset.StandardCharsets
import kotlin.math.max

private fun arg(args: Array<String>, key: String, fallback: String? = null): String {
    val i = args.indexOf(key)
    return if (i >= 0 && i + 1 < args.size) args[i + 1] else fallback ?: error("Missing argument: $key")
}
private fun corpusVocab(text: String): List<String> {
    val counts = HashMap<String, Int>()
    for (c in text) {
        val token = c.toString()
        counts[token] = (counts[token] ?: 0) + 1
    }
    val ranked = counts.entries
        .filter { it.key != "\n" && it.key != "�" }
        .sortedByDescending { it.value }
        .take(510)
        .map { it.key }
    return listOf("\n", "�") + ranked
}
private fun score(model: Nlm, text: String): Float {
    val ids = model.encode(text)
    return model.bpc(ids, 128)
}

/** Generate a real continuation from the trained NLM2 weights using its recurrent GRU state. */
private fun generateAnswer(model: Nlm, question: String, seed: Long): String {
    val state1 = FloatArray(96)
    val state2 = FloatArray(96)
    val prompt = "س: $question\nج:"
    var probabilities = FloatArray(0)
    for (ch in prompt) {
        probabilities = model.stepChar(model.encode(ch.toString())[0], state1, state2)
    }
    val rng = java.util.Random(seed)
    val out = StringBuilder()
    val maxChars = 100
    repeat(maxChars) { step ->
        val candidates = probabilities.indices
            .filter { model.chars[it] != "\n" && model.chars[it] != "�" }
            .sortedByDescending { probabilities[it] }
            .take(8)
        if (candidates.isEmpty()) return@repeat
        val temperature = 0.85
        val weights = candidates.map { Math.pow(probabilities[it].toDouble().coerceAtLeast(1e-12), 1.0 / temperature) }
        val total = weights.sum()
        var pick = rng.nextDouble() * total
        var chosen = candidates.last()
        for (i in candidates.indices) {
            pick -= weights[i]
            if (pick <= 0.0) { chosen = candidates[i]; break }
        }
        val token = model.chars[chosen]
        if (token == "\n" && step >= 8) return out.toString().trim()
        out.append(token)
        probabilities = model.stepChar(chosen, state1, state2)
    }
    return out.toString().trim()
}
fun main(args: Array<String>) {
    require(args.isNotEmpty()) { "Expected train, verify, or qa" }
    when (args[0]) {
        "train" -> {
            val corpusFile = File(arg(args, "--corpus"))
            val epochs = arg(args, "--epochs", "2").toInt().coerceIn(1, 50)
            val lr = arg(args, "--learning-rate", "0.0004").toFloat()
            require(lr > 0f && lr <= 0.01f) { "learning rate must be in (0, 0.01]" }
            val out = File(arg(args, "--output"))
            val report = File(arg(args, "--report"))
            require(corpusFile.isFile && corpusFile.length() > 0) { "corpus file missing or empty" }
            val text = corpusFile.readText(Charsets.UTF_8).removePrefix("\uFEFF")
            require(text.length >= 1000) { "corpus is too short" }
            val chars = corpusVocab(text)
            require(chars.size >= 3)
            // Bound training work while sampling across the entire source file, not just its beginning.
            val maxTrainingChars = 8_192
            val trainingText = if (text.length <= maxTrainingChars) text else buildString(maxTrainingChars) {
                val segments = 200
                val segmentSize = maxTrainingChars / segments
                for (i in 0 until segments) {
                    val start = ((text.length - segmentSize).toLong() * i / (segments - 1)).toInt()
                    append(text, start, minOf(start + segmentSize, text.length))
                }
            }
            val model = Nlm.create(48, 96, chars, 1729L)
            val window = 128
            var seen = 0L
            val evalText = text.take(minOf(text.length, 100_000))
            val before = score(model, evalText)
            for (epoch in 0 until epochs) {
                var pos = 0
                while (pos < trainingText.length - 1) {
                    val end = minOf(pos + window, trainingText.length)
                    if (end - pos >= 8) model.trainWindow(trainingText.substring(pos, end), lr)
                    seen += (end - pos).toLong()
                    pos = end
                }
                println("epoch ${epoch + 1}/$epochs; processed $seen characters across sampled corpus")
            }
            val after = score(model, evalText)
            require(after.isFinite() && before.isFinite()) { "non-finite training loss" }
            out.parentFile?.mkdirs()
            out.writeBytes(model.toBytes())
            val json = """{"engine":"Nlm Kotlin GRU","format":"NLM2","corpus_characters":${text.length},"vocabulary_size":${chars.size},"embedding":48,"hidden":96,"epochs":$epochs,"learning_rate":$lr,"evaluation_bpc_before":$before,"evaluation_bpc_after":$after,"processed_characters":$seen,"training_sample_characters":${trainingText.length},"weights_bytes":${out.length()},"note":"Evaluation is on the first 100k corpus characters; this is a character-model loss metric, not semantic QA accuracy."}"""
            report.parentFile?.mkdirs(); report.writeText(json + "\n")
            println(json)
        }
        "verify" -> {
            val f = File(arg(args, "--weights"))
            require(f.isFile && f.length() > 32) { "weights file missing or too small" }
            val model = Nlm.load(f.readBytes())
            require(model.paramCount() > 0 && model.v >= 3)
            val probe = model.bpc(model.encode("اختبار النموذج العربي\n"), 32)
            require(probe.isFinite()) { "model produced non-finite loss" }
            println("Verified NLM2: vocab=${model.v}, parameters=${model.paramCount()}, probe_bpc=$probe")
        }
        "qa" -> {
            val weights = File(arg(args, "--weights"))
            val questions = File(arg(args, "--questions"))
            val minimum = arg(args, "--minimum-questions", "80").toInt()
            val report = File(arg(args, "--report"))
            val model = Nlm.load(weights.readBytes())
            val arr = org.json.JSONArray(questions.readText(Charsets.UTF_8))
            require(arr.length() >= minimum) { "only ${arr.length()} QA questions; expected at least $minimum" }
            val rows = org.json.JSONArray()
            for (i in 0 until arr.length()) {
                val q = arr.getJSONObject(i).optString("question")
                require(q.isNotBlank()) { "empty QA question at index $i" }
                val answer = generateAnswer(model, q, 20261121L + i)
                rows.put(org.json.JSONObject()
                    .put("id", arr.getJSONObject(i).optInt("id", i + 1))
                    .put("question", q)
                    .put("generated_answer", answer)
                    .put("answer_characters", answer.length)
                    .put("character_model_bpc", score(model, "س:$q\nج:")))
            }
            report.parentFile?.mkdirs()
            report.writeText(org.json.JSONObject().put("count", rows.length()).put("metric", "generated_text_plus_character_model_bpc").put("generation_engine", "Nlm.stepChar").put("semantic_accuracy_evaluated", false).put("results", rows).toString(2) + "\n")
            println("Completed $minimum+ character-model probes; semantic correctness is not claimed.")
        }
        else -> error("Unknown command: ${args[0]}")
    }
}
