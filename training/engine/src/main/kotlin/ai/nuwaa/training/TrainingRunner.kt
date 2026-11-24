package ai.nuwaa.training

import java.io.File
import java.nio.charset.StandardCharsets
import kotlin.math.max

private fun arg(args: Array<String>, key: String, fallback: String? = null): String {
    val i = args.indexOf(key)
    return if (i >= 0 && i + 1 < args.size) args[i + 1] else fallback ?: error("Missing argument: $key")
}
private fun corpusVocab(text: String, requiredText: String = ""): List<String> {
    // Kotlin Char is a UTF-16 code unit. A primitive counter avoids allocating
    // tens of millions of one-character String objects for large corpora.
    val counts = IntArray(Char.MAX_VALUE.code + 1)
    for (c in text) counts[c.code]++
    val ranked = counts.indices.asSequence()
        .filter { it != '\\n'.code && it != '�'.code && counts[it] > 0 }
        .sortedByDescending { counts[it] }
        .take(510)
        .map { it.toChar().toString() }
        .toList()
    val rankedSet = ranked.toHashSet()
    val required = requiredText.asSequence().distinct()
        .filter { it != '\\n' && it != '�' && it.toString() !in rankedSet }
        .map { it.toString() }
        .toList()
    return listOf("\\n", "�") + ranked + required
}
private fun normalizeArabicQuestion(text: String): String {
    return text.lowercase()
        .replace(Regex("[\\u064B-\\u065F\\u0670\\u0640]"), "")
        .replace(Regex("[أإآ]"), "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .replace(Regex("[^\\p{L}\\p{N}]+"), " ")
        .trim()
}

private fun characterNgrams(text: String, n: Int = 3): Set<String> {
    val padded = " $text "
    if (padded.length < n) return setOf(padded)
    return (0..padded.length - n).map { padded.substring(it, it + n) }.toSet()
}

private fun jaccard(a: Set<String>, b: Set<String>): Float {
    if (a.isEmpty() || b.isEmpty()) return 0f
    val intersection = a.count { it in b }
    return intersection.toFloat() / (a.size + b.size - intersection)
}

/** Conservative question retrieval: known answers override character generation only for strong matches. */
private fun retrieveAnswer(question: String, pairs: List<Pair<String, String>>): Triple<String, String, Float>? {
    val q = normalizeArabicQuestion(question)
    val stopWords = setOf("ما", "ماذا", "كيف", "هل", "كم", "من", "في", "على", "إلى", "عن", "مع", "لي", "هو", "هي", "هذا", "هذه", "أريد", "ابغى", "اسم", "عاصمة", "عدد", "يوم", "أيام", "يتكون", "الأسبوع", "كلمة", "المقصود", "معنى", "اشرح", "الفرق", "بين", "لو", "سمحت", "عبارة", "ممارسة", "شخص", "جميع", "جدا")
    fun contentTokens(value: String): Set<String> = normalizeArabicQuestion(value).split(" ")
        .filter { it.isNotBlank() && it !in stopWords }
        .map { if (it.startsWith("ال") && it.length > 3) it.substring(2) else it }
        .map { when (it) { "ايام" -> "يوم"; "يوما" -> "يوم"; else -> it } }
        .toSet()
    val qTokens = contentTokens(q)
    val qNgrams = characterNgrams(q)
    val ranked = pairs.map { pair ->
        val p = normalizeArabicQuestion(pair.first)
        val tokens = contentTokens(p)
        val tokenScore = jaccard(qTokens, tokens)
        val charScore = jaccard(qNgrams, characterNgrams(p))
        Triple(pair.first, pair.second, 0.9f * tokenScore + 0.1f * charScore)
    }.sortedByDescending { it.third }
    if (ranked.isEmpty()) return null
    val best = ranked.first()
    val second = ranked.getOrNull(1)?.third ?: 0f
    if (best.third < 0.62f || best.third - second < 0.08f) return null
    return best
}

private fun score(model: Nlm, text: String): Float {
    val ids = model.encode(text)
    return model.bpc(ids, 128)
}

/** Generate a real continuation from the trained NLM2 weights using its recurrent GRU state. */
private fun generateAnswer(model: Nlm, question: String, seed: Long): String {
    val state1 = FloatArray(model.h)
    val state2 = FloatArray(model.h)
    val prompt = "س: $question\nج: "
    var probabilities = FloatArray(0)
    for (ch in prompt) {
        probabilities = model.stepChar(model.encode(ch.toString())[0], state1, state2)
    }
    val out = StringBuilder()
    val random = java.util.Random(seed)
    val recent = ArrayDeque<String>()
    val maxChars = 100
    repeat(maxChars) { step ->
        val candidates = probabilities.indices
            .filter { model.chars[it] != "�" && (step >= 8 || model.chars[it] != "\n") }
            .sortedByDescending { probabilities[it] }
            .take(5)
        if (candidates.isEmpty()) return out.toString().trim()
        val weighted = candidates.map { id ->
            val count = recent.count { it == model.chars[id] }
            probabilities[id].toDouble() / (1.0 + count * 0.8)
        }
        val total = weighted.sum()
        var draw = random.nextDouble() * total
        var chosen = candidates.last()
        for (i in candidates.indices) {
            draw -= weighted[i]
            if (draw <= 0.0) { chosen = candidates[i]; break }
        }
        val token = model.chars[chosen]
        if (token == "\n" && step >= 8) return out.toString().trim()
        out.append(token)
        recent.addLast(token)
        if (recent.size > 12) recent.removeFirst()
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
            val conversationFile = File(arg(args, "--conversations"))
            require(conversationFile.isFile && conversationFile.length() > 0) { "conversation dataset missing or empty" }
            val conversationArray = org.json.JSONArray(conversationFile.readText(Charsets.UTF_8))
            require(conversationArray.length() >= 100) { "conversation dataset must contain at least 100 pairs" }
            val conversationText = buildString {
                for (i in 0 until conversationArray.length()) {
                    val pair = conversationArray.getJSONObject(i)
                    val question = pair.optString("question").trim()
                    val answer = pair.optString("answer").trim()
                    require(question.isNotBlank() && answer.isNotBlank()) { "empty conversation pair at index $i" }
                    append("س: ").append(question).append("\nج: ").append(answer).append("\n")
                }
            }
            println("Counting corpus vocabulary from ${text.length} characters...")
            val chars = corpusVocab(text, conversationText)
            require(chars.size >= 3)

            // Keep the same bounded, evenly distributed corpus sample used by the successful
            // training runs. Training every character in a multi-million-character corpus
            // creates hundreds of thousands of expensive GRU backpropagation windows.
            val maxTrainingChars = 8_192
            val trainingText = if (text.length <= maxTrainingChars) text else buildString(maxTrainingChars) {
                val segments = 200
                val segmentSize = maxTrainingChars / segments
                for (i in 0 until segments) {
                    val start = ((text.length - segmentSize).toLong() * i / (segments - 1)).toInt()
                    append(text, start, minOf(start + segmentSize, text.length))
                }
            }
            println("Vocabulary ready: ${chars.size} symbols; corpus sample=${trainingText.length}/${text.length} characters.")
            val conversationRepeats = arg(args, "--conversation-repeats", "3").toInt().coerceIn(1, 20)
            val model = Nlm.create(48, 96, chars, 1729L)
            val window = 128
            var seen = 0L
            val evalText = text.take(minOf(text.length, 100_000))
            val conversationEval = conversationText.take(100_000)
            println("Calculating baseline evaluation...")
            val before = score(model, evalText)
            val conversationBefore = score(model, conversationEval)
            var supervisedSteps = 0L
            var supervisedLoss = 0.0
            val pairs = (0 until conversationArray.length()).map { i ->
                val obj = conversationArray.getJSONObject(i)
                Pair(obj.optString("question").trim(), obj.optString("answer").trim())
            }
            val rng = java.util.Random(1729L)
            for (epoch in 0 until epochs) {
                var pos = 0
                var corpusWindows = 0
                while (pos < trainingText.length - 1) {
                    val end = minOf(pos + window, trainingText.length)
                    if (end - pos >= 8) model.trainWindow(trainingText.substring(pos, end), lr * 0.25f)
                    seen += (end - pos).toLong()
                    corpusWindows++
                    pos = end
                    if (corpusWindows % 16 == 0 || pos >= trainingText.length - 1) {
                        println("epoch ${epoch + 1}/$epochs; corpus windows=${corpusWindows}; sample chars processed=$seen/${trainingText.length * (epoch + 1)}; supervised updates=$supervisedSteps")
                    }
                }
                repeat(conversationRepeats) { repeatIndex ->
                    val shuffled = pairs.shuffled(rng)
                    for ((pairIndex, pair) in shuffled.withIndex()) {
                        val loss = model.trainAnswerPair(pair.first, pair.second, lr)
                        supervisedLoss += loss
                        supervisedSteps++
                        if (supervisedSteps % 25L == 0L || pairIndex == shuffled.lastIndex) {
                            println("epoch ${epoch + 1}/$epochs; QA repeat=${repeatIndex + 1}/$conversationRepeats; pair=${pairIndex + 1}/${shuffled.size}; supervised updates=$supervisedSteps")
                        }
                    }
                }
            }
            val after = score(model, evalText)
            val conversationAfter = score(model, conversationEval)
            require(after.isFinite() && before.isFinite() && conversationAfter.isFinite() && conversationBefore.isFinite()) { "non-finite training loss" }
            out.parentFile?.mkdirs()
            out.writeBytes(model.toBytes())
            val json = """{"engine":"Nlm Kotlin GRU","format":"NLM2","corpus_characters":${text.length},"conversation_pairs":${conversationArray.length()},"conversation_characters":${conversationText.length},"conversation_repeats":$conversationRepeats,"vocabulary_size":${chars.size},"embedding":48,"hidden":96,"epochs":$epochs,"learning_rate":$lr,"evaluation_bpc_before":$before,"evaluation_bpc_after":$after,"conversation_bpc_before":$conversationBefore,"conversation_bpc_after":$conversationAfter,"processed_characters":$seen,"training_sample_characters":${trainingText.length},"supervised_updates":$supervisedSteps,"supervised_loss_bits_average":${if (supervisedSteps > 0) supervisedLoss / supervisedSteps else 0.0},"weights_bytes":${out.length()},"note":"Evaluation is on the first 100k corpus characters; this is a character-model loss metric, not semantic QA accuracy."}"""
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
            val conversations = File(arg(args, "--conversations", "training/conversation_pairs.json"))
            val minimum = arg(args, "--minimum-questions", "80").toInt()
            val report = File(arg(args, "--report"))
            val model = Nlm.load(weights.readBytes())
            val qaPairsJson = org.json.JSONArray(conversations.readText(Charsets.UTF_8))
            val knownPairs = (0 until qaPairsJson.length()).map { i ->
                val item = qaPairsJson.getJSONObject(i)
                Pair(item.optString("question").trim(), item.optString("answer").trim())
            }.filter { it.first.isNotBlank() && it.second.isNotBlank() }
            val arr = org.json.JSONArray(questions.readText(Charsets.UTF_8))
            require(arr.length() >= minimum) { "only ${arr.length()} QA questions; expected at least $minimum" }
            val rows = org.json.JSONArray()
            var retrievalHits = 0
            for (i in 0 until arr.length()) {
                val q = arr.getJSONObject(i).optString("question")
                require(q.isNotBlank()) { "empty QA question at index $i" }
                val retrieved = retrieveAnswer(q, knownPairs)
                val answer = retrieved?.second ?: generateAnswer(model, q, 20261121L + i)
                if (retrieved != null) retrievalHits++
                val row = org.json.JSONObject()
                    .put("id", arr.getJSONObject(i).optInt("id", i + 1))
                    .put("question", q)
                    .put("generated_answer", answer)
                    .put("answer_characters", answer.length)
                    .put("character_model_bpc", score(model, "س:$q\\nج:"))
                    .put("retrieval_used", retrieved != null)
                if (retrieved != null) row.put("matched_training_question", retrieved.first).put("retrieval_score", retrieved.third)
                rows.put(row)
            }
            report.parentFile?.mkdirs()
            report.writeText(org.json.JSONObject()
                .put("count", rows.length())
                .put("metric", "conservative_qa_retrieval_plus_character_model_fallback")
                .put("generation_engine", "retrieval_then_Nlm.stepChar")
                .put("semantic_accuracy_evaluated", false)
                .put("retrieval_hits", retrievalHits)
                .put("retrieval_threshold", 0.62)
                .put("retrieval_margin", 0.08)
                .put("results", rows).toString(2) + "\\n")
            println("Completed $minimum+ probes; retrieved known answers=$retrievalHits; semantic correctness is not claimed.")
        }
        else -> error("Unknown command: ${args[0]}")
    }
}
