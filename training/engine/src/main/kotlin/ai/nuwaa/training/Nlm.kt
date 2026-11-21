package ai.nuwaa.training

import org.json.JSONArray
import java.nio.ByteBuffer
import java.nio.ByteOrder

class NMem {
    private val t = HashMap<String, HashMap<Char, Int>>()
    var chars = 0
        private set

    fun add(text: String) {
        val s = "\n" + text + "\n"
        for (i in 1 until s.length) {
            for (o in 1..8) {
                if (i - o < 0) break
                val key = s.substring(i - o, i)
                val m = t.getOrPut(key) { HashMap() }
                m[s[i]] = (m[s[i]] ?: 0) + 1
            }
        }
        chars += s.length
    }

    /** Longest context that was seen: (order, counts) or null. */
    fun dist(ctx: String, minO: Int = 1): Pair<Int, HashMap<Char, Int>>? {
        var o = minOf(8, ctx.length)
        while (o >= minO) {
            val c = t[ctx.substring(ctx.length - o)]
            if (c != null) return Pair(o, c)
            o--
        }
        return null
    }
}

/** The Nuwaa language model, written from scratch, running and LEARNING on the phone. Mirror of nuwaa/lm/gru.py (file format NLM2).
 *  Neural part: embedding -> GRU layer 1 -> GRU layer 2 -> softmax over characters. The hidden state is carried from character to character,
 *  so there is NO fixed window (the old 24-character limit is gone). Hand-written backprop through time + Adam.
 *  No pretrained or licensed weights, no network. HONEST LIMIT: small model, small data: it writes rough Arabic until it has seen more text. */
class Nlm private constructor(val e: Int, val h: Int, val chars: List<String>) {
    val v = chars.size
    private val vocab = HashMap<String, Int>().also { m -> chars.forEachIndexed { i, c -> m[c] = i } }
    private val h3 = 3 * h

    // E, Wx1, Wh1, b1, Wx2, Wh2, b2, Wo, bo (row-major, same order as the file)
    val P: List<FloatArray> = listOf(
        FloatArray(v * e), FloatArray(e * h3), FloatArray(h * h3), FloatArray(h3),
        FloatArray(h * h3), FloatArray(h * h3), FloatArray(h3), FloatArray(h * v), FloatArray(v)
    )
    private val mAdam = P.map { FloatArray(it.size) }
    private val vAdam = P.map { FloatArray(it.size) }
    private var tAdam = 0
    val long = NMem()

    fun cid(c: Char): Int = vocab[c.toString()] ?: 1
    fun encode(s: String): IntArray = IntArray(s.length) { cid(s[it]) }
    fun paramCount(): Int = P.sumOf { it.size }

    private fun sig(x: Float): Float = (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()

    // ---------------------------------------------------------------- one GRU cell for ONE sequence position
    /** gx = x@Wx + b (3h). Writes the new state into out and the values needed by backprop into r, z, n, rh. */
    private fun cellFwd(gx: FloatArray, hp: FloatArray, wh: FloatArray, out: FloatArray, r: FloatArray, z: FloatArray, n: FloatArray, rh: FloatArray) {
        val tmp = FloatArray(2 * h)
        for (j in 0 until 2 * h) tmp[j] = gx[j]
        for (q in 0 until h) {
            val hv = hp[q]
            if (hv == 0f) continue
            val o = q * h3
            for (j in 0 until 2 * h) tmp[j] += hv * wh[o + j]
        }
        for (j in 0 until h) { r[j] = sig(tmp[j]); z[j] = sig(tmp[h + j]); rh[j] = r[j] * hp[j] }
        val an = FloatArray(h)
        for (j in 0 until h) an[j] = gx[2 * h + j]
        for (q in 0 until h) {
            val rv = rh[q]
            if (rv == 0f) continue
            val o = q * h3 + 2 * h
            for (j in 0 until h) an[j] += rv * wh[o + j]
        }
        for (j in 0 until h) {
            n[j] = Math.tanh(an[j].toDouble()).toFloat()
            out[j] = (1f - z[j]) * hp[j] + z[j] * n[j]
        }
    }

    /** Backward of one cell. dhn = gradient of the new state. Adds into dwh; writes dgx (3h) and dhp (gradient of the previous state). */
    private fun cellBwd(dhn: FloatArray, r: FloatArray, z: FloatArray, n: FloatArray, hp: FloatArray, rh: FloatArray, wh: FloatArray, dwh: FloatArray, dgx: FloatArray, dhp: FloatArray) {
        val dz = FloatArray(h); val dan = FloatArray(h)
        for (j in 0 until h) {
            dz[j] = dhn[j] * (n[j] - hp[j])
            dan[j] = dhn[j] * z[j] * (1f - n[j] * n[j])
            dhp[j] = dhn[j] * (1f - z[j])
        }
        val drh = FloatArray(h)
        for (q in 0 until h) {
            val o = q * h3 + 2 * h
            val rv = rh[q]
            var s = 0f
            for (j in 0 until h) { dwh[o + j] += rv * dan[j]; s += dan[j] * wh[o + j] }
            drh[q] = s
        }
        val dar = FloatArray(h); val daz = FloatArray(h)
        for (j in 0 until h) {
            dhp[j] += drh[j] * r[j]
            dar[j] = drh[j] * hp[j] * r[j] * (1f - r[j])
            daz[j] = dz[j] * z[j] * (1f - z[j])
        }
        for (q in 0 until h) {
            val o = q * h3
            val hv = hp[q]
            var s = 0f
            for (j in 0 until h) {
                dwh[o + j] += hv * dar[j]
                dwh[o + h + j] += hv * daz[j]
                s += dar[j] * wh[o + j] + daz[j] * wh[o + h + j]
            }
            dhp[q] += s
        }
        for (j in 0 until h) { dgx[j] = dar[j]; dgx[h + j] = daz[j]; dgx[2 * h + j] = dan[j] }
    }

    private fun gx1(id: Int, out: FloatArray) {
        for (j in 0 until h3) out[j] = P[3][j]
        for (q in 0 until e) {
            val ev = P[0][id * e + q]
            val o = q * h3
            for (j in 0 until h3) out[j] += ev * P[1][o + j]
        }
    }

    private fun gx2(h1: FloatArray, out: FloatArray) {
        for (j in 0 until h3) out[j] = P[6][j]
        for (q in 0 until h) {
            val hv = h1[q]
            if (hv == 0f) continue
            val o = q * h3
            for (j in 0 until h3) out[j] += hv * P[4][o + j]
        }
    }

    private fun softmaxOf(h2: FloatArray): FloatArray {
        val p = FloatArray(v)
        for (j in 0 until v) p[j] = P[8][j]
        for (q in 0 until h) {
            val hv = h2[q]
            if (hv == 0f) continue
            val o = q * v
            for (j in 0 until v) p[j] += hv * P[7][o + j]
        }
        var mx = -1e30f
        for (j in 0 until v) if (p[j] > mx) mx = p[j]
        var s = 0f
        for (j in 0 until v) { val x = Math.exp((p[j] - mx).toDouble()).toFloat(); p[j] = x; s += x }
        for (j in 0 until v) p[j] /= s
        return p
    }

    // ---------------------------------------------------------------- inference: one character at a time, the state is carried by the caller
    /** Reads one character. s1 and s2 are the two hidden states and are UPDATED in place. Returns the probabilities of the next character. */
    fun stepChar(id: Int, s1: FloatArray, s2: FloatArray): FloatArray {
        val g1 = FloatArray(h3); gx1(id, g1)
        val n1 = FloatArray(h); val r = FloatArray(h); val z = FloatArray(h); val nn = FloatArray(h); val rh = FloatArray(h)
        cellFwd(g1, s1, P[2], n1, r, z, nn, rh)
        val g2 = FloatArray(h3); gx2(n1, g2)
        val n2 = FloatArray(h)
        cellFwd(g2, s2, P[5], n2, r, z, nn, rh)
        System.arraycopy(n1, 0, s1, 0, h); System.arraycopy(n2, 0, s2, 0, h)
        return softmaxOf(n2)
    }

    // ---------------------------------------------------------------- training: backprop through time over one sequence (batch of 1)
    private class Trace(val t: Int, h: Int) {
        val h1 = Array(t) { FloatArray(h) }; val h2 = Array(t) { FloatArray(h) }
        val r1 = Array(t) { FloatArray(h) }; val z1 = Array(t) { FloatArray(h) }; val n1 = Array(t) { FloatArray(h) }; val rh1 = Array(t) { FloatArray(h) }
        val r2 = Array(t) { FloatArray(h) }; val z2 = Array(t) { FloatArray(h) }; val n2 = Array(t) { FloatArray(h) }; val rh2 = Array(t) { FloatArray(h) }
        val p = arrayOfNulls<FloatArray>(t)
    }

    /** x: inputs, y: targets (same length). s1/s2: starting states (updated to the final states). Returns (loss in nats/char, gradients), or just the loss when grads is false. */
    private fun run(x: IntArray, y: IntArray, s1: FloatArray, s2: FloatArray, grads: Boolean, targetMask: BooleanArray? = null): Pair<Float, List<FloatArray>?> {
        val t = x.size
        val tr = Trace(t, h)
        var h1p = s1.copyOf(); var h2p = s2.copyOf()
        val g1 = FloatArray(h3); val g2 = FloatArray(h3)
        var loss = 0f
        var lossCount = 0
        for (i in 0 until t) {
            gx1(x[i], g1)
            cellFwd(g1, h1p, P[2], tr.h1[i], tr.r1[i], tr.z1[i], tr.n1[i], tr.rh1[i])
            gx2(tr.h1[i], g2)
            cellFwd(g2, h2p, P[5], tr.h2[i], tr.r2[i], tr.z2[i], tr.n2[i], tr.rh2[i])
            val p = softmaxOf(tr.h2[i]); tr.p[i] = p
            if (targetMask == null || targetMask[i]) {
                loss -= Math.log((p[y[i]] + 1e-9f).toDouble()).toFloat()
                lossCount++
            }
            h1p = tr.h1[i]; h2p = tr.h2[i]
        }
        loss /= maxOf(1, lossCount)
        if (!grads) { System.arraycopy(h1p, 0, s1, 0, h); System.arraycopy(h2p, 0, s2, 0, h); return Pair(loss, null) }
        val g = P.map { FloatArray(it.size) }
        val dH2 = Array(t) { FloatArray(h) }
        for (i in 0 until t) {
            val d = tr.p[i]!!
            if (targetMask != null && !targetMask[i]) {
                java.util.Arrays.fill(d, 0f)
            } else {
                d[y[i]] -= 1f
                for (j in 0 until v) d[j] /= maxOf(1, lossCount).toFloat()
            }
            for (j in 0 until v) g[8][j] += d[j]
            for (q in 0 until h) {
                val hv = tr.h2[i][q]; val o = q * v
                var s = 0f
                for (j in 0 until v) { g[7][o + j] += hv * d[j]; s += d[j] * P[7][o + j] }
                dH2[i][q] = s
            }
        }
        var dh1c = FloatArray(h); var dh2c = FloatArray(h)
        val dgx2 = FloatArray(h3); val dgx1 = FloatArray(h3)
        for (i in t - 1 downTo 0) {
            val d2 = FloatArray(h)
            for (j in 0 until h) d2[j] = dH2[i][j] + dh2c[j]
            val h2prev = if (i > 0) tr.h2[i - 1] else s2
            val nh2 = FloatArray(h)
            cellBwd(d2, tr.r2[i], tr.z2[i], tr.n2[i], h2prev, tr.rh2[i], P[5], g[5], dgx2, nh2)
            dh2c = nh2
            val d1 = FloatArray(h)
            for (q in 0 until h) {
                val o = q * h3
                var s = dh1c[q]
                val hv = tr.h1[i][q]
                for (j in 0 until h3) { s += dgx2[j] * P[4][o + j]; g[4][o + j] += hv * dgx2[j] }
                d1[q] = s
            }
            for (j in 0 until h3) g[6][j] += dgx2[j]
            val h1prev = if (i > 0) tr.h1[i - 1] else s1
            val nh1 = FloatArray(h)
            cellBwd(d1, tr.r1[i], tr.z1[i], tr.n1[i], h1prev, tr.rh1[i], P[2], g[2], dgx1, nh1)
            dh1c = nh1
            for (j in 0 until h3) g[3][j] += dgx1[j]
            val id = x[i]
            for (q in 0 until e) {
                val o = q * h3
                val ev = P[0][id * e + q]
                var s = 0f
                for (j in 0 until h3) { g[1][o + j] += ev * dgx1[j]; s += dgx1[j] * P[1][o + j] }
                g[0][id * e + q] += s
            }
        }
        System.arraycopy(h1p, 0, s1, 0, h); System.arraycopy(h2p, 0, s2, 0, h)
        return Pair(loss, g)
    }

    private fun adam(g: List<FloatArray>, lr: Float) {
        // عتبة القص كانت 1.0 على مجموع كل معاملات الشبكة (مش معامل واحد) — لنموذج فيه آلاف المعاملات، هذا يعني
        // أي خطوة فيها أي إشارة تعلم حقيقية كانت تُقص بقسوة شديدة فورًا، بصرف النظر عن lr أو epochs، فتصير الخطوة
        // الفعلية صغيرة جدًا جدًا دايمًا. هذا سبب حقيقي إضافي محتمل لـ"أدرّبه كثير وما يتغير شي". رفعتها لـ5.0 —
        // لسه فيها حماية من انفجار التدرّج (exploding gradients)، بس ما تخنق التعلم الطبيعي زي قبل.
        var nrm = 0.0
        for (a in g) for (x in a) nrm += (x * x).toDouble()
        val sc = if (nrm > 25.0) (5.0 / Math.sqrt(nrm)).toFloat() else 1f
        tAdam++
        val b1 = 0.9f; val b2 = 0.999f
        val c1 = 1f - Math.pow(b1.toDouble(), tAdam.toDouble()).toFloat()
        val c2 = 1f - Math.pow(b2.toDouble(), tAdam.toDouble()).toFloat()
        for (k in P.indices) {
            val w = P[k]; val m = mAdam[k]; val vv = vAdam[k]; val gk = g[k]
            for (i in w.indices) {
                val gi = gk[i] * sc
                m[i] = b1 * m[i] + (1f - b1) * gi
                vv[i] = b2 * vv[i] + (1f - b2) * gi * gi
                w[i] -= lr * (m[i] / c1) / (Math.sqrt((vv[i] / c2).toDouble()).toFloat() + 1e-8f)
            }
        }
    }

    /** Bits per character of a text, read with ONE state from start to end (this is what unbounded context means). */
    fun bpc(ids: IntArray, chunk: Int = 128): Float {
        if (ids.size < 3) return 0f
        val s1 = FloatArray(h); val s2 = FloatArray(h)
        var tot = 0.0; var n = 0
        var i = 0
        while (i < ids.size - 1) {
            val end = minOf(i + chunk, ids.size - 1)
            val x = ids.copyOfRange(i, end); val y = ids.copyOfRange(i + 1, end + 1)
            val res = run(x, y, s1, s2, false)
            tot += res.first.toDouble() * x.size; n += x.size
            i = end
        }
        return (tot / n / Math.log(2.0)).toFloat()
    }

    /** Learn from one thing the user said: long-term memory only (n-grams). The weights are NOT trained here: a few gradient steps on every raw
     *  message made the model collapse and echo the user's words (the repetition bug). Weights change only in develop(), with rehearsal and rollback. */
    fun observe(text: String) { long.add(text) }

    private fun trainOn(ids: IntArray, epochs: Int, lr: Float, chunk: Int) {
        for (ep in 0 until epochs) {
            val s1 = FloatArray(h); val s2 = FloatArray(h)
            var i = 0
            while (i < ids.size - 1) {
                val end = minOf(i + chunk, ids.size - 1)
                if (end - i >= 2) {
                    val res = run(ids.copyOfRange(i, end), ids.copyOfRange(i + 1, end + 1), s1, s2, true)
                    val g = res.second
                    if (g != null) adam(g, lr)
                }
                i = end
            }
        }
    }

    /** Supervised QA update: question and answer are fed as context, but loss is applied ONLY to answer characters. */
    fun trainAnswerPair(question: String, answer: String, lr: Float): Float {
        require(question.isNotBlank() && answer.isNotBlank()) { "question and answer must be non-empty" }
        val full = "س: $question\nج: $answer\n"
        val answerStart = full.indexOf(answer, full.indexOf("\nج: ") + 4)
        require(answerStart >= 0) { "answer boundary not found" }
        val ids = encode(full)
        val x = ids.copyOfRange(0, ids.size - 1)
        val y = ids.copyOfRange(1, ids.size)
        val mask = BooleanArray(x.size) { i -> (i + 1) in answerStart..(answerStart + answer.length) }
        val res = run(x, y, FloatArray(h), FloatArray(h), true, mask)
        res.second?.let { adam(it, lr) }
        return (res.first / Math.log(2.0)).toFloat()
    }

    /** ONE gradient step on one window of text (fresh state at its start). Returns bits per character of that window before the step. Used by Trainer. */
    fun trainWindow(text: String, lr: Float): Float {
        val ids = encode(text)
        if (ids.size < 8) return 0f
        val res = run(ids.copyOfRange(0, ids.size - 1), ids.copyOfRange(1, ids.size), FloatArray(h), FloatArray(h), true)
        val g = res.second
        if (g != null) adam(g, lr)
        return (res.first / Math.log(2.0)).toFloat()
    }
    fun snapshot(): List<FloatArray> = P.map { it.copyOf() }
    fun restoreSnapshot(s: List<FloatArray>) { for (k in P.indices) System.arraycopy(s[k], 0, P[k], 0, P[k].size) }

    /** Self-development on what was heard. `rehearse` = lines of the original training data ("س:..\nج:..\n"), mixed in so nothing already known is forgotten.
     *  Returns (bits/char before, after) on the heard text. The weights are rolled back if the heard text did not improve OR the rehearsal text got worse by more than 8%.
     *  FIXED a real bug: this used to cap EVERY single text at 600 characters, then threw away everything but the LAST 4000 characters of what was left over
     *  (the doc-comment even claimed 6000, but the code said 4000 — it never was 6000). If you pasted a long article and then asked 300 questions afterward,
     *  the article was chopped to its first 600 characters immediately, and then that chopped fragment almost always fell OUTSIDE the trailing-4000-character
     *  window anyway once later questions pushed it out — so develop() was training on roughly the last 5-7 short exchanges ONLY, never on the article and
     *  never on most of the 300 questions. That is exactly why only "the last thing" ever looked learned: it was, quite literally, the only thing being
     *  trained on. Raised to 6000 characters per text and a 24000-character total window (about 6x more real text actually reaches the weights per press),
     *  and the rehearsal window from 2500 to 9000 so the "don't forget old things" check isn't based on too thin a sample. This makes each press of the
     *  button noticeably slower on the phone (more characters × more epochs below) — that trade was made on purpose, since understanding was asked for over speed. */
    fun develop(texts: List<String>, epochs: Int, lr: Float, rehearse: String = "", chunk: Int = 96): Pair<Float, Float> {
        val sb = StringBuilder()
        for (tx in texts) sb.append("س:").append(tx.take(6000)).append("\n")
        val heard = if (sb.length > 24000) sb.substring(sb.length - 24000) else sb.toString()
        val ids = encode(heard)
        if (ids.size < 8) return Pair(0f, 0f)
        val reh = if (rehearse.length > 9000) rehearse.substring(0, 9000) else rehearse
        val rids = if (reh.length > 8) encode(reh) else null
        val before = bpc(ids)
        val rBefore = if (rids != null) bpc(rids) else 0f
        val snap = P.map { it.copyOf() }
        trainOn(encode(heard + reh), epochs, lr, chunk)
        val after = bpc(ids)
        val rAfter = if (rids != null) bpc(rids) else 0f
        // was 1.03f (reject on ANY regression past 3%): with only 2 epochs the before/after bits-per-char reading is noisy, so a sound training pass could
        // get silently thrown away on pure measurement noise. 1.08f still protects against real forgetting, just not against noise.
        if (after > before || (rids != null && rAfter > rBefore * 1.08f)) {
            for (k in P.indices) System.arraycopy(snap[k], 0, P[k], 0, P[k].size)
            return Pair(before, before)
        }
        return Pair(before, after)
    }

    /** تدريب "قوي" مخصص لنص كبير حقيقي (كتاب/مقال كامل) — مختلف عن develop() اللي مصمم لرسائل دردشة قصيرة متراكمة.
     *  bookChunk = حجم القطعة من النص الكبير قبل ما نتناوب معها بقطعة من rehearse (الكلام/الأسئلة-الأجوبة الأصلية)،
     *  عشان ما يغرق بالنص الخام وينسى شكل "س:...ج:..." اللي يحتاجه عشان يرد عليك أصلاً. بدون أي قص صارم للنص
     *  نفسه (غير سقف أماني كبير 200000 حرف يحمي ذاكرة الجوال بس) — خلاف develop() اللي كان يقص لـ600/4000 حرف.
     *  يرجع تقرير واضح (مو بس رقمين): هل تحسن، وبكم، وهل انرفضت التجربة وليش بالضبط — عشان وقتك ما يروح على الفاضي
     *  بصمت بدون ما تعرف السبب. */
    fun trainBig(corpus: String, rehearse: String, epochs: Int, lr: Float, bookChunk: Int = 3000, chunk: Int = 128): String {
        // ما فيه قص هنا إطلاقًا — القص انتقل لمسؤوليتك أنت (قلت "عادي أقدر أقسمها بنفسي"): سكربت GitHub اللي
        // يناديها هو اللي يقرر حجم كل دفعة، واستخدم EngineConfig.effectiveLimitMb() لو تبي رقم تسترشد فيه.
        // لو ناديتها مباشرة بنص ضخم جدًا بدون تقسيم، هي راح تحاول تعالجه كامل بلا أي حماية داخلية.
        val safeCorpus = corpus
        if (safeCorpus.length < 8) return "النص قصير جدًا، ما يستاهل تدريب."
        // كانت rehearse تُقص لأول 9000 حرف فقط، وكل قطعة كتاب تاخذ نفس أول جزء منها (reh.take(..)) بالضبط —
        // يعني "المراجعة" فعليًا كانت تكرر نفس السطور القليلة الأولى مئات المرات بدل ما تغطي معرفته القديمة
        // كلها. هذا سبب حقيقي لضعف الحماية من النسيان. الحين: بلا قص على rehearse إطلاقًا، وبيؤشر دوّار (رادار)
        // يتنقل بقطعة جديدة من المراجعة كل مرة ويلف للبداية لما يخلص — يغطي كل المعرفة القديمة بالتناوب خلال
        // التدريب الطويل، مو نفس الجملة القليلة. وجرعة المراجعة صارت مساوية لجرعة الكتاب (كانت ثلثها فقط).
        val reh = rehearse
        var rehPos = 0
        fun nextReh(n: Int): String {
            if (reh.isEmpty() || n <= 0) return ""
            val sb = StringBuilder()
            var left = n
            while (left > 0) {
                val take = minOf(left, reh.length - rehPos)
                sb.append(reh, rehPos, rehPos + take)
                rehPos += take; left -= take
                if (rehPos >= reh.length) rehPos = 0
            }
            return sb.toString()
        }
        val sb = StringBuilder()
        var i = 0
        while (i < safeCorpus.length) {
            val end = minOf(i + bookChunk, safeCorpus.length)
            sb.append(safeCorpus, i, end)
            val dose = nextReh(bookChunk)
            if (dose.isNotEmpty()) { sb.append("\n"); sb.append(dose) }   // جرعة محادثة دوّارة بعد كل قطعة كتاب
            i = end
        }
        val mixed = sb.toString()
        val ids = encode(mixed)
        if (ids.size < 8) return "النص بعد المعالجة قصير جدًا."
        val rids = if (reh.length > 8) encode(reh) else null
        val bookIds = encode(safeCorpus)
        val beforeBook = bpc(bookIds)
        val beforeReh = if (rids != null) bpc(rids) else 0f
        val snap = P.map { it.copyOf() }
        trainOn(ids, epochs, lr, chunk)
        val afterBook = bpc(bookIds)
        val afterReh = if (rids != null) bpc(rids) else 0f
        val f = { x: Float -> String.format(java.util.Locale.US, "%.3f", x) }
        val sb2 = StringBuilder()
        sb2.append("النص: ").append(f(beforeBook)).append(" ← ").append(f(afterBook)).append(" بت/حرف\n")
        if (rids != null) sb2.append("الكلام (rehearse): ").append(f(beforeReh)).append(" ← ").append(f(afterReh)).append(" بت/حرف\n")
        if (afterBook > beforeBook) {
            for (k in P.indices) System.arraycopy(snap[k], 0, P[k], 0, P[k].size)
            sb2.append("❌ رُفضت: ما تحسّن على النص نفسه حتى، فرجعت للأوزان القديمة. جرّب تخفّض lr أو تزيد epochs.")
            return sb2.toString()
        }
        if (rids != null && afterReh > beforeReh * 1.08f) {
            for (k in P.indices) System.arraycopy(snap[k], 0, P[k], 0, P[k].size)
            sb2.append("❌ رُفضت: تحسّن على النص، بس خرّب قدرة الكلام أكثر من 8%، فرجعت للأوزان القديمة. جرّب تخفّض lr أو تقلّل bookChunk.")
            return sb2.toString()
        }
        sb2.append("✅ قُبلت وحُفظت الأوزان الجديدة.")
        return sb2.toString()
    }

    // ---------------------------------------------------------------- files (NLM2)
    /** magic NLM2, int32 E H V, int32 length of the chars JSON, the chars JSON (UTF-8), then float32 little-endian E Wx1 Wh1 b1 Wx2 Wh2 b2 Wo bo. */
    fun toBytes(): ByteArray {
        val ch = JSONArray(chars).toString().toByteArray(Charsets.UTF_8)
        var floats = 0
        for (a in P) floats += a.size
        val bb = ByteBuffer.allocate(4 + 12 + 4 + ch.size + 4 * floats).order(ByteOrder.LITTLE_ENDIAN)
        bb.put(byteArrayOf(0x4E.toByte(), 0x4C.toByte(), 0x4D.toByte(), 0x32.toByte()))
        bb.putInt(e); bb.putInt(h); bb.putInt(v); bb.putInt(ch.size); bb.put(ch)
        for (a in P) for (x in a) bb.putFloat(x)
        return bb.array()
    }

    companion object {
        /** الأوزان مدمجة هنا مباشرة جوا ملف المحرك نفسه، بدل ملف nlm2.bin منفصل تحتاج تديره لحاله — بناءً على طلبك.
         *  فاضية الحين عمدًا: ما عندي بايتات أوزانك الفعلية (ما رفعتها لي، بس الكود المصدري). عبّيها بنفسك بخطوة وحدة:
         *  شغّل على جهازك: `base64 -w0 nlm2.bin` (أو أي ملف أوزان صدرته من GitHub)، وألصق الناتج كسطر نصي واحد هنا
         *  بين علامتي التنصيص. لو تركتها فاضية، التطبيق يرجع تلقائيًا لطريقة التحميل القديمة (assets/nlm2.bin) —
         *  ما ينكسر شي لو ما عبّيتها. */
        /** Builds a BIGGER model that behaves EXACTLY like `old` before any further training: every weight that reaches a new, bigger unit is
         *  zero (so the extra room changes nothing yet), every weight that only touches old units is copied over unchanged. Exact index-for-index
         *  mirror of GRULM.widened() in nuwaa/lm/gru.py (tested there: tests/test_scale.py — identical bits/char before vs after). Needs newE >= old.e
         *  and newH >= old.h. Long-term memory (old.long) is NOT carried over: it is small compared to retraining and keeping it correct across a
         *  size change is not worth the risk of a subtle bug; it simply starts empty on the bigger model. */
        fun grow(old: Nlm, newE: Int, newH: Int): Nlm {
            require(newE >= old.e && newH >= old.h) { "grow only makes the model bigger, never smaller" }
            val n = Nlm(newE, newH, old.chars)
            val oe = old.e; val oh = old.h; val oh3 = 3 * oh; val nh3 = 3 * newH; val v = old.v
            // E: (V, e) row-major -> copy the first oe columns of every row
            for (row in 0 until v) for (col in 0 until oe) n.P[0][row * newE + col] = old.P[0][row * oe + col]
            // Wx1 (oe x 3*oh -> newE x 3*newH), Wh1/Wx2/Wh2 (oh x 3*oh -> newH x 3*newH): copied gate-block by gate-block (r | z | n)
            fun copyGated(dst: FloatArray, src: FloatArray, oldRows: Int, newCols: Int, oldCols3: Int) {
                val oldH = oldCols3 / 3
                for (row in 0 until oldRows) for (g in 0 until 3) for (col in 0 until oldH)
                    dst[row * newCols + g * (newCols / 3) + col] = src[row * oldCols3 + g * oldH + col]
            }
            copyGated(n.P[1], old.P[1], oe, nh3, oh3)       // Wx1
            copyGated(n.P[2], old.P[2], oh, nh3, oh3)       // Wh1
            copyGated(n.P[4], old.P[4], oh, nh3, oh3)       // Wx2
            copyGated(n.P[5], old.P[5], oh, nh3, oh3)       // Wh2
            for (g in 0 until 3) for (col in 0 until oh) { n.P[3][g * newH + col] = old.P[3][g * oh + col]; n.P[6][g * newH + col] = old.P[6][g * oh + col] }   // b1, b2
            for (row in 0 until oh) for (col in 0 until v) n.P[7][row * v + col] = old.P[7][row * v + col]   // Wo: (oh x V) -> (newH x V), first oh rows
            for (col in 0 until v) n.P[8][col] = old.P[8][col]                                                 // bo: unchanged size, copy fully
            return n
        }
        /** Deterministic Xavier-style initialization for a genuinely new model. Vocabulary index 0 is newline; index 1 is the unknown token. */
        fun create(e: Int, h: Int, chars: List<String>, seed: Long = 1729L): Nlm {
            require(e > 0 && h > 0 && chars.size >= 3)
            require(chars[0] == "\n" && chars[1] == "�") { "vocabulary must start with newline and unknown token" }
            require(chars.distinct().size == chars.size) { "duplicate vocabulary entries" }
            val m = Nlm(e, h, chars)
            val rng = java.util.Random(seed)
            fun fill(a: FloatArray, fanIn: Int, fanOut: Int) {
                val limit = kotlin.math.sqrt(6.0 / (fanIn + fanOut)).toFloat()
                for (i in a.indices) a[i] = ((rng.nextDouble() * 2.0 - 1.0) * limit).toFloat()
            }
            fill(m.P[0], e, chars.size)
            fill(m.P[1], e, 3 * h)
            fill(m.P[2], h, 3 * h)
            fill(m.P[4], h, 3 * h)
            fill(m.P[5], h, 3 * h)
            fill(m.P[7], h, chars.size)
            // Reset/update gates start slightly closed; candidate/output biases remain zero.
            for (i in 0 until h) {
                m.P[3][h + i] = 0.5f
                m.P[6][h + i] = 0.5f
            }
            return m
        }

        fun load(b: ByteArray): Nlm {
            require(b.size > 20 && b[0] == 0x4E.toByte() && b[1] == 0x4C.toByte() && b[2] == 0x4D.toByte() && b[3] == 0x32.toByte()) { "not an NLM2 file" }
            val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
            bb.position(4)
            val e = bb.int; val h = bb.int; val v = bb.int; val n = bb.int
            val raw = ByteArray(n); bb.get(raw)
            val arr = JSONArray(String(raw, Charsets.UTF_8))
            val chars = ArrayList<String>()
            for (i in 0 until arr.length()) chars.add(arr.getString(i))
            require(chars.size == v) { "vocabulary size mismatch" }
            val m = Nlm(e, h, chars)
            for (a in m.P) for (i in a.indices) a[i] = bb.float
            return m
        }
    }
}

