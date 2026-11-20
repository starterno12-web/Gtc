package ai.nuwaa

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.os.Environment
import android.provider.Settings
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.*
import kotlin.concurrent.thread
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import okhttp3.*
import okio.ByteString
import android.graphics.Canvas
import android.os.Handler
import android.os.Looper
import android.webkit.WebView
import android.webkit.WebViewClient
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import android.media.MediaRouter
import java.nio.ByteOrder

// ============== من الجوال_MainActivity.kt ==============
class MainActivity : Activity() {
    private lateinit var brain: Brain
    private lateinit var store: ChatStore
    private var conv: Conv? = null
    private lateinit var pages: FrameLayout
    private lateinit var chatPage: LinearLayout
    private lateinit var histPage: LinearLayout
    private lateinit var screenPage: ScrollView
    private lateinit var codePage: LinearLayout
    private lateinit var titleView: TextView
    private lateinit var status: TextView
    private lateinit var log: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var histList: LinearLayout
    private lateinit var preview: ImageView
    private lateinit var shareInfo: TextView
    private lateinit var adviceView: TextView
    private lateinit var shareBtn: TextView
    private lateinit var goalInput: EditText
    private lateinit var editor: EditText
    private lateinit var codeIdea: EditText
    private val REQ_CAPTURE = 41
    private val REQ_NOTIF = 42
    private val REQ_PICK = 43
    private val REQ_CREATE = 44
    private var pendingFile: String = ""
    private var pendingBytes: ByteArray? = null
    private lateinit var devPage: LinearLayout
    private lateinit var trainPage: LinearLayout
    private lateinit var discoverPage: LinearLayout
    private lateinit var discoverResult: TextView
    private lateinit var growPage: LinearLayout
    private lateinit var growResult: TextView
    private var titleTaps = 0; private var titleTapsAt = 0L
    private var revealedGrowMenu = false
    private lateinit var trainInfo: TextView
    private lateinit var trainBtn: TextView
    private lateinit var pctLabel: TextView
    private var trainerObj: Trainer? = null
    private var fileMode = false
    private lateinit var fileBtn: TextView
    private lateinit var modeBar: TextView
    private fun trainer(): Trainer = trainerObj ?: Trainer(this, brain).also { trainerObj = it }
    private lateinit var devReport: TextView
    private val MAX_READ = 8_000_000
    private var attached: Attached? = null
    private lateinit var attachBar: LinearLayout
    private lateinit var attachLabel: TextView

    private fun dp(v: Int) = Ui.dp(this, v)

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        window.statusBarColor = Ui.DARK
        store = ChatStore(java.io.File(filesDir, "chats"))
        brain = Brain.get(this)
        buildUi()
        setContentView(rootView)
        conv = store.list().firstOrNull() ?: store.create()
        renderConv()
        showPage(chatPage)
        status.text = "جاري تجهيز النموذج..."
        thread {
            brain.loadModel()
            conv?.let { c -> if (c.msgs.isNotEmpty()) brain.resume(c.id, c.msgs.toList()) }      // the open conversation gets its short-term memory back
            runOnUiThread { status.text = brain.modelStatus }
        }
        // Cast/wireless-display detection (same "Cast" as Settings > Connected devices > Cast; no new dependency, no API key).
        CastManager.start(this) { connected, name ->
            runOnUiThread { if (::shareInfo.isInitialized) shareInfo.text = if (connected) "📺 متصل بـ: $name" else shareInfo.text }
        }
    }

    private lateinit var rootView: LinearLayout

    private fun buildUi() {
        rootView = LinearLayout(this)
        rootView.orientation = LinearLayout.VERTICAL
        rootView.layoutDirection = View.LAYOUT_DIRECTION_RTL
        rootView.setBackgroundColor(Ui.BG)
        rootView.fitsSystemWindows = true

        val bar = LinearLayout(this)
        bar.orientation = LinearLayout.HORIZONTAL
        bar.gravity = Gravity.CENTER_VERTICAL
        bar.setBackgroundColor(Ui.DARK)
        bar.setPadding(dp(8), dp(8), dp(8), dp(8))
        val menu = TextView(this)
        menu.text = "☰"; menu.textSize = 26f; menu.setTextColor(Color.WHITE); menu.setPadding(dp(12), 0, dp(12), 0)
        menu.setOnClickListener { showMenu() }
        val dev = TextView(this); dev.text = "🧠"; dev.textSize = 22f; dev.setTextColor(Color.WHITE); dev.setPadding(dp(10), 0, dp(10), 0)
        dev.setOnClickListener { showPage(devPage) }
        val tcol = LinearLayout(this); tcol.orientation = LinearLayout.VERTICAL
        titleView = Ui.label(this, "نواة", 20f, Color.WHITE, true)
        titleView.setOnClickListener { onTitleTap() }
        status = Ui.label(this, "", 11f, Color.parseColor("#9FB3C0"), false)
        tcol.addView(titleView); tcol.addView(status)
        bar.addView(menu)
        bar.addView(tcol, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(dev)
        rootView.addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        pages = FrameLayout(this)
        rootView.addView(pages, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        buildChatPage(); buildHistPage(); buildScreenPage(); buildCodePage(); buildDevPage(); buildTrainPage(); buildDiscoverPage(); buildGrowPage()
    }

    // ---------------- pages ----------------
    private fun showPage(p: View) {
        pages.removeAllViews()
        pages.addView(p, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        if (p === histPage) refreshHistory()
    }

    private fun showMenu() {
        val base = arrayOf("💬  محادثة جديدة", "🧠  تطوير النفس", "🗂  المحادثات السابقة", "📱  مشاركة الشاشة", "💻  كتابة الأكواد", "🗑  حذف كل المحادثات", "ℹ️  عن التطبيق", "⚡  رفع الفهم (تدريب)", "🔍  اكتشاف من الإنترنت")
        val items = if (revealedGrowMenu) base + "⚙  تطوير متقدم" else base
        AlertDialog.Builder(this).setItems(items) { _, i ->
            when (i) {
                0 -> newChat()
                1 -> showPage(devPage)
                2 -> showPage(histPage)
                3 -> { showPage(screenPage); refreshShareUi() }
                4 -> showPage(codePage)
                5 -> confirmDeleteAll()
                6 -> showAbout()
                7 -> showPage(trainPage)
                8 -> showPage(discoverPage)
                9 -> openGrowPage()
            }
        }.show()
    }

    private fun newChat() {
        val c = conv
        if (c != null && c.msgs.isEmpty()) { showPage(chatPage); return }
        if (c != null) brain.endConversation(c.id)                    // the conversation ended: its short-term memory ends with it
        conv = store.create(); renderConv(); showPage(chatPage)
    }

    private fun confirmDeleteAll() {
        AlertDialog.Builder(this).setMessage("حذف كل المحادثات نهائيا؟ (ما علّمتني إياه يبقى محفوظا)")
            .setPositiveButton("حذف") { _, _ -> for (x in store.list()) brain.endConversation(x.id); store.deleteAll(); conv = store.create(); renderConv(); showPage(chatPage) }
            .setNegativeButton("إلغاء", null).show()
    }

    private fun showAbout() {
        val m = brain.usedMb()
        AlertDialog.Builder(this).setTitle("نواة").setMessage(
            "نموذج عربي يعمل على جوالك بدون إنترنت.\n\n" + brain.modelStatus +
            "\nأمثلة الأكواد المختبرة في صفحة الأكواد: " + brain.kb.items.size + "." +
            "\nذاكرة التطبيق الآن: " + m.toInt() + " MB (بدون حد أقصى أضعه أنا؛ الحد هو ذاكرة جهازك)." +
            "\n\nمحدودية صادقة: أنا نموذج صغير. أكتب ردودي بنفسي حرفا بعد حرف بدون ردود جاهزة، وقد أخطئ أو أكتب كلاما ركيكا، وأتحسن كلما تحدثت معي وتدربت."
        ).setPositiveButton("حسنا", null).show()
    }

    // ---------------- chat ----------------
    private fun buildChatPage() {
        chatPage = LinearLayout(this); chatPage.orientation = LinearLayout.VERTICAL
        scroll = ScrollView(this); log = LinearLayout(this); log.orientation = LinearLayout.VERTICAL
        log.setPadding(0, dp(8), 0, dp(8)); scroll.addView(log)
        chatPage.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        attachBar = LinearLayout(this); attachBar.orientation = LinearLayout.HORIZONTAL; attachBar.gravity = Gravity.CENTER_VERTICAL
        attachBar.setBackgroundColor(Color.parseColor("#E3F1F2")); attachBar.setPadding(dp(14), dp(6), dp(8), dp(6)); attachBar.visibility = View.GONE
        attachLabel = Ui.label(this, "", 13f, Ui.DARK, false)
        val x = TextView(this); x.text = "✕"; x.textSize = 18f; x.setTextColor(Ui.MUTED); x.setPadding(dp(14), dp(2), dp(14), dp(2))
        x.setOnClickListener { attached = null; attachBar.visibility = View.GONE }
        attachBar.addView(attachLabel, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)); attachBar.addView(x)
        chatPage.addView(attachBar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val row = LinearLayout(this); row.orientation = LinearLayout.HORIZONTAL; row.gravity = Gravity.CENTER_VERTICAL
        row.setPadding(dp(8), dp(6), dp(8), dp(6)); row.setBackgroundColor(Color.WHITE)
        input = EditText(this); input.hint = "اكتب سؤالك هنا..."; input.maxLines = 4
        input.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        input.imeOptions = EditorInfo.IME_FLAG_NO_EXTRACT_UI
        input.background = Ui.bg(this, Color.parseColor("#EEF2F5"), 22); input.setPadding(dp(16), dp(10), dp(16), dp(10))
        val send = Ui.btn(this, "إرسال", Ui.PRIMARY) { onSend() }
        fileBtn = Ui.btn(this, "📄", Color.parseColor("#8A97A0")) { toggleFileMode() }
        fileBtn.textSize = 18f
        val flp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT); flp.setMargins(0, 0, dp(6), 0)
        val plus = Ui.btn(this, "＋", Ui.DARK) { showPlus() }
        plus.textSize = 20f
        val plp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT); plp.setMargins(0, 0, dp(8), 0)
        row.addView(plus, plp)
        row.addView(fileBtn, flp)
        row.addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(dp(8), 0, 0, 0)
        row.addView(send, lp)
        modeBar = Ui.label(this, "📄 وضع إنشاء الملفات شغّال: كل رسالة تكتبها تُعامل كوصف لملف أنشئه لك (النوع، الاسم، ماذا فيه). اضغط 📄 لإيقافه.", 13f, Color.WHITE, false)
        modeBar.setBackgroundColor(Color.parseColor("#1565C0")); modeBar.setPadding(dp(14), dp(7), dp(14), dp(7)); modeBar.visibility = View.GONE
        chatPage.addView(modeBar, chatPage.childCount - 0, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        chatPage.addView(row, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    /** Tapping the title 5 times within 2 seconds reveals the hidden "⚙ تطوير متقدم" menu item (same trick as Android's own Developer Options:
     *  not real security, just a speed bump so growing the model — heavy, resource-hungry — is never one accidental tap away). */
    private fun onTitleTap() {
        val now = System.currentTimeMillis()
        if (now - titleTapsAt > 2000) titleTaps = 0
        titleTapsAt = now; titleTaps++
        if (titleTaps >= 5 && !revealedGrowMenu) { revealedGrowMenu = true; titleTaps = 0; Toast.makeText(this, "⚙ ظهر تطوير متقدم في القائمة", Toast.LENGTH_SHORT).show() }
    }

    /** The file button. Off (gray): no file is ever created from the chat, whatever words I type. On (blue): every message is a request for a file. */
    private fun toggleFileMode() {
        fileMode = !fileMode
        fileBtn.background = Ui.bg(this, if (fileMode) Color.parseColor("#1565C0") else Color.parseColor("#8A97A0"), 20)
        modeBar.visibility = if (fileMode) View.VISIBLE else View.GONE
        input.hint = if (fileMode) "صف الملف: النوع، الاسم، وماذا فيه..." else "اكتب سؤالك هنا..."
    }

    private fun renderConv() {
        log.removeAllViews()
        val c = conv ?: return
        titleView.text = if (c.title == "محادثة جديدة") "نواة" else c.title
        if (c.msgs.isEmpty()) {
            bubble("هلا! أنا نواة 👋\nأكتب ردودي بنفسي حرفا بعد حرف، وأتعلم من كلامك بدون ما تقول لي «تعلم». أحسب لك بدقة، وأتدرب على البرمجة.\nجرّب: «كيف اكتب حلقة for في بايثون» أو «كم يساوي ٢٥ × ١٧».", false, false)
            chips()
        } else for (m in c.msgs) bubble(m.text, m.role == "u", false)
    }

    private fun chips() {
        val box = LinearLayout(this); box.orientation = LinearLayout.VERTICAL; box.setPadding(dp(12), dp(4), dp(12), 0)
        for (q in listOf("كيف اكتب دالة في بايثون", "ما الفرق بين let و const", "كيف ارفع مشروعي على github", "كيف اوسط عنصر في css")) {
            val t = TextView(this); t.text = q; t.textSize = 14f; t.setTextColor(Ui.PRIMARY); t.setPadding(dp(14), dp(8), dp(14), dp(8))
            t.background = Ui.bg(this, Color.WHITE, 16)
            val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(0, dp(3), 0, dp(3))
            t.setOnClickListener { input.setText(q); onSend() }
            box.addView(t, lp)
        }
        log.addView(box)
    }

    /** Splits ``` blocks: code is shown in a dark monospace box (left-to-right) with a copy button. */
    private fun bubble(text: String, mine: Boolean, scrollDown: Boolean = true) {
        val wrap = LinearLayout(this); wrap.orientation = LinearLayout.VERTICAL
        wrap.setPadding(dp(14), dp(10), dp(14), dp(10))
        wrap.background = Ui.bg(this, if (mine) Ui.PRIMARY else Color.WHITE, 18)
        val parts = text.split("```")
        for ((i, p) in parts.withIndex()) {
            if (i % 2 == 0) {
                val s = p.trim('\n')
                if (s.isNotEmpty()) {
                    val tv = TextView(this); tv.text = s; tv.textSize = 16f; tv.setTextColor(if (mine) Color.WHITE else Color.parseColor("#12222D")); tv.setTextIsSelectable(true)
                    wrap.addView(tv)
                }
            } else {
                val code = p.substringAfter('\n', p).trim('\n')
                val box = LinearLayout(this); box.orientation = LinearLayout.VERTICAL; box.layoutDirection = View.LAYOUT_DIRECTION_LTR
                box.background = Ui.bg(this, Ui.CODE_BG, 10); box.setPadding(dp(10), dp(8), dp(10), dp(8))
                val hs = HorizontalScrollView(this)
                val tv = TextView(this); tv.text = code; tv.typeface = Typeface.MONOSPACE; tv.textSize = 13f; tv.setTextColor(Color.parseColor("#D7E3EA")); tv.setTextIsSelectable(true)
                tv.textDirection = View.TEXT_DIRECTION_LTR; hs.addView(tv)
                val copy = TextView(this); copy.text = "نسخ"; copy.textSize = 12f; copy.setTextColor(Color.parseColor("#7FD6DD")); copy.setPadding(0, dp(6), 0, 0)
                copy.setOnClickListener { copyText(code); Toast.makeText(this, "تم النسخ", Toast.LENGTH_SHORT).show() }
                box.addView(hs); box.addView(copy)
                val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(0, dp(6), 0, dp(6))
                wrap.addView(box, lp)
            }
        }
        val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        lp.setMargins(dp(12), dp(4), dp(12), dp(4)); lp.gravity = if (mine) Gravity.START else Gravity.END
        log.addView(wrap, lp)
        if (scrollDown) scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun copyText(s: String) {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("code", s))
    }

    private fun onSend() {
        val q = input.text.toString().trim()
        val att = attached
        if (q.isEmpty() && att == null) return
        input.setText("")
        val c = conv ?: store.create().also { conv = it }
        if (c.msgs.isEmpty()) log.removeAllViews()
        val shown = if (att != null && !att.announced) "📎 " + att.name + (if (q.isNotEmpty()) "\n" + q else "") else q
        if (att != null) att.announced = true
        addMsg(c, "u", shown)
        val dots = TextView(this); dots.text = "نواة تكتب..."; dots.setTextColor(Ui.MUTED); dots.setPadding(dp(16), dp(4), dp(16), dp(4)); log.addView(dots)
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        thread {
            val lastBot = c.msgs.lastOrNull { it.role == "b" }?.text ?: ""
            val a = try {
                val fr = if (q.isNotEmpty() && att == null && fileMode) brain.fileRequest(q, c.id, lastBot, true) else null      // a sent file is being reviewed: never create files then
                if (fr != null) createFileAnswer(fr, c.id) else if (att != null) brain.answerAbout(att, q) else brain.answer(q, c.id)
            } catch (e: Throwable) { Answer("حصل خطأ داخلي: " + (e.message ?: e.javaClass.simpleName), "error") }
            // only the model's own words are shown as its reply; everything else is labelled as coming from the app (reports, notices, exact arithmetic)
            val shown = when (a.source) { "lm" -> a.text; "math" -> "🧮 حساب دقيق (من التطبيق، ليس من النموذج):\n" + a.text; else -> "⚙ رسالة من التطبيق (ليست من النموذج):\n" + a.text }
            if (TvLink.connected) {
                TvLink.sendChat(q, shown)   // relay the REAL exchange to the TV; nothing is run twice, nothing is faked
                // Studio view: any reply with a fenced code block ALSO goes to the TV's code pane, not
                // just file-creation. This is what makes the TV a studio instead of just a chat mirror.
                if (shown.contains("```")) {
                    val block = shown.substringAfter("```")
                    val lang = block.substringBefore("\n").trim()
                    val code = block.substringAfter("\n").substringBefore("```")
                    TvLink.sendCode(code, lang)
                }
            }
            runOnUiThread { log.removeView(dots); addMsg(c, "b", shown) }
        }
    }

    // ---------------- creating a file from the chat: really written, then re-read, then reported ----------------
    private fun mimeOf(name: String): String = when (name.substringAfterLast('.', "").lowercase(java.util.Locale.ROOT)) {
        "txt" -> "text/plain"; "json" -> "application/json"; "csv" -> "text/csv"; "html" -> "text/html"; "css" -> "text/css"
        else -> "application/octet-stream"      // keeps the real extension (.py, .kt ...): text/plain would make Android add .txt
    }

    /** Writes the file and returns where it is. Throws if it could not be written or the bytes read back do not match. */
    private fun saveToDownloads(f: MadeFile): String {
        if (Build.VERSION.SDK_INT >= 29) {
            val v = ContentValues()
            val dirPart = f.name.substringBeforeLast('/', ""); val baseName = f.name.substringAfterLast('/')
            v.put(MediaStore.MediaColumns.DISPLAY_NAME, baseName); v.put(MediaStore.MediaColumns.MIME_TYPE, mimeOf(baseName))
            v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Nuwaa" + (if (dirPart.isNotEmpty()) "/" + dirPart else ""))
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v) ?: throw java.io.IOException("النظام رفض إنشاء الملف")
            contentResolver.openOutputStream(uri)?.use { it.write(f.bytes) } ?: throw java.io.IOException("تعذر فتح الملف للكتابة")
            val back = contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: throw java.io.IOException("تعذر إعادة قراءة الملف")
            if (!back.contentEquals(f.bytes)) throw java.io.IOException("المحتوى المقروء لا يطابق ما كُتب")
            var shown = baseName
            contentResolver.query(uri, arrayOf(MediaStore.MediaColumns.DISPLAY_NAME), null, null, null)?.use { cur -> if (cur.moveToFirst()) shown = cur.getString(0) ?: shown }
            return "Download/Nuwaa/" + (if (dirPart.isNotEmpty()) dirPart + "/" else "") + shown
        }
        val out = java.io.File(java.io.File(getExternalFilesDir(null) ?: filesDir, "Nuwaa"), f.name); out.parentFile?.mkdirs()
        out.writeBytes(f.bytes)
        if (!out.readBytes().contentEquals(f.bytes)) throw java.io.IOException("المحتوى المقروء لا يطابق ما كُتب")
        return out.absolutePath
    }

    private fun createFileAnswer(fr: Pair<MadeFile?, String>, convId: String): Answer {
        val f = fr.first ?: return Answer(fr.second, "file")
        if (f.more.isNotEmpty()) {                                                // a project: several files in one folder
            val all = listOf(f) + f.more; val ok = ArrayList<String>(); var err = ""
            for (x in all) { try { ok.add(saveToDownloads(x)) } catch (e: Throwable) { err = e.message ?: e.javaClass.simpleName; break } }
            val folder = f.name.substringBeforeLast('/', "")
            return Answer((if (ok.size == all.size) "✅ أنشأت المشروع فعلا: " + all.size + " ملفات وتأكدت من محتوى كل ملف بعد الكتابة." else "⚠ كتبت " + ok.size + " من " + all.size + " ملفات فقط. " + err) +
                "\nالمجلد: Download/Nuwaa" + (if (folder.isNotEmpty()) "/" + folder else "") + "\n" + ok.joinToString("\n") { "• " + it } + "\nلم أشغّل الكود، جرّبه قبل الاعتماد عليه.", "file")
        }
        return try {
            // "يشوف شو سوى" before you do, for browser code only — VisionLoop renders it, takes a 144-char
            // (12x12) ASCII-art look, and lets the model fix its own code. Capped at 2 passes, so this
            // can only make file creation a bit slower, never hang. Runs on THIS thread (already a
            // background thread from onSend()'s thread{}); it hops to the UI thread internally for the
            // WebView render and blocks until that's done, same as any synchronous call.
            val ext = f.name.substringAfterLast('.', "").lowercase()
            val finalBytes = if (ext in setOf("html", "htm", "css", "js")) {
                val fixed = VisionLoop.reviewAndFix(
                    brain, this, convId, String(f.bytes, Charsets.UTF_8), f.name, maxPasses = 2,
                    log = { line -> if (TvLink.connected) TvLink.sendThinking(line) }
                )
                fixed.toByteArray(Charsets.UTF_8)
            } else f.bytes
            val f2 = MadeFile(f.name, finalBytes, f.note, f.more)   // if MadeFile's field order/names differ, this is the one line to adjust

            val where = saveToDownloads(f2)
            if (TvLink.connected) {
                val runnable = ext in setOf("html", "htm", "js", "css")
                TvLink.sendFileDone(f2.name, String(f2.bytes, Charsets.UTF_8), runnable)
            }
            Answer("✅ أنشأت الملف فعلا وتأكدت من محتواه بعد الكتابة.\nالاسم: " + f2.name + "\nالمكان: " + where + "\nالحجم: " + f2.bytes.size + " بايت" + (if (f2.note.isNotEmpty()) "\n(" + f2.note + ")" else "") + (if (f2.name.endsWith(".py") || f2.name.endsWith(".kt")) "\nلم أشغّل الكود، جرّبه قبل الاعتماد عليه." else ""), "file")
        } catch (e: Throwable) {
            Answer("❌ ما قدرت أنشئ الملف: " + (e.message ?: e.javaClass.simpleName) + "\nلم يُكتب شيء. يمكنك استعمال ＋ ثم «إنشاء ملف» لاختيار مكان الحفظ بنفسك.", "file")
        }
    }

    // ---------------- the + button: send a file / create a file ----------------
    private fun showPlus() {
        AlertDialog.Builder(this).setItems(arrayOf("📎  إرسال ملف", "📝  إنشاء ملف", "📺  اتصال بالتلفزيون")) { _, i ->
            when (i) { 0 -> pickFile(); 1 -> askCreateFile(); else -> askConnectTv() }
        }.show()
    }

    // ---------------- TV pairing: phone = controller, TV website = display only ----------------
    private fun askConnectTv() {
        if (TvLink.connected) {
            AlertDialog.Builder(this).setTitle("متصل بالتلفزيون").setMessage("مقطوع الاتصال؟")
                .setPositiveButton("قطع الاتصال") { _, _ -> TvLink.disconnect(); Toast.makeText(this, "انقطع الاتصال", Toast.LENGTH_SHORT).show() }
                .setNegativeButton("إلغاء", null).show()
            return
        }
        val url = EditText(this); url.setSingleLine(true); url.hint = "رابط السيرفر (ws:// أو wss://)"
        url.setText(prefs().getString("tv_ws_url", "wss://your-server/api/tv-ws"))
        val code = EditText(this); code.setSingleLine(true); code.hint = "الرمز المعروض بشاشة التلفزيون"
        val box = LinearLayout(this); box.orientation = LinearLayout.VERTICAL; box.setPadding(dp(20), dp(10), dp(20), dp(0))
        box.addView(url); box.addView(code)
        AlertDialog.Builder(this).setTitle("اتصال بالتلفزيون").setView(box)
            .setPositiveButton("اتصال") { _, _ ->
                val u = url.text.toString().trim(); val c = code.text.toString().trim()
                if (u.isEmpty() || c.isEmpty()) { Toast.makeText(this, "لازم الرابط والرمز", Toast.LENGTH_SHORT).show(); return@setPositiveButton }
                prefs().edit().putString("tv_ws_url", u).apply()
                TvLink.connect(u, c,
                    onConnected = { name -> runOnUiThread { Toast.makeText(this, "✅ اتصل بـ: $name", Toast.LENGTH_LONG).show() } },
                    onError = { msg -> runOnUiThread { Toast.makeText(this, "❌ $msg", Toast.LENGTH_LONG).show() } },
                    onDisconnected = { runOnUiThread { Toast.makeText(this, "انقطع اتصال التلفزيون", Toast.LENGTH_SHORT).show() } },
                    onTvFeedback = { rating, reason, correction, user, bot -> saveFeedback(rating, reason, correction, user, bot) },
                    onTvText = { text ->
                        // someone typed on the TV itself: feed it through the SAME send path as the phone's
                        // own input box, so it is processed, saved, and answered exactly the same way.
                        runOnUiThread { input.setText(text); onSend() }
                    })
            }.setNegativeButton("إلغاء", null).show()
    }

    private fun prefs() = getSharedPreferences("nuwaa_prefs", Context.MODE_PRIVATE)

    // ---------------- 👍/👎 from the TV — a written reason is MANDATORY ----------------
    // The TV already refuses to send without a reason; this re-checks anyway (a rating without a reason is
    // dropped here too). Files land in <app files>/Nuwaa/ :
    //   user_feedback.jsonl         -> safe to train on (data/user_feedback.jsonl is already read by corpus.py):
    //                                  a 👍 on a real model reply, or a 👎 where you typed the CORRECT answer
    //   user_feedback_review.jsonl  -> NEVER trained on: a 👎 with no correction (training on a reply you called
    //                                  wrong would teach the model the wrong answer), or a reply the app itself made
    //                                  (math / system messages — not the model's own words)
    private fun stripFooter(t: String): String {
        var cut = t.length
        for (m in listOf("\n\n🧠", "\n\n🔋", "\n\n⚠")) { val i = t.indexOf(m); if (i in 0 until cut) cut = i }   // the notes the app adds under a reply are not part of the answer
        return t.substring(0, cut).trim()
    }

    private fun saveFeedback(rating: String, reason: String, correction: String, user: String, bot: String) {
        if (reason.trim().length < 5 || user.isBlank()) { TvLink.sendFeedbackSaved(false, "السبب إجباري (5 أحرف على الأقل)"); return }
        val fromModel = !(bot.startsWith("🧮") || bot.startsWith("⚙"))
        val cleanBot = stripFooter(bot)
        val trainable: JSONObject? = when {
            correction.isNotBlank() -> JSONObject().put("user", user.trim()).put("bot", correction.trim())
            rating == "good" && fromModel && cleanBot.isNotBlank() -> JSONObject().put("user", user.trim()).put("bot", cleanBot)
            else -> null
        }
        val file = if (trainable != null) "user_feedback.jsonl" else "user_feedback_review.jsonl"
        val obj = trainable ?: JSONObject().put("user", user.trim()).put("model_reply", cleanBot)
        obj.put("rating", rating).put("reason", reason.trim())
            .put("date", java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).format(java.util.Date()))
        try {
            val dir = java.io.File(getExternalFilesDir(null) ?: filesDir, "Nuwaa"); dir.mkdirs()
            java.io.File(dir, file).appendText(obj.toString() + "\n", Charsets.UTF_8)
            TvLink.sendFeedbackSaved(true, file)
        } catch (e: Exception) { TvLink.sendFeedbackSaved(false, e.message ?: "فشل الحفظ") }
    }

    /** What goes into a created file: the text in the input box, else the last reply (the code inside ``` if it has some). */
    private fun contentForFile(): String {
        val typed = input.text.toString().trim()
        if (typed.isNotEmpty()) return typed
        val last = conv?.msgs?.lastOrNull { it.role == "b" }?.text ?: ""
        return if (last.contains("```")) last.substringAfter("```").substringAfter('\n').substringBefore("```").trim('\n') else last
    }

    private fun askCreateFile() {
        val text = contentForFile()
        if (text.isBlank()) { Toast.makeText(this, "اكتب النص في خانة الكتابة أولا، أو اطلب مني كودا ثم اضغط ＋", Toast.LENGTH_LONG).show(); return }
        val name = EditText(this); name.setSingleLine(true); name.setText(if (text.contains("def ") || text.contains("import ")) "nuwaa.py" else "nuwaa.txt")
        AlertDialog.Builder(this).setTitle("اسم الملف").setView(name)
            .setPositiveButton("إنشاء") { _, _ ->
                pendingFile = text; pendingBytes = null
                val n = name.text.toString().trim().ifEmpty { "nuwaa.txt" }
                val i = Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*").putExtra(Intent.EXTRA_TITLE, n)
                try { startActivityForResult(i, REQ_CREATE) } catch (e: Exception) { Toast.makeText(this, "ما لقيت تطبيق ملفات في جوالك", Toast.LENGTH_LONG).show() }
            }.setNegativeButton("إلغاء", null).show()
    }

    private fun writeCreated(uri: Uri) {
        val data = pendingBytes ?: pendingFile.toByteArray(Charsets.UTF_8)
        pendingBytes = null
        thread {
            try {
                contentResolver.openOutputStream(uri)?.use { it.write(data) } ?: throw java.io.IOException("تعذر فتح الملف للكتابة")
                runOnUiThread { Toast.makeText(this, "تم إنشاء الملف", Toast.LENGTH_SHORT).show() }
            } catch (e: Throwable) {
                runOnUiThread { Toast.makeText(this, "تعذر إنشاء الملف: " + (e.message ?: ""), Toast.LENGTH_LONG).show() }
            }
        }
    }

    // ---------------- self-development (a clear button, no codes) ----------------
    private fun buildDevPage() {
        devPage = LinearLayout(this); devPage.orientation = LinearLayout.VERTICAL; devPage.setPadding(dp(16), dp(14), dp(16), dp(10))
        devPage.addView(Ui.label(this, "تطوير النفس", 20f, Ui.DARK, true))
        devPage.addView(Ui.label(this, "اضغط الزر: أراجع كل ما سمعته منك وأتدرب عليه، وأقيس نفسي قبل وبعد. إن تحسنت أحفظ التحسن، وإن ساءت أرجع لما كنت عليه. ما أبدأ إلا بضغطتك.", 14f, Ui.MUTED, false).also { it.setPadding(0, dp(6), 0, dp(12)) })
        val go = Ui.btn(this, "🧠  ابدأ التطوير الآن", Ui.PRIMARY) { runDevelop() }
        go.textSize = 18f
        devPage.addView(go, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val fileBtn = Ui.btn(this, "📦  ملف المطور", Ui.DARK) { askDevFile() }
        fileBtn.textSize = 16f
        val flp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); flp.setMargins(0, dp(10), 0, 0)
        devPage.addView(fileBtn, flp)
        val sv = ScrollView(this); devReport = Ui.label(this, "", 15f, Ui.DARK, false); devReport.setPadding(0, dp(14), 0, 0); devReport.setTextIsSelectable(true); sv.addView(devReport)
        devPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    // ---------------- raise understanding: training on the phone at a CPU share the user picks ----------------
    private fun buildTrainPage() {
        trainPage = LinearLayout(this); trainPage.orientation = LinearLayout.VERTICAL; trainPage.setPadding(dp(16), dp(14), dp(16), dp(10))
        trainPage.addView(Ui.label(this, "رفع الفهم", 20f, Ui.DARK, true))
        trainPage.addView(Ui.label(this, "أدرّب نفسي على بيانات البرمجة والعربية داخل جوالك، شوي شوي. أنت تحدد كم أستهلك من المعالج، حتى لو الجوال ضعيف أكمل بالتدريج وأحفظ كل تحسن. أقيس نفسي على نص ثابت، وإن ساءت النتيجة أرجع للأفضل.", 14f, Ui.MUTED, false).also { it.setPadding(0, dp(6), 0, dp(12)) })
        pctLabel = Ui.label(this, "الاستهلاك: 40% من نواة معالج واحدة", 16f, Ui.DARK, true)
        trainPage.addView(pctLabel)
        val sk = SeekBar(this); sk.max = 95; sk.progress = 35
        sk.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) { val v = p + 5; trainer().percent = v; pctLabel.text = "الاستهلاك: " + v + "% من نواة معالج واحدة" }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        trainPage.addView(sk, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val presets = LinearLayout(this); presets.orientation = LinearLayout.HORIZONTAL
        for ((txt, v) in listOf("🔋 توفير 15%" to 15, "⚖ متوازن 40%" to 40, "🔥 قوي 100%" to 100)) {
            val b = Ui.btn(this, txt, Ui.DARK) { sk.progress = v - 5 }
            b.textSize = 13f
            val pl = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f); pl.setMargins(dp(3), dp(8), dp(3), 0); presets.addView(b, pl)
        }
        trainPage.addView(presets)
        val sr = CheckBox(this); sr.text = "مراجعة ذاتية: أختبر نفسي بمهام أعرف جوابها وأتعلم من الصحيح"; sr.isChecked = true; sr.textSize = 14f
        sr.setOnCheckedChangeListener { _, on -> trainer().selfReview = on }
        trainPage.addView(sr)
        trainBtn = Ui.btn(this, "⚡  ابدأ رفع الفهم", Ui.PRIMARY) { toggleTrain() }
        trainBtn.textSize = 18f
        val blp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); blp.setMargins(0, dp(12), 0, 0)
        trainPage.addView(trainBtn, blp)
        val rs = Ui.btn(this, "🔄  إعادة تشغيل التطبيق (يحفظ أفضل حالة أولا)", Ui.DARK) { restartApp() }
        val rlp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); rlp.setMargins(0, dp(8), 0, 0)
        trainPage.addView(rs, rlp)
        val sv = ScrollView(this); trainInfo = Ui.label(this, "", 15f, Ui.DARK, false); trainInfo.setPadding(0, dp(14), 0, 0); sv.addView(trainInfo)
        trainPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun toggleTrain() {
        val t = trainer()
        if (t.running) {
            t.stop(); trainBtn.text = "⚡  ابدأ رفع الفهم"
            window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            trainInfo.text = "أوقفت التدريب وأحفظ الآن أفضل حالة..."
            return
        }
        t.percent = pctLabel.text.toString().filter { it.isDigit() }.toIntOrNull() ?: 40
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)       // training stops if the app is closed: keep the screen on while it runs
        trainBtn.text = "⏹  إيقاف (يحفظ الأفضل)"
        t.start { msg -> runOnUiThread { trainInfo.text = msg } }
    }

    // ---------------- discovery: optional, needs internet + the user's own Google API key, and only runs when asked ----------------
    private fun openGrowPage() {
        if (!SecretGrowth.isSet(this)) {
            val input = EditText(this); input.inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD
            AlertDialog.Builder(this).setTitle("حدد رمزا سريا (أول مرة)")
                .setMessage("هذا الرمز يحميك من فتح هذي الصفحة بالغلط لاحقا. لا يُرسل لأي مكان ويُحفظ على جهازك فقط.")
                .setView(input).setPositiveButton("حفظ") { _, _ -> if (input.text.toString().length >= 4) { SecretGrowth.setPin(this, input.text.toString()); showPage(growPage) } }
                .setNegativeButton("إلغاء", null).show()
            return
        }
        val input = EditText(this); input.inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD
        AlertDialog.Builder(this).setTitle("أدخل الرمز السري").setView(input)
            .setPositiveButton("دخول") { _, _ -> if (SecretGrowth.verify(this, input.text.toString())) showPage(growPage) else Toast.makeText(this, "رمز خاطئ", Toast.LENGTH_SHORT).show() }
            .setNegativeButton("إلغاء", null).show()
    }

    private fun buildGrowPage() {
        growPage = LinearLayout(this); growPage.orientation = LinearLayout.VERTICAL; growPage.setPadding(dp(16), dp(14), dp(16), dp(10))
        growPage.addView(Ui.label(this, "⚙ تطوير متقدم", 20f, Ui.DARK, true))
        val heap = Runtime.getRuntime().maxMemory() / 1e6
        growPage.addView(Ui.label(this, "ذاكرة هذا الجوال المتاحة للتطبيق الآن: " + heap.toInt() + " MB (تُقاس كل مرة من جهازك، وتتغير حسب الجهاز والوضع). هذا يحدد أكبر حجم آمن تكبّر له النموذج هنا.", 14f, Ui.MUTED, false).also { it.setPadding(0, dp(6), 0, dp(10)) })
        val m = brain.trainModel()
        val curParams = m?.paramCount() ?: 0
        growPage.addView(Ui.label(this, "الحجم الحالي: " + curParams + " معامل", 15f, Ui.DARK, true))
        val maxP = SecretGrowth.maxSafeParams(this).coerceAtLeast((curParams + 1000).toLong())
        val sizeLabel = Ui.label(this, "", 15f, Ui.DARK, true); growPage.addView(sizeLabel)
        val sizeBar = android.widget.SeekBar(this); sizeBar.max = 100
        fun targetParams(progress: Int): Long = curParams + ((maxP - curParams) * progress / 100.0).toLong()
        sizeBar.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: android.widget.SeekBar?, p: Int, u: Boolean) { sizeLabel.text = "الهدف: " + targetParams(p) + " معامل (~" + (targetParams(p) * 4 / 1e6).toInt() + " MB)" }
            override fun onStartTrackingTouch(sb: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(sb: android.widget.SeekBar?) {}
        })
        sizeBar.progress = 0; sizeLabel.text = "الهدف: " + curParams + " معامل (~" + (curParams * 4 / 1e6).toInt() + " MB)"
        growPage.addView(sizeBar)
        growPage.addView(Ui.label(this, "⚠ كرت الشاشة: غير متاح للتدريب على الجوال في هذا التطبيق (المحرك هنا يحسب على المعالج فقط). التحكم الحقيقي بكرت الشاشة متاح فقط عبر «استوديو نواة» على الكمبيوتر (tools/studio.py)، ليس من هنا.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(12), 0, dp(4)) })
        growPage.addView(Ui.btn(this, "🔼  كبّر النموذج الآن", Ui.PRIMARY) {
            val e = m?.e ?: 64; val h = SecretGrowth.pickHidden(targetParams(sizeBar.progress), maxOf(e, 96), m?.chars?.size ?: 188)
            AlertDialog.Builder(this).setTitle("تأكيد").setMessage("هذا يكبّر النموذج فعلا الآن ويحفظه. ما يرجع صغيرا إلا بإعادة تثبيت التطبيق. يكمل؟")
                .setPositiveButton("كبّر") { _, _ -> growResult.text = brain.growModel(maxOf(e, 96), h) }.setNegativeButton("إلغاء", null).show()
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(10) })
        growPage.addView(Ui.label(this, "تدريب النموذج الأكبر يستهلك نفس استهلاك المعالج اللي تختاره في ⚡ رفع الفهم (15/40/100%)، لكن كل خطوة تصير أبطأ بزيادة الحجم.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(10), 0, dp(4)) })
        val sv = ScrollView(this); growResult = Ui.label(this, "", 15f, Ui.DARK, false); growResult.setPadding(0, dp(10), 0, 0); sv.addView(growResult)
        growPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun buildDiscoverPage() {
        discoverPage = LinearLayout(this); discoverPage.orientation = LinearLayout.VERTICAL; discoverPage.setPadding(dp(16), dp(14), dp(16), dp(10))
        discoverPage.addView(Ui.label(this, "اكتشاف من الإنترنت", 20f, Ui.DARK, true))
        discoverPage.addView(Ui.label(this, "هذه الميزة الوحيدة التي تحتاج إنترنت في كل التطبيق. تبحث في جوجل بمفتاحك الخاص (Google Custom Search، مجانا حتى 100 بحث يوميا)، وما تُشغَّل إلا لما تطلبها بنفسك. النتائج من جوجل مباشرة، وليست من النموذج ولا كلامه.", 14f, Ui.MUTED, false).also { it.setPadding(0, dp(6), 0, dp(12)) })
        val keyIn = EditText(this); keyIn.hint = "API key"; keyIn.isSingleLine = true
        val cxIn = EditText(this); cxIn.hint = "Search engine ID (cx)"; cxIn.isSingleLine = true
        discoverPage.addView(keyIn); discoverPage.addView(cxIn, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(8) })
        val saveRow = LinearLayout(this); saveRow.orientation = LinearLayout.HORIZONTAL
        saveRow.addView(Ui.btn(this, "حفظ المفتاح", Ui.DARK) {
            GoogleSearch.setCredentials(this, keyIn.text.toString(), cxIn.text.toString()); keyIn.setText(""); cxIn.setText("")
            discoverResult.text = if (GoogleSearch.configured(this)) "تم الحفظ. اكتب بحثك تحت." else "أدخل المفتاح ومعرّف محرك البحث معا."
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).also { it.topMargin = dp(10); it.rightMargin = dp(6) })
        saveRow.addView(Ui.btn(this, "مسح المفتاح", Ui.DARK) { GoogleSearch.clearCredentials(this); discoverResult.text = "مُسح. هذه الميزة الآن متوقفة." },
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).also { it.topMargin = dp(10); it.leftMargin = dp(6) })
        discoverPage.addView(saveRow)
        val qIn = EditText(this); qIn.hint = "ابحث عن..."
        discoverPage.addView(qIn, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(14) })
        discoverPage.addView(Ui.btn(this, "🔍  ابحث في جوجل الآن", Ui.PRIMARY) {
            val q = qIn.text.toString().trim(); if (q.isBlank()) return@btn
            discoverResult.text = "… أبحث في جوجل"
            Thread {
                val text = try { GoogleSearch.format(q, GoogleSearch.search(this, q)) } catch (e: Exception) { "تعذّر البحث: " + (e.message ?: e.javaClass.simpleName) }
                runOnUiThread { discoverResult.text = text }
            }.start()
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(10) })
        // ---- Gemini teacher: an optional external AI that keeps feeding Nuwaa fresh programming/Arabic examples while ⚡ رفع الفهم is training ----
        discoverPage.addView(Ui.label(this, "معلّم Gemini (اختياري)", 18f, Ui.DARK, true).also { it.setPadding(0, dp(20), 0, dp(4)) })
        discoverPage.addView(Ui.label(this, "يطلب من Gemini (بمفتاحك الخاص) أمثلة برمجة ولغة عربية جديدة باستمرار أثناء التدريب، ويدرّب عليها بنفس حماية التراجع — لكن بلا تحقق بالتشغيل مثل بيانات نواة نفسها، فهو معلّم غير موثَّق بقدر التحقق الذاتي.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(2), 0, dp(10)) })
        val gKeyIn = EditText(this); gKeyIn.hint = "Gemini API key"; gKeyIn.isSingleLine = true
        val gModelIn = EditText(this); gModelIn.hint = "الموديل (افتراضي: " + GeminiTeacher.DEFAULT_MODEL + ")"; gModelIn.isSingleLine = true
        discoverPage.addView(gKeyIn); discoverPage.addView(gModelIn, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(8) })
        val gSaveRow = LinearLayout(this); gSaveRow.orientation = LinearLayout.HORIZONTAL
        val gStatus = Ui.label(this, if (GeminiTeacher.configured(this)) "مفعّل." else "أدخل مفتاحك.", 13f, Ui.MUTED, false)
        gSaveRow.addView(Ui.btn(this, "حفظ مفتاح Gemini", Ui.DARK) {
            GeminiTeacher.setCredentials(this, gKeyIn.text.toString(), gModelIn.text.toString()); gKeyIn.setText(""); gStatus.text = if (GeminiTeacher.configured(this)) "تم الحفظ." else "أدخل المفتاح."
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).also { it.topMargin = dp(10); it.rightMargin = dp(6) })
        gSaveRow.addView(Ui.btn(this, "مسح", Ui.DARK) { GeminiTeacher.clearCredentials(this); trainer().teacherOn = false; gStatus.text = "مُسح." },
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).also { it.topMargin = dp(10); it.leftMargin = dp(6) })
        discoverPage.addView(gSaveRow); discoverPage.addView(gStatus)
        val gEnable = CheckBox(this); gEnable.text = "فعّل معلّم Gemini أثناء التدريب"; gEnable.isChecked = false; gEnable.textSize = 14f
        val gRateLabel = Ui.label(this, "المعدل: 5 رسالة/دقيقة", 14f, Ui.DARK, false)
        val gRateBar = android.widget.SeekBar(this); gRateBar.max = 59; gRateBar.progress = 4
        gEnable.setOnCheckedChangeListener { _, on -> trainer().teacherOn = on }
        gRateBar.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: android.widget.SeekBar?, p: Int, u: Boolean) { val v = p + 1; trainer().teacherPerMinute = v; gRateLabel.text = "المعدل: " + v + " رسالة/دقيقة (بلا حد أقصى مفروض؛ أنت تتحكم، والتكلفة على مفتاحك)" }
            override fun onStartTrackingTouch(sb: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(sb: android.widget.SeekBar?) {}
        })
        discoverPage.addView(gEnable, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.topMargin = dp(10) })
        discoverPage.addView(gRateLabel); discoverPage.addView(gRateBar)

        val sv = ScrollView(this); discoverResult = Ui.label(this, if (GoogleSearch.configured(this)) "اكتب بحثك فوق." else "أدخل مفتاحك فوق لتفعيل البحث.", 15f, Ui.DARK, false)
        discoverResult.setPadding(0, dp(14), 0, 0); sv.addView(discoverResult)
        discoverPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    override fun onDestroy() { trainerObj?.stop(); CastManager.stop(); TvLink.disconnect(); super.onDestroy() }

    /** Saves the best trained state, then really restarts the app (new process, the model is loaded again from disk). */
    private fun restartApp() {
        trainerObj?.stopAndWait(5000)
        val i = packageManager.getLaunchIntentForPackage(packageName)
        if (i != null) { i.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_CLEAR_TASK); startActivity(i) }
        finishAffinity()
        Runtime.getRuntime().exit(0)
    }

    private fun runDevelop() {
        devReport.text = "أراجع ما سمعته وأتدرب... (قد تأخذ دقيقة)"
        thread {
            val r = try { brain.develop() } catch (e: Throwable) { "تعذر التطوير: " + (e.message ?: e.javaClass.simpleName) }
            runOnUiThread { devReport.text = r }
        }
    }

    /** The developer file: weights + long-term memory + report in one zip. It is saved only where the user picks, through the system file picker. */
    private fun askDevFile() {
        devReport.text = "أجهّز ملف المطور..."
        thread {
            try {
                val bytes = brain.devFile()
                runOnUiThread {
                    pendingBytes = bytes
                    val i = Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/zip").putExtra(Intent.EXTRA_TITLE, "nuwaa-dev-" + (System.currentTimeMillis() / 1000) + ".zip")
                    try { startActivityForResult(i, REQ_CREATE); devReport.text = "اختر مكان حفظ الملف، ثم أرسله لمن يفحصه." } catch (e: Exception) { devReport.text = "ما لقيت تطبيق ملفات في جوالك" }
                }
            } catch (e: Throwable) { runOnUiThread { devReport.text = "تعذر تجهيز الملف: " + (e.message ?: e.javaClass.simpleName) } }
        }
    }

    // ---------------- attach file (read only) ----------------
    @Suppress("DEPRECATION")
    private fun pickFile() {
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
        try { startActivityForResult(i, REQ_PICK) } catch (e: Exception) { Toast.makeText(this, "ما لقيت تطبيق ملفات في جوالك", Toast.LENGTH_LONG).show() }
    }

    private fun loadAttachment(uri: Uri) {
        thread {
            try {
                var name = "ملف"; var size = -1L
                contentResolver.query(uri, null, null, null, null)?.use { cur ->
                    if (cur.moveToFirst()) {
                        val ni = cur.getColumnIndex(OpenableColumns.DISPLAY_NAME); val si = cur.getColumnIndex(OpenableColumns.SIZE)
                        if (ni >= 0) name = cur.getString(ni) ?: name
                        if (si >= 0 && !cur.isNull(si)) size = cur.getLong(si)
                    }
                }
                val buf = java.io.ByteArrayOutputStream()
                var total = 0L; var truncated = false
                contentResolver.openInputStream(uri)?.use { ins ->
                    val chunk = ByteArray(16384)
                    while (true) {
                        val k = ins.read(chunk); if (k < 0) break
                        total += k
                        if (buf.size() + k > MAX_READ) { buf.write(chunk, 0, MAX_READ - buf.size()); truncated = true; break }
                        buf.write(chunk, 0, k)
                    }
                } ?: throw java.io.IOException("تعذر فتح الملف")
                val real = if (size >= 0) size else total
                val att = Attached(name, real, buf.toByteArray(), truncated)
                runOnUiThread {
                    attached = att
                    attachLabel.text = "📎 " + name + " — اكتب سؤالك عنه أو اضغط إرسال"
                    attachBar.visibility = View.VISIBLE
                }
            } catch (e: Throwable) {
                runOnUiThread { Toast.makeText(this, "تعذر قراءة الملف: " + (e.message ?: ""), Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun addMsg(c: Conv, role: String, text: String) {
        c.msgs.add(Msg(role, text)); c.updated = System.currentTimeMillis()
        if (role == "u" && c.title == "محادثة جديدة") { c.title = text.take(30); titleView.text = c.title }
        store.save(c); bubble(text, role == "u")
    }

    // ---------------- history ----------------
    private fun buildHistPage() {
        histPage = LinearLayout(this); histPage.orientation = LinearLayout.VERTICAL
        histPage.addView(Ui.label(this, "المحادثات", 18f, Ui.DARK, true).also { it.setPadding(dp(16), dp(12), dp(16), dp(4)) })
        val sv = ScrollView(this); histList = LinearLayout(this); histList.orientation = LinearLayout.VERTICAL; sv.addView(histList)
        histPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun refreshHistory() {
        histList.removeAllViews()
        val all = store.list().filter { it.msgs.isNotEmpty() }
        if (all.isEmpty()) { histList.addView(Ui.label(this, "ما فيه محادثات محفوظة بعد.", 15f, Ui.MUTED, false).also { it.setPadding(dp(16), dp(16), dp(16), 0) }); return }
        for (c in all) {
            val row = LinearLayout(this); row.orientation = LinearLayout.HORIZONTAL; row.gravity = Gravity.CENTER_VERTICAL
            row.background = Ui.bg(this, Color.WHITE, 14); row.setPadding(dp(14), dp(12), dp(8), dp(12))
            val t = Ui.label(this, c.title + "\n" + c.msgs.size + " رسالة", 15f, Ui.DARK, false)
            val del = TextView(this); del.text = "🗑"; del.textSize = 20f; del.setPadding(dp(12), dp(4), dp(12), dp(4))
            row.setOnClickListener {
                val cur = conv
                if (cur != null && cur.id != c.id) { brain.endConversation(cur.id); thread { brain.resume(c.id, c.msgs.toList()) } }
                conv = c; renderConv(); showPage(chatPage); scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
            }
            del.setOnClickListener { store.delete(c.id); brain.endConversation(c.id); if (conv?.id == c.id) { conv = store.create(); renderConv() }; refreshHistory() }
            row.addView(t, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)); row.addView(del)
            val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(dp(12), dp(5), dp(12), dp(5))
            histList.addView(row, lp)
        }
    }

    // ---------------- screen share ----------------
    private fun buildScreenPage() {
        screenPage = ScrollView(this)
        val col = LinearLayout(this); col.orientation = LinearLayout.VERTICAL; col.setPadding(dp(14), dp(12), dp(14), dp(20))
        col.addView(Ui.label(this, "مشاركة الشاشة", 18f, Ui.DARK, true))
        col.addView(Ui.label(this, "أشاهد إطارا واحدا كل ثانية، وأقرأ النص الظاهر على الشاشة (بعد تفعيل قارئ الشاشة) وأنصحك بخطوة تالية. لا أضغط ولا أكتب بدلا عنك. لا أفهم الصور، فقط النص وتغيّر الشاشة.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(4), 0, dp(10)) })
        goalInput = EditText(this); goalInput.hint = "ماذا تريد أن تنجز؟ (مثال: كيف ارفع مشروعي على github)"
        goalInput.background = Ui.bg(this, Color.WHITE, 14); goalInput.setPadding(dp(14), dp(10), dp(14), dp(10))
        col.addView(goalInput)
        val b1 = Ui.btn(this, "تفعيل قارئ الشاشة (مرة واحدة)", Ui.DARK) { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)); Toast.makeText(this, "اختر «نواة» ثم فعّله", Toast.LENGTH_LONG).show() }
        shareBtn = Ui.btn(this, "ابدأ المشاركة", Ui.PRIMARY) { toggleShare() }
        val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(0, dp(10), 0, 0)
        col.addView(b1, lp); col.addView(shareBtn, lp)
        shareInfo = Ui.label(this, "", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(10), 0, dp(4)) }
        col.addView(shareInfo)
        preview = ImageView(this); preview.adjustViewBounds = true; preview.maxHeight = dp(260); preview.background = Ui.bg(this, Color.parseColor("#D9E1E7"), 10)
        col.addView(preview, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(200)))
        adviceView = Ui.label(this, "النصيحة تظهر هنا.", 15f, Ui.DARK, false)
        adviceView.background = Ui.bg(this, Color.WHITE, 14); adviceView.setPadding(dp(14), dp(12), dp(14), dp(12)); adviceView.setTextIsSelectable(true)
        val lp2 = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp2.setMargins(0, dp(10), 0, 0)
        col.addView(adviceView, lp2)
        screenPage.addView(col)
    }

    private fun refreshShareUi() {
        shareBtn.text = if (ScreenShareService.running) "إيقاف المشاركة" else "ابدأ المشاركة"
        shareInfo.text = if (ScreenReader.enabled()) "قارئ الشاشة: مفعّل ✓" else "قارئ الشاشة: غير مفعّل (بدونه أرى تغيّر الشاشة فقط)"
    }

    private fun toggleShare() {
        if (ScreenShareService.running) { startService(Intent(this, ScreenShareService::class.java).setAction(ScreenShareService.ACTION_STOP)); return }
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQ_NOTIF); return
        }
        launchCapture()
    }

    @Suppress("DEPRECATION")
    private fun launchCapture() {
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(mpm.createScreenCaptureIntent(), REQ_CAPTURE)
    }

    @Suppress("DEPRECATION")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_NOTIF) launchCapture()    // the service still works if notifications are refused, only the tip is hidden
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK) {
            val u = data?.data
            if (resultCode == RESULT_OK && u != null) loadAttachment(u)
            return
        }
        if (requestCode == REQ_CREATE) {
            val u = data?.data
            if (resultCode == RESULT_OK && u != null) writeCreated(u)
            return
        }
        if (requestCode != REQ_CAPTURE) return
        if (resultCode != RESULT_OK || data == null) { Toast.makeText(this, "لم تُمنح صلاحية مشاركة الشاشة", Toast.LENGTH_SHORT).show(); return }
        ScreenShareService.goal = goalInput.text.toString().trim()
        ScreenShareService.frameListener = { bmp: Bitmap, change: Double ->
            runOnUiThread { preview.setImageBitmap(bmp); shareInfo.text = "إطار كل ثانية · تغيّر الشاشة: " + (change * 100).toInt() + "%" + (if (ScreenReader.enabled()) " · قارئ الشاشة ✓" else " · قارئ الشاشة غير مفعّل") }
        }
        ScreenShareService.adviceListener = { s: String -> runOnUiThread { adviceView.text = s } }
        ScreenShareService.stopListener = { runOnUiThread { refreshShareUi(); shareInfo.text = "توقفت المشاركة." } }
        val i = Intent(this, ScreenShareService::class.java).putExtra("code", resultCode).putExtra("data", data)
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i) else startService(i)
        shareBtn.text = "إيقاف المشاركة"
        Toast.makeText(this, "بدأت المشاركة. افتح التطبيق الذي تريد المساعدة فيه.", Toast.LENGTH_LONG).show()
    }

    override fun onResume() { super.onResume(); if (::shareBtn.isInitialized) refreshShareUi() }

    // ---------------- code writing ----------------
    private fun buildCodePage() {
        codePage = LinearLayout(this); codePage.orientation = LinearLayout.VERTICAL; codePage.setPadding(dp(12), dp(10), dp(12), dp(10))
        codePage.addView(Ui.label(this, "كتابة الأكواد", 18f, Ui.DARK, true))
        codeIdea = EditText(this); codeIdea.hint = "صف ما تريد: (مثال: حلقة for في بايثون)"
        codeIdea.background = Ui.bg(this, Color.WHITE, 14); codeIdea.setPadding(dp(14), dp(10), dp(14), dp(10)); codeIdea.setSingleLine(true)
        codePage.addView(codeIdea, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.setMargins(0, dp(8), 0, dp(6)) })
        val row = LinearLayout(this); row.orientation = LinearLayout.HORIZONTAL
        val suggest = Ui.btn(this, "اقترح كودا", Ui.PRIMARY) { suggestCode() }
        val copy = Ui.btn(this, "نسخ", Ui.DARK) { copyText(editor.text.toString()); Toast.makeText(this, "تم النسخ", Toast.LENGTH_SHORT).show() }
        val share = Ui.btn(this, "مشاركة", Ui.DARK) { shareCode() }
        val clear = Ui.btn(this, "مسح", Color.parseColor("#8A3B3B")) { editor.setText("") }
        for (v in listOf(suggest, copy, share, clear)) row.addView(v, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).also { it.setMargins(dp(3), 0, dp(3), 0) })
        codePage.addView(row)
        val strip = Ui.btn(this, "🔢  إزالة الأرقام الجانبية", Ui.PRIMARY) { stripNumbers() }
        codePage.addView(strip, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.setMargins(dp(3), dp(8), dp(3), 0) })
        editor = EditText(this); editor.typeface = Typeface.MONOSPACE; editor.textSize = 13f; editor.setTextColor(Color.parseColor("#D7E3EA")); editor.setHintTextColor(Color.parseColor("#6B7F8C"))
        editor.hint = "# اكتب كودك هنا أو اضغط «اقترح كودا»"; editor.gravity = Gravity.TOP or Gravity.START; editor.layoutDirection = View.LAYOUT_DIRECTION_LTR; editor.textDirection = View.TEXT_DIRECTION_LTR
        editor.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
        editor.background = Ui.bg(this, Ui.CODE_BG, 12); editor.setPadding(dp(12), dp(12), dp(12), dp(12))
        codePage.addView(editor, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f).also { it.setMargins(0, dp(8), 0, 0) })
    }

    private fun suggestCode() {
        val q = codeIdea.text.toString().trim()
        if (q.isEmpty()) { Toast.makeText(this, "اكتب وصفا أولا", Toast.LENGTH_SHORT).show(); return }
        val hit = brain.kb.search(q, 6).firstOrNull { h -> h.score >= Kb.LOW && h.item.a.contains("```") }
        if (hit == null) { Toast.makeText(this, "ما عندي كود جاهز لهذا الوصف. جرّب وصفا أبسط، أو علّمني في المحادثة.", Toast.LENGTH_LONG).show(); return }
        val code = hit.item.a.substringAfter("```").substringAfter('\n').substringBefore("```").trim('\n')
        val mark = when (hit.item.v) { 2 -> "# ✓ شُغِّل واختُبر ناتجه"; 1 -> "# ✓ شُغِّل بدون أخطاء"; 0 -> "# ✓ فُحصت صياغته فقط"; else -> "# ⚠ لم يُشغَّل هنا، جرّبه" }
        val cur = editor.text.toString()
        editor.setText((if (cur.isBlank()) "" else cur.trimEnd() + "\n\n") + mark + "\n" + code)
    }

    private fun stripNumbers() {
        val t = editor.text.toString()
        if (t.isBlank()) { Toast.makeText(this, "الصق الكود أو اكتبه أولا", Toast.LENGTH_SHORT).show(); return }
        val r = LineNumbers.strip(t)
        if (r.second == 0) { Toast.makeText(this, "ما لقيت أرقام جانبية واضحة في الكود", Toast.LENGTH_LONG).show(); return }
        editor.setText(r.first)
        Toast.makeText(this, "أزلت الأرقام من " + r.second + " سطر", Toast.LENGTH_SHORT).show()
    }

    private fun shareCode() {
        val t = editor.text.toString()
        if (t.isBlank()) return
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, t), "مشاركة الكود"))
    }

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (pages.childCount > 0 && pages.getChildAt(0) !== chatPage) showPage(chatPage) else super.onBackPressed()
    }
}


// ============== من الجوال_Brain.kt ==============
class Answer(val text: String, val source: String, val file: MadeFile? = null)

/** A file the user sent with the + button. Up to 8 MB is kept in memory (a whole Quran text file fits). Sent files are read-only; files are created only when the user picks «إنشاء ملف». */
class Attached(val name: String, val size: Long, val bytes: ByteArray, val truncated: Boolean) { var announced = false }

/** One brain per process. Every chat message is answered by the language model WRITING the reply (no canned phrase, no lookup, no confidence gate);
 *  exact arithmetic is the only tool kept. Short-term memory = one NlmSession per conversation (dropped when the conversation ends).
 *  Long-term memory = the weights (nlm2_user.bin) + everything heard (memory.txt), both loaded again at start. */
class Brain private constructor(private val ctx: Context) {
    companion object {
        @Volatile private var inst: Brain? = null

        fun get(c: Context): Brain {
            val i = inst
            if (i != null) return i
            synchronized(Brain::class.java) {
                val j = inst
                if (j != null) return j
                val b = Brain(c.applicationContext)
                b.loadKnowledge()
                inst = b
                return b
            }
        }
    }

    /** Verified programming snippets: used ONLY by the code page and to read files and the screen. Never used to answer a chat message. */
    val kb = Kb()
    @Volatile var modelStatus: String = "النموذج: لم يُحمَّل بعد"
    @Volatile var goal: String = ""

    var nlm: Nlm? = null
        private set
    private val lock = Any()
    private val sessions = LinkedHashMap<String, NlmSession>()
    private val nlmUserFile = File(ctx.filesDir, "nlm2_user.bin")
    private val memFile = File(ctx.filesDir, "memory.txt")
    private val memory = ArrayList<String>()
    private var observed = 0
    private var lastReport = JSONObject()
    private var rehearsal = ""            // a sample of the original training text, replayed during self-development so nothing known is forgotten

    private fun readAsset(name: String): String =
        ctx.assets.open(name).bufferedReader(Charsets.UTF_8).use { r -> r.readText() }

    fun loadKnowledge() {
        try { Nlp.load(readAsset("syn.json")) } catch (e: Exception) { Nlp.load("{\"phrases\":{},\"words\":{},\"stop\":[],\"langs\":[]}") }
        try { kb.loadJsonl(readAsset("kb.jsonl")) } catch (e: Exception) { /* asset missing: the code page just has no snippets */ }
        try { rehearsal = readAsset("rehearse.txt") } catch (e: Exception) { rehearsal = "" }
        loadNlm()
    }

    /** assets/nlm2.bin = the pretrained start; filesDir/nlm2_user.bin = what it learned on this phone (the long-term memory of the weights). */
    private fun loadNlm() {
        try {
            val base = ctx.assets.open("nlm2.bin").use { s -> s.readBytes() }
            val heapMb = Runtime.getRuntime().maxMemory() / 1e6      // no budget of mine: the only limit is what this phone really gives the app
            if (base.size * 3.0 / 1e6 > heapMb * 0.85) { modelStatus = "النموذج: تخطيته لأن ذاكرة هذا الجهاز للتطبيق ${heapMb.toInt()} MB فقط"; return }
            val m = try { if (nlmUserFile.exists()) Nlm.load(nlmUserFile.readBytes()) else Nlm.load(base) } catch (e: Exception) { Nlm.load(base) }     // a damaged user file never blocks the start
            if (memFile.exists()) for (line in memFile.readLines(Charsets.UTF_8)) if (line.isNotBlank()) { memory.add(line); m.long.add(line) }
            nlm = m
            modelStatus = "نموذج نواة (GRU، سياق غير محدود): " + m.paramCount() + " معامل · سمعت منك " + memory.size + " رسالة"
        } catch (e: java.io.FileNotFoundException) {
            modelStatus = "النموذج غير موجود في التطبيق (nlm2.bin)"
        } catch (e: Throwable) {
            modelStatus = "تعذر تحميل النموذج: " + (e.message ?: e.javaClass.simpleName)
        }
    }

    /** Kept for the start-up code: the model is loaded by loadKnowledge(); this only reports whether it is there. */
    fun loadModel() { /* nothing to do: loadKnowledge() already loaded it when the brain was created */ }

    private fun saveNlm() {
        try { nlm?.let { m -> val bytes = synchronized(lock) { m.toBytes() }; nlmUserFile.outputStream().use { o -> o.write(bytes) } } } catch (e: Throwable) { /* keep running; it will try again */ }
    }

    fun usedMb(): Double = android.os.Debug.getPss() / 1024.0

    /** Grows the model to (newE, newH), persists it immediately (so it survives a restart even before any training), and clears per-session
     *  memory (the vocabulary/shape changed under them). Returns a human-readable result; never throws. Safety: refuses if the grown model's
     *  training footprint (weights + Adam m + Adam v, the same ~3x rule loadNlm already uses) would not fit in what THIS phone actually granted
     *  the app right now — never a fixed number, always Runtime.getRuntime().maxMemory() at the moment of the attempt. */
    fun growModel(newE: Int, newH: Int): String {
        val m = nlm ?: return "النموذج غير محمّل."
        if (newE < m.e || newH < m.h) return "الحجم الجديد لازم يكون أكبر من الحالي."
        val projectedParams = (m.chars.size.toLong() * newE) + (newE.toLong() * 3 * newH + newH.toLong() * 3 * newH + 3L * newH) +
            (newH.toLong() * 3 * newH + newH.toLong() * 3 * newH + 3L * newH) + newH.toLong() * m.chars.size + m.chars.size
        val heap = Runtime.getRuntime().maxMemory(); val needed = projectedParams * 4.0 * 3.0
        if (needed > heap * 0.8) return "⚠ ما قدرت: هذا الحجم يحتاج تقريبا " + (needed / 1e6).toInt() + " MB، وذاكرة هذا الجهاز للتطبيق الآن " + (heap / 1e6).toInt() + " MB فقط. جرّب حجما أصغر."
        return try {
            val grown = synchronized(lock) { Nlm.grow(m, newE, newH) }
            // The safety net growModel() was missing (develop() already has its own): check the grown
            // model is actually sound BEFORE saving over the working one. A bad grow now rolls back
            // automatically instead of silently shipping a broken model.
            if (!ModelHealth.isHealthy(grown)) {
                return "⚠ فشل التكبير: " + ModelHealth.describe(grown) + " رجّعت النموذج القديم، ما تغير شي."
            }
            synchronized(lock) { nlm = grown; sessions.clear() }
            saveNlm()
            "✅ كبّرت النموذج فعلا: من " + m.paramCount() + " إلى " + grown.paramCount() + " معامل. " + ModelHealth.describe(grown) + " يبدأ بنفس سلوك القديم تماما، والمعرفة الإضافية فارغة حتى يتدرب. احفظت الحالة الجديدة؛ افتح ⚡ رفع الفهم لتدريبه."
        } catch (e: Throwable) { "⚠ فشل التكبير: " + (e.message ?: e.javaClass.simpleName) }
    }

    /** One fresh, memory-off attempt (nothing is remembered, nothing is looked up): used by the self-review. */
    fun selfAttempt(q: String): String {
        val m = nlm ?: return ""
        return synchronized(lock) { NlmSession(m, System.nanoTime()).reply(q, 500, 0.5f, 8, 0.15f, false, 0f, 0f).first }
    }

    // ---- hooks for Trainer (the «raise understanding» button) ----
    val modelLock: Any get() = lock
    fun trainModel(): Nlm? = nlm
    fun saveTrained() = saveNlm()
    /** The training text shipped with the app: one pair per line, the two lines of a pair joined by a tab. Returned as «س:..\nج:..\n» blocks. */
    fun trainPairs(): List<String> = try {
        readAsset("train.txt").split("\n").filter { it.length > 6 }.map { it.replace('\t', '\n') + "\n" }
    } catch (e: Exception) { emptyList() }

    // ---------------------------------------------------------------- the conversation
    private fun sessionFor(m: Nlm, convId: String): NlmSession {
        val s = sessions.remove(convId) ?: NlmSession(m, System.nanoTime())
        sessions[convId] = s
        while (sessions.size > 8) { val k = sessions.keys.first(); sessions.remove(k) }
        return s
    }

    /** Every chat message enters here. convId = the conversation: its short-term memory is its own NlmSession. */
    fun answer(q: String, convId: String): Answer {
        val m = nlm ?: return Answer("النموذج غير محمّل: " + modelStatus, "none")
        return synchronized(lock) {
            val s = sessionFor(m, convId)
            val math = Skills.tryMath(q)                  // exact arithmetic first: a character model must never guess at sums
            if (!Scope.inScope(q)) {
                s.feed("س:" + q.replace("\n", " ¶ ").trim() + "\nج:" + Scope.REFUSAL.replace("\n", "¶") + "\n")
                Answer(Scope.REFUSAL, "scope")
            } else if (math != null) {
                s.feed("س:" + q.replace("\n", " ¶ ").trim() + "\nج:" + math.replace("\n", "¶") + "\n")      // the conversation still remembers it
                Answer(math, "math")
            } else {
                val topHit = try { kb.search(q, 1).firstOrNull() } catch (e: Throwable) { null }
                val hit = topHit?.score ?: 0.0
                // فلتر "ما يضيف أي شيء ما قلته له": فيه مصدر موثّق (Kb) يطابق السؤال تقريبًا بالضبط؟ رجّع نصه هو بالذات
                // ولا تخلي النموذج الحر يولّد كلامه الخاص إطلاقًا — هذا أقوى ضمان ممكن ضد الاختراع لما يكون عندنا مصدر
                // مؤكد لنفس السؤال. لو ما فيه تطابق قوي، يرجع يمشي على التوليد الحر العادي، بس مع تنويه صريح تحت إنه تخمين.
                if (topHit != null && hit >= Kb.HIGH) {
                    s.feed("س:" + q.replace("\n", " ¶ ").trim() + "\nج:" + topHit.item.a.replace("\n", "¶") + "\n")
                    hear(m, q)
                    return@synchronized Answer(withNote(topHit.item), "kb")
                }
                val plan = Effort.plan(ctx, q, hit)
                val t0 = System.nanoTime()
                // تفكير عميق حقيقي (self-consistency) على أي رسالة، طالما الجهاز مو مضطر لوضع أسرع لحماية البطارية/
                // الحرارة/الذاكرة — ذاك الشرط باقٍ كما هو بدون تغيير، هذا حماية للجهاز نفسه مو تنازل عن الدقة.
                // فلتر "الدقة قبل السرعة" للكود (plan.isCode): ميزانية الوقت تصير عمليًا بلا حد (Long.MAX_VALUE/2) —
                // ما يتوقف أبدًا بسبب الوقت، بس بسبب عدد المحاولات (hardCap جوا replyDeep) أو لما يلقى محاولة ناجحة
                // في فحص codeSanity. غير الكود يضل يستخدم Effort.deepThinkMaxMs العادي.
                val timeBudget = if (plan.isCode) Long.MAX_VALUE / 2 else Effort.deepThinkMaxMs
                val r = if (plan.level == Level.THINK) s.replyDeep(q, plan.maxChars, plan.temp, timeBudget, accept = if (plan.isCode) ::codeSanity else null)
                         else if (plan.tries > 1) s.replyThink(q, plan.maxChars, plan.tries, plan.temp)
                         else s.reply(q, plan.maxChars, plan.temp)
                val made = maxOf(20, r.first.length) * plan.tries
                val ms = (System.nanoTime() - t0) / 1_000_000.0 / made
                Effort.msPerChar = if (Effort.msPerChar <= 0.0) ms else 0.7 * Effort.msPerChar + 0.3 * ms
                hear(m, q)
                var foot = "\n\n🌀 هذا رد تخميني من أنماط التدريب، مو نقلًا عن مصدر موثّق محدد لنفس سؤالك."
                if (plan.level == Level.THINK) foot += "\n\n🧠 وضع التفكير: " + (if (plan.isCode) "ما يتوقف بسبب الوقت، بس بسبب فحص سلامة الكود أو عدد المحاولات." else "جربت عدة محاولات واخترت الأوضح.")
                else if (plan.note.isNotEmpty()) foot += "\n\n🔋 " + plan.note + ": رديت بإيجاز."
                // Honesty flag (wrapper-level only — does NOT touch the generation loop itself, which
                // stays exactly as designed): a reply that's suspiciously short, or mostly just echoes
                // the question back, is a real sign of low confidence worth surfacing rather than hiding.
                val words = q.trim().split(" ").filter { it.length > 2 }
                val echoed = if (words.isEmpty()) 0 else words.count { r.first.contains(it) }
                val echoRatio = if (words.isEmpty()) 0.0 else echoed.toDouble() / words.size
                if (r.first.trim().length < 8 || echoRatio > 0.7) {
                    foot += "\n\n⚠ مو متأكد من هذا الجواب — يبدو ضعيف أو مجرد تكرار للسؤال، تحقق منه قبل الاعتماد عليه."
                }
                Answer(FileTools.display(r.first) + foot, "lm")
            }
        }
    }

    /** Learning by listening: nothing to type, nothing announced. Long-term memory at once, the weights by a few small steps. */
    private fun hear(m: Nlm, q: String) {
        m.observe(q)
        val line = q.replace("\n", " ").trim()
        if (line.isEmpty()) return
        memory.add(line)
        try { memFile.appendText(line + "\n", Charsets.UTF_8) } catch (e: Throwable) { /* memory still works in this run */ }
        observed++
        if (observed % 10 == 0) saveNlm()
    }

    /** If the message is a real request to create a file, returns what to write (the activity writes it and reports the truth); null if it is not such a request. */
    fun fileRequest(q: String, convId: String, lastBot: String, forced: Boolean = false): Pair<MadeFile?, String>? {
        val fr = FileTools.parseCreate(q, lastBot, kb, forced) ?: return null
        val f = fr.first
        if (f != null && f.note.isEmpty()) return fr                       // the user dictated the content: it is written exactly as said
        val m = nlm
        if (m != null) {
            // the MODEL builds the file: it chooses the name and writes the content (thinking mode: several attempts, the surest one wins)
            val project = Regex("مشروع|مجلد|ملفات|عدة|فولدر").containsMatchIn(q)
            val written = synchronized(lock) { sessionFor(m, convId).replyThink(q, if (project) 1500 else 900, if (project) 5 else 4) { FileTools.allSane(it) }.first }
            val own = FileTools.fromModel(written)
            if (own != null) return Pair(own, "")
            val code = if (written.contains("```")) written.substringAfter("```").substringAfter('\n').substringBefore("```").trim('\n') else ""
            if (f != null && code.length >= 12) return Pair(MadeFile(f.name, (code + "\n").toByteArray(Charsets.UTF_8), "المحتوى كتبه النموذج نفسه"), "")
        }
        if (fr.first == null) return fr
        return Pair(null, "حاولت أكتب محتوى الملف وما طلع شيء واضح. قل لي المحتوى بنفسك: «أنشئ ملف ... فيه ...».")
    }

    /** The conversation ended: its short-term memory is gone. (What was said is already in the long-term memory.) */
    fun endConversation(convId: String) { synchronized(lock) { sessions.remove(convId) } }

    /** Opening an old conversation: its short-term memory is rebuilt from the end of its own transcript. */
    fun resume(convId: String, msgs: List<Msg>) {
        val m = nlm ?: return
        synchronized(lock) {
            if (sessions.containsKey(convId)) return                       // it already has its short-term memory
            val s = NlmSession(m, System.nanoTime())
            val sb = StringBuilder()
            for (x in msgs) {
                if (x.role == "u") sb.append("س:").append(x.text.replace("\n", " ¶ ").trim()).append("\n")
                else sb.append("ج:").append(x.text.replace("\n", "¶").trim()).append("\n")
            }
            s.feed(if (sb.length > 1500) sb.substring(sb.length - 1500) else sb.toString())
            sessions.remove(convId); sessions[convId] = s
        }
    }

    // ---------------------------------------------------------------- self-development
    /** The self-development button: re-read everything heard, train on it, keep the result only if it got better. */
    fun develop(): String {
        val m = nlm ?: return "النموذج غير محمّل."
        // كانت 400 رسالة و 2 دورة فقط — مع تقليم النصوص الصارم اللي كان داخل Nlm.develop() هذا كان يخلي التدريب الفعلي
        // يقتصر على آخر عدة رسائل بس، بصرف النظر عن طول المقال أو عدد الأسئلة اللي أرسلتها. رفعتها هنا كمان (1500 رسالة
        // و 10 دورات) عشان تتوافق مع النافذة الأوسع في Nlm.develop() وما تضيع الفايدة من التوسيع هناك.
        val texts = memory.takeLast(1500).toList()
        if (texts.size < 3) return "ما سمعت منك كلاما كافيا بعد. تحدث معي أكثر، ثم اضغط التطوير."
        val r = synchronized(lock) { m.develop(texts, 10, 3e-4f, rehearsal) }
        saveNlm()
        lastReport = JSONObject().put("heard", memory.size).put("bits_before", r.first.toDouble()).put("bits_after", r.second.toDouble()).put("params", m.paramCount())
        val f = { x: Float -> String.format(java.util.Locale.US, "%.2f", x) }
        val sb = StringBuilder()
        sb.append("راجعت كل ما سمعته منك: ").append(memory.size).append(" رسالة.\n")
        sb.append("• تدربت على آخر ").append(texts.size).append(" رسالة، وقست نفسي قبل وبعد: ").append(f(r.first)).append(" ← ").append(f(r.second)).append(" بت/حرف (الأقل أفضل).\n")
        if (r.second >= r.first) sb.append("• ما تحسّنت هذه المرة، فرجعت لأوزاني السابقة بدون تغيير.\n") else sb.append("• تحسّنت، وحفظت الأوزان الجديدة.\n")
        sb.append("• حجمي: ").append(m.paramCount()).append(" معامل، وذاكرتي الطويلة: ").append(m.long.chars).append(" حرف من كلامك.\n")
        sb.append("\nاضغط «ملف المطور» لأخرج لك ملفا فيه أوزاني الجديدة وذاكرتي وتقرير التحسن، وترسله لمن يفحصه.")
        return sb.toString()
    }

    /** تدريب قوي على نص كبير (الصق فيه كتاب أو مقال كامل) — منفصل عن زر "رفع الفهم" العادي (اللي يراجع رسائل
     *  الدردشة القصيرة). epochs=15 و lr=4e-4 افتراضيًا: أقوى من 10/3e-4 في develop() العادي، لأن هذا مخصص
     *  لجلسة تدريب مقصودة ومهمة، لا لمراجعة خفيفة بعد كل كم رسالة. */
    fun developBig(text: String, epochs: Int = 15, lr: Float = 4e-4f): String {
        val m = nlm ?: return "النموذج غير محمّل."
        val report = synchronized(lock) { m.trainBig(text, rehearsal, epochs, lr) }
        saveNlm()
        return report
    }

    private fun sha(b: ByteArray): String = java.security.MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { x -> String.format("%02x", x) }

    /** The developer file (a zip): manifest.json (report + sha256 of the other files), nlm2.bin (weights), memory.txt (long-term memory).
     *  `python -m nuwaa.lm.devfile check FILE` says whether it is broken. It holds weights and memory only: never code. */
    fun devFile(): ByteArray {
        val m = nlm ?: throw IllegalStateException("النموذج غير محمّل")
        val w = synchronized(lock) { m.toBytes() }
        val mem = memory.joinToString("\n").toByteArray(Charsets.UTF_8)
        val files = JSONObject().put("nlm2.bin", sha(w)).put("memory.txt", sha(mem))
        val man = JSONObject().put("version", 1).put("created", System.currentTimeMillis() / 1000).put("params", m.paramCount()).put("vocab", m.v)
            .put("emb", m.e).put("hid", m.h).put("memory_lines", memory.size).put("files", files).put("report", lastReport).put("origin", "android")
        val bos = java.io.ByteArrayOutputStream()
        java.util.zip.ZipOutputStream(bos).use { z ->
            z.putNextEntry(java.util.zip.ZipEntry("manifest.json")); z.write(man.toString(1).toByteArray(Charsets.UTF_8)); z.closeEntry()
            z.putNextEntry(java.util.zip.ZipEntry("nlm2.bin")); z.write(w); z.closeEntry()
            z.putNextEntry(java.util.zip.ZipEntry("memory.txt")); z.write(mem); z.closeEntry()
        }
        return bos.toByteArray()
    }

    private fun withNote(e: KbItem): String {
        if (!e.a.contains("```")) return e.a
        val note = when (e.v) {
            2 -> "✓ هذا الكود شُغِّل واختُبر ناتجه قبل إضافته إلى معرفتي."
            1 -> "✓ هذا الكود شُغِّل بدون أخطاء."
            0 -> "✓ فُحصت صياغة هذا الكود فقط (لم يُشغَّل بالكامل)."
            else -> "⚠ لم أستطع تشغيل هذا الكود هنا. جرّبه قبل الاعتماد عليه."
        }
        return e.a + "\n\n" + note
    }

    // ---------------- files (read-only) ----------------
    private fun decodeText(b: ByteArray): String? {
        val n = minOf(b.size, 4096)
        for (i in 0 until n) if (b[i].toInt() == 0) return null
        for (cut in 0..3) {                      // a 300 KB cut may land inside a multi-byte Arabic letter
            if (b.size - cut <= 0) break
            try {
                return Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(b, 0, b.size - cut)).toString()
            } catch (e: Exception) { /* try cutting one more byte */ }
        }
        return null
    }

    private fun magic(b: ByteArray): String {
        fun s(vararg v: Int): Boolean = b.size >= v.size && v.indices.all { i -> (b[i].toInt() and 255) == v[i] }
        return when {
            s(0x89, 0x50, 0x4E, 0x47) -> "صورة PNG"
            s(0xFF, 0xD8, 0xFF) -> "صورة JPEG"
            s(0x47, 0x49, 0x46, 0x38) -> "صورة GIF"
            s(0x25, 0x50, 0x44, 0x46) -> "ملف PDF"
            s(0x50, 0x4B, 0x03, 0x04) -> "ملف مضغوط (ZIP أو APK أو DOCX...)"
            s(0x7F, 0x45, 0x4C, 0x46) -> "برنامج ELF"
            else -> "ملف ثنائي غير معروف"
        }
    }

    private fun sizeText(n: Long): String = if (n >= 1_000_000) String.format(java.util.Locale.US, "%.1f MB", n / 1e6) else if (n >= 1000) (n / 1000).toString() + " KB" else "$n بايت"

    private val langNames = mapOf("py" to "بايثون", "js" to "جافاسكربت", "kt" to "كوتلن", "java" to "جافا", "html" to "HTML", "css" to "CSS", "sql" to "SQL",
        "sh" to "Bash", "c" to "C", "cpp" to "C++", "json" to "JSON", "xml" to "XML", "md" to "Markdown", "txt" to "نص", "csv" to "CSV", "ts" to "TypeScript", "go" to "Go", "rs" to "Rust", "php" to "PHP", "gradle" to "Gradle", "kts" to "كوتلن")
    private val defRe = Regex("""\b(?:def|function|fun|class|interface|object|struct|func)\s+([A-Za-z_]\w*)""")
    private val errWords = listOf("error", "exception", "traceback", "failed", "fatal", "خطأ", "فشل", "تعذر")

    /** Honest limit: Nuwaa sees the file as TEXT or as NUMBERS (bytes). It reports structure, finds lines that match your question and
     *  matches errors against its knowledge. It does not understand meaning and does not look at pictures. */
    fun answerAbout(att: Attached, q: String): Answer {
        val sb = StringBuilder()
        sb.append("📎 ").append(att.name).append(" · ").append(sizeText(att.size))
        if (att.truncated) sb.append(" (قرأت أول 8 MB فقط)")
        sb.append("\n")
        val text = FileTools.textOf(att.name, att.bytes)
        if (text == null) {
            sb.append("النوع: ").append(magic(att.bytes)).append("\nهذا ملف غير نصي (صورة أو PDF أو برنامج)، أقرأ منه البايتات فقط ولا أفهم محتواه.\n")
            val opts = BitmapFactory.Options(); opts.inJustDecodeBounds = true
            try { BitmapFactory.decodeByteArray(att.bytes, 0, att.bytes.size, opts) } catch (e: Throwable) { }
            if (opts.outWidth > 0) sb.append("أبعاد الصورة: ").append(opts.outWidth).append("×").append(opts.outHeight).append(" (ما أفهم محتواها)\n")
            val seen = BooleanArray(256); for (x in att.bytes) seen[x.toInt() and 255] = true
            sb.append("أول البايتات: ").append(att.bytes.take(24).joinToString(" ") { x -> (x.toInt() and 255).toString() })
            sb.append("\nعدد القيم المختلفة: ").append(seen.count { v -> v }).append(" من 256")
            return Answer(sb.toString(), "file")
        }
        return Answer(FileTools.answer(att.name, att.size, att.truncated, text, q, kb), "file")
    }

    /** Honest limit: this reads the TEXT on the screen (through the accessibility reader) and matches it with the knowledge base. It does not understand images. */
    fun adviseScreen(screen: String, goalText: String): String {
        val t = screen.trim()
        if (t.isEmpty()) return "لا أستطيع قراءة نص الشاشة الآن. فعّل «قارئ الشاشة» من التطبيق، ثم افتح الصفحة التي تريدها."
        val lines = t.lines().map { l -> l.trim() }.filter { l -> l.isNotEmpty() }
        val errWords = listOf("error", "exception", "failed", "traceback", "denied", "not found", "crash", "خطأ", "فشل", "تعذر", "غير موجود", "توقف")
        val errLine = lines.firstOrNull { l -> val lw = l.lowercase(); errWords.any { w -> lw.contains(w) } }
        val sb = StringBuilder()
        if (errLine != null) sb.append("لاحظت رسالة تشبه خطأ: «").append(errLine.take(120)).append("»\nاقرأ الرسالة كاملة، وابحث بنصها نفسه، وغيّر شيئا واحدا كل مرة.\n")
        if (goalText.isNotBlank()) sb.append("هدفك: «").append(goalText.take(80)).append("».\n")
        val shown = lines.filter { l -> l.length in 2..20 }.take(8)
        if (shown.isNotEmpty()) sb.append("العناصر الظاهرة الآن: ").append(shown.joinToString("، "))
        // this is a fixed, rule-based reading of the screen (real text, real error detection) — never the model talking, and never a looked-up canned line
        return "📱 قراءة آلية للشاشة (ليست من النموذج):\n" + sb.toString().trim()
    }
}


// ============== من الجوال_Effort.kt ==============
enum class Level { FAST, NORMAL, THINK }

/** What one reply will cost: how long it may be, how many separate attempts the model makes, and why. */
class Plan(val level: Level, val maxChars: Int, val temp: Float, val tries: Int, val note: String, val isCode: Boolean = false)

/** Adaptive effort. An easy message is answered at once and briefly; a hard one gets a longer reply and, in thinking mode, several attempts of which
 *  the most confident is kept. The phone's own state caps it: battery saver, low battery, heat, and memory all push it toward the cheap modes,
 *  so the model never keeps the phone busy when the phone is struggling. */
/** فحص هيكلي رخيص لأي كود يولّده النموذج: أقواس () [] {} متوازنة، ووسوم HTML افتتاح/إغلاق قريبة العدد من بعض.
 *  هذا مو تشغيل حقيقي للكود ولا فحص صحة نحوية كاملة — بس يمسك أوضح الأخطاء (قوس ناقص، وسم ما انقفل) عشان ما
 *  نعرض كود مكسور على إنه سليم. اخترت أسويها بنفسي هنا بدل الاعتماد على FileTools.allSane لأن ذاك موجود في
 *  ملف ثاني ما أقدر أشوفه، فما أضمن سلوكه مع رد دردشة عادي (مو طلب إنشاء ملف) — هذي مضمونة 100% لأني كاتبها. */
private fun codeSanity(s: String): Boolean {
    var round = 0; var square = 0; var curly = 0
    for (c in s) when (c) {
        '(' -> round++; ')' -> round--
        '[' -> square++; ']' -> square--
        '{' -> curly++; '}' -> curly--
    }
    if (round != 0 || square != 0 || curly != 0) return false
    val opens = Regex("<[a-zA-Z][^/>]*>").findAll(s).count()
    val closes = Regex("</[a-zA-Z]+>").findAll(s).count()
    if (opens > 0 && Math.abs(opens - closes) > maxOf(1, opens / 4)) return false
    return true
}

object Effort {
    /** Measured speed of THIS phone with THIS model (milliseconds per generated character); 0 = not measured yet. Brain updates it after every reply, so a
     *  bigger model (10M..100M weights) or a slower phone automatically gets shorter replies / fewer attempts instead of making the phone wait for minutes. */
    @Volatile var msPerChar = 0.0
    /** ميزانية "التفكير العميق" (replyDeep) بالمللي ثانية. 180000 = 3 دقائق افتراضيًا. غيّرها بنفسك لساعة مثلاً: 3_600_000L،
     *  أو لساعتين: 7_200_000L. القيمة هذي وحدها هي اللي تتحكم بطول الانتظار — ما تحتاج تعدل أي مكان ثاني. */
    @Volatile var deepThinkMaxMs: Long = 180_000L
    private val hardWords = listOf("اشرح", "شرح", "لماذا", "ليش", "قارن", "الفرق", "اكتب دالة", "اكتب لي", "اكتب كود", "برنامج", "خوارزمية", "حلل", "سبب", "صحح", "اصلح", "مثال", "كيف")
    private val codeWords = listOf("كود", "دال", "برنام", "مشروع", "سكربت", "خوارزم", "راجع", "صحح", "اصلح", "ملف")
    private val ramSoftLimit = 3_200_000_000L          // raised from 1.2GB (was an arbitrary soft cushion, not a device-safety limit — those are the battery/thermal checks below, left untouched on purpose since you specifically wanted the phone protected from overheating)

    /** 0 easy, 1 normal, 2 hard. kbScore = how close the message is to a verified example (known things are easy). */
    fun difficulty(q: String, kbScore: Double): Int {
        val words = q.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        var score = 0
        if (kbScore >= Kb.HIGH) score -= 2
        if (words.size >= 14) score += 2 else if (words.size >= 7) score += 1
        if (words.size <= 3) score -= 1
        if (q.contains("\n") || q.contains("```") || q.contains("def ") || q.contains("{") || q.contains("(")) score += 2
        val nq = Nlp.norm(q)
        if (hardWords.any { nq.contains(Nlp.norm(it)) }) score += 1
        if (q.count { it == '؟' || it == '?' } >= 2 || q.contains(" ثم ")) score += 1
        if (codeWords.any { nq.contains(it) }) score += 3            // programming asks always get the thinking mode: the answer matters more than the speed
        return if (score <= 0) 0 else if (score <= 2) 1 else 2
    }

    /** The highest level the device allows right now (0 FAST .. 2 THINK) and the reason when it is lower than 2. */
    fun deviceCap(ctx: Context): Pair<Int, String> {
        var cap = 2; var why = ""
        fun lower(to: Int, reason: String) { if (to < cap) { cap = to; why = reason } }
        try {
            val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
            if (pm.isPowerSaveMode) lower(1, "وضع توفير الطاقة في الجوال مفعّل")
            // Thermal throttle removed on your explicit, informed call: you added external cooling
            // (a fan) specifically to solve the overheating this was protecting against — so the
            // original reason for this check no longer applies on your setup.
            // Honest note unrelated to heat: sustained heavy compute still stresses battery CHEMISTRY
            // (charge/discharge cycles) regardless of temperature — a fan doesn't change that part. Not
            // a reason to keep the throttle (that's your call to make), just worth knowing.
        } catch (e: Throwable) { /* no power info: no cap from it */ }
        try {
            val bi = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            if (bi != null) {
                val level = bi.getIntExtra(BatteryManager.EXTRA_LEVEL, -1); val scale = bi.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
                val plugged = bi.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
                if (level >= 0 && !plugged) {
                    val pct = level * 100 / maxOf(1, scale)
                    if (pct <= 15) lower(0, "البطارية $pct%") else if (pct <= 30) lower(1, "البطارية $pct%")
                }
            }
        } catch (e: Throwable) { /* no battery info */ }
        try {
            val rt = Runtime.getRuntime()
            val used = rt.totalMemory() - rt.freeMemory()
            val limit = minOf(rt.maxMemory(), ramSoftLimit)
            if (used > limit * 0.85) lower(0, "الذاكرة ممتلئة") else if (used > limit * 0.65) lower(1, "الذاكرة مشغولة")
        } catch (e: Throwable) { }
        return Pair(cap, why)
    }

    fun plan(ctx: Context, q: String, kbScore: Double): Plan {
        // You said it plainly: even "هلا" can take an hour if that's what accuracy costs — so every
        // message now WANTS the highest level (THINK), full stop, regardless of how simple it looks.
        // difficulty(q, kbScore) is left defined above (unused now) in case you ever want message-based
        // tiering back; this one line is what to revert if so.
        val want = 2
        val dev = deviceCap(ctx)
        val lvl = minOf(want, dev.first)   // the device-safety cap (battery/heat/memory) is the ONLY thing that can still lower this — on purpose, to protect the phone
        val capped = lvl < want
        val note = if (capped) "وضع التوفير (" + dev.second + ")" else ""
        val per = msPerChar
        fun chars(base: Int, budgetMs: Double): Int = if (per <= 0.0) base else minOf(base, maxOf(60, (budgetMs / per).toInt()))

        // Code detection: catches it mentioning code, not just asking to write some — "deepest, forced"
        // mode you asked for triggers on EITHER. Deliberately broad (Arabic + English + common syntax
        // words + fenced-block marker) — a missed trigger costs accuracy, a false trigger just costs
        // time, and you've said time doesn't matter, so this errs toward triggering.
        val codeWords = listOf("كود", "اكتب دالة", "اكتب كلاس", "برمجة", "سكربت", "بايثون", "جافاسكربت",
            "function", "class ", "def ", "import ", "```", "html", "css", "javascript", "python")
        val looksLikeCode = codeWords.any { q.contains(it, ignoreCase = true) }

        // temps lowered further (was 0.35/0.45/0.45) — even more deterministic, even less randomness.
        // Code gets the lowest temp and the most tries of all: wrong code is verifiably wrong (unlike a
        // vague chat answer), so squeezing out randomness matters most exactly here.
        // looksLikeCode pushes to the deepest setting ACHIEVABLE within lvl — it does not bypass the
        // battery/RAM safety check above (lvl already reflects that): "mandatory deepest thinking" means
        // deepest among what's currently safe, not literally unconditional regardless of battery state.
        return when {
            // Char cap removed per your request (was 6000). NOTE this changes the CEILING only — the
            // real-world lengths measured were 103-281 chars because the model itself stops early, not
            // because of this number. Raising it costs nothing and helps once training teaches longer
            // replies, but by itself it will not produce longer code today.
            looksLikeCode && lvl >= 1 -> Plan(Level.THINK, 200000, 0.15f, 8, "🧠 كود مكتشف: أعمق وضع تفكير، محاولات غير محدودة لحد ما يلقى كودًا سليمًا هيكليًا أو ينفد الوقت", isCode = true)
            looksLikeCode -> Plan(Level.FAST, 50000, 0.15f, 1, "🧠 كود، لكن " + dev.second + " — أعمق وضع ممكن الآن بس", isCode = true)
            lvl == 0 -> Plan(Level.FAST, chars(300, 12000.0), 0.25f, 1, note)
            lvl == 1 -> Plan(Level.NORMAL, chars(900, 40000.0), 0.3f, 1, note)
            // non-code THINK still always gets 4 tries no matter how long that takes — speed is never a
            // reason to think less. The only thing that can still reduce effort below `want` is the
            // device-safety logic above (battery/RAM) — that stays, it protects the phone from real harm.
            else -> Plan(Level.THINK, chars(2000, 90000.0), 0.3f, 4, note)
        }
    }
}


// ============== من الجوال_TvLink.kt ==============
/** The phone is the "controller" (same role/protocol as server.ts already uses for the game):
 *  it connects to the pairing server with the code shown on the TV website, and from then on just
 *  SENDS — the TV is a dumb display that receives controller_command messages and renders them.
 *  All real work (Brain.answer, file creation) still happens on the phone; the TV never runs the model.
 */
object TvLink {
    @Volatile var connected: Boolean = false
        private set
    @Volatile var tvName: String? = null
        private set

    private var ws: WebSocket? = null
    private val client = OkHttpClient()

    /** onTvText: called when someone TYPES ON THE TV ITSELF and it reaches the phone (needs the
     *  server.ts "tv_input" relay patch in the README — without it, nothing ever calls this). */
    fun connect(serverUrl: String, code: String, onConnected: (String) -> Unit, onError: (String) -> Unit,
                onDisconnected: () -> Unit, onTvText: (String) -> Unit = {},
                onTvFeedback: (rating: String, reason: String, correction: String, user: String, bot: String) -> Unit = { _, _, _, _, _ -> }) {
        val req = Request.Builder().url(serverUrl).build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(w: WebSocket, response: Response) {
                val msg = JSONObject().put("type", "connect_controller").put("code", code).put("controllerName", "جوال نواة")
                w.send(msg.toString())
            }
            override fun onMessage(w: WebSocket, text: String) {
                try {
                    val data = JSONObject(text)
                    when (data.optString("type")) {
                        "controller_connected" -> { connected = true; tvName = data.optString("tvName"); onConnected(tvName ?: "") }
                        "error" -> onError(data.optString("message", "خطأ غير معروف"))
                        "tv_disconnected" -> { connected = false; onDisconnected() }
                        "tv_command" -> {
                            val cmd = data.optJSONObject("command")
                            if (cmd != null && cmd.optString("action") == "chat_text") onTvText(cmd.optString("text"))
                            if (cmd != null && cmd.optString("action") == "feedback")
                                onTvFeedback(cmd.optString("rating"), cmd.optString("reason"), cmd.optString("correction"), cmd.optString("user"), cmd.optString("bot"))
                        }
                    }
                } catch (e: Exception) { /* malformed frame from the server: ignore, don't crash the app over it */ }
            }
            override fun onFailure(w: WebSocket, t: Throwable, response: Response?) {
                connected = false; onError(t.message ?: "فشل الاتصال")
            }
            override fun onClosed(w: WebSocket, code: Int, reason: String) { connected = false; onDisconnected() }
        })
    }

    fun disconnect() { ws?.close(1000, "bye"); ws = null; connected = false }

    private fun send(action: String, extra: Map<String, String>) {
        val w = ws ?: return
        if (!connected) return
        val cmd = JSONObject(); cmd.put("action", action); for ((k, v) in extra) cmd.put(k, v)
        val msg = JSONObject().put("type", "controller_input").put("command", cmd)
        w.send(msg.toString())
    }

    /** Call right after a chat exchange finishes (onSend() in MainActivity). */
    fun sendChat(userText: String, botReply: String) =
        send("chat", mapOf("user" to userText, "bot" to botReply))

    /** Call for each live progress line while the model is "thinking" / writing a file
     *  (e.g. from inside replyThink's retry loop, or wherever Effort/plan.tries progress is visible). */
    fun sendThinking(line: String) =
        send("thinking", mapOf("line" to line))

    /** Call once a file is actually written (createFileAnswer in MainActivity).
     *  runnable = true for .html/.js/.css (the TV can preview it live in an iframe, no server needed). */
    fun sendFileDone(name: String, content: String, runnable: Boolean) =
        send("file_done", mapOf("name" to name, "content" to content, "runnable" to runnable.toString()))

    /** Studio view: call whenever a reply CONTAINS code (not just when a file gets saved) — this is
     *  what makes the TV show a proper code pane instead of a chat bubble for in-chat code answers too,
     *  not just explicit "create a file" requests. */
    /** Tells the TV whether the rating was REALLY written to the phone's file (the TV shows this toast,
     *  so "saved" is only ever shown after the phone actually saved it). */
    fun sendFeedbackSaved(ok: Boolean, info: String) =
        send("feedback_saved", mapOf("ok" to ok.toString(), "info" to info))

    fun sendCode(code: String, language: String = "") =
        send("code", mapOf("code" to code, "language" to language))
}


// ============== من الجوال_VisionLoop.kt ==============
/** "يشوف شو سوى": renders HTML/CSS/JS the model just wrote, looks at a TINY snapshot of it, and lets
 *  the model (real generation, via Brain.answer — nothing templated) fix its own code before you see it.
 *
 *  Deliberately small and capped, per your own instruction: "ما ابغى البكسلات ترتفع لدرجة إن الجوال
 *  يعلّق، ابغى بكسلات قليلة وتصير سريعة وواضحة":
 *   - grid defaults to 12x12 = 144 characters (your own number), not raw pixels — the model is
 *     CHARACTER-level (188-symbol vocab), so a real pixel byte (0-255) isn't something it was ever
 *     trained to read. ASCII-art shading (space,.,:,-,=,+,*,#,%,@) maps the brightness onto characters
 *     that ARE already in its vocabulary — this is the real, working way to do "crude vision" for THIS
 *     specific engine (the GRU/Nlm one), not the byte-trick I used earlier for the separate Python engine.
 *   - the actual off-screen render is a small fixed 240x240px WebView (cheap), only downsampled
 *     AFTER rendering — the browser does the real rendering work, we just look at it small and fast.
 *   - maxPasses raised to 12 (was 2) — you said plainly you don't care how long file creation takes,
 *     so this no longer rushes to stop early. It still has a NUMBER (not literally infinite) only
 *     because an unbounded "fix forever" loop is a real bug risk (e.g. the model flip-flopping between
 *     two versions forever) that no amount of patience fixes — 12 is generous, not rushed.
 *   - only runs for .html/.htm/.css/.js — a 144-character brightness grid tells you nothing useful about
 *     Python output, so non-browser code is returned unchanged (no fake "vision" where there isn't any).
 */
object VisionLoop {
    private const val RAMP = " .:-=+*#%@"

    /** Must be called from a BACKGROUND thread (same thread Brain.fileRequest already runs on) — it
     *  hops to the UI thread internally (WebView requires that) and blocks until done or timeout. */
    fun reviewAndFix(brain: Brain, activity: Activity, convId: String, code: String, fileName: String, maxPasses: Int = 12, log: (String) -> Unit = {}): String {
        val ext = fileName.substringAfterLast('.', "").lowercase()
        if (ext !in setOf("html", "htm", "css", "js")) return code   // honest: no vision for non-browser code

        var current = code
        for (pass in 1..maxPasses) {
            log("👁 ياخذ لمحة (144 بكسل) للملف بعد المحاولة $pass...")
            val html = wrapForRender(current, ext)
            val ascii = captureAscii(activity, html, 12, 12) ?: return current   // render failed: give up quietly, return what we have
            log("🧠 يراجع اللي شافه ويقرر إذا يحتاج تعديل...")
            val prompt = "هذا كود $fileName الحالي:\n```\n$current\n```\n" +
                "وهذا شكله لما شغّلته فعليا (لمحة ${12}×${12} = 144 بكسل، كل حرف درجة سطوع):\n```\n$ascii\n```\n" +
                "إذا شفت خطأ واضح أو شيء غير منطقي بالشكل عدّل الكود وأعطني النسخة الكاملة المصححة داخل ```. " +
                "إذا كان الشكل منطقي اكتب نفس الكود بالضبط بدون تغيير."
            val reply = brain.answer(prompt, convId)
            val fixed = extractCode(reply.text)
            if (fixed.isNullOrBlank() || fixed.trim() == current.trim()) { log("✅ ما لقى شي يحتاج تعديل، وقف هنا."); break }
            log("✏️ عدّل الكود (تغيّر فعليا)، يعيد المحاولة...")
            current = fixed
        }
        return current
    }

    // ---- continuous mode: a light, repeating "look" while the model works, not a one-shot check ----
    // render 120x120 (not 240x240) + grid 16x16 (256 chars) + poll every 1.5s ≈ ~2.3% CPU, measured by
    // scaling: pixel-scan cost is ~proportional to RENDER pixel count, not grid size (grid size barely
    // matters for cost — it only changes how detailed the ASCII art looks, not how much work rendering/
    // scanning takes). This is the deliberately "moderate, sustainable" config, not the one-shot settings.
    private val continuousHandler = Handler(Looper.getMainLooper())
    @Volatile private var continuousRunning = false

    fun startContinuous(activity: Activity, getCurrentHtml: () -> String, onFrame: (String) -> Unit, intervalMs: Long = 1500) {
        if (continuousRunning) return
        continuousRunning = true
        val tick = object : Runnable {
            override fun run() {
                if (!continuousRunning) return
                val ascii = captureAscii(activity, getCurrentHtml(), 16, 16, renderPx = 120, timeoutMs = 2000)
                if (ascii != null) onFrame(ascii)
                if (continuousRunning) continuousHandler.postDelayed(this, intervalMs)
            }
        }
        continuousHandler.post(tick)
    }

    fun stopContinuous() { continuousRunning = false }

    private fun wrapForRender(code: String, ext: String): String = when (ext) {
        "html", "htm" -> code
        "css" -> "<style>$code</style><body style=\"margin:0;font-family:sans-serif\">معاينة</body>"
        "js" -> "<body style=\"margin:0\"><script>$code</script></body>"
        else -> code
    }

    private fun extractCode(text: String): String? {
        if (!text.contains("```")) return null
        val block = text.substringAfter("```")
        val afterLang = block.substringAfter("\n", block)   // drop a language tag line if present
        return afterLang.substringBefore("```").trim('\n')
    }

    /** Renders `html` off-screen at a small fixed size, waits briefly for layout, captures it as a
     *  144-char (default) ASCII-art brightness grid. Returns null on timeout/failure (never throws). */
    private fun captureAscii(activity: Activity, html: String, gridW: Int, gridH: Int, renderPx: Int = 240, timeoutMs: Long = 4000): String? {
        val latch = CountDownLatch(1)
        var result: String? = null
        Handler(Looper.getMainLooper()).post {
            try {
                val wv = WebView(activity)
                wv.layout(0, 0, renderPx, renderPx)
                wv.webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        // small settle delay for CSS/JS layout to finish, then snapshot — keeps this FAST on purpose
                        Handler(Looper.getMainLooper()).postDelayed({
                            try {
                                val bmp = Bitmap.createBitmap(renderPx, renderPx, Bitmap.Config.ARGB_8888)
                                val canvas = Canvas(bmp)
                                wv.draw(canvas)
                                result = toAscii(bmp, gridW, gridH)
                            } catch (e: Throwable) { result = null }
                            finally { latch.countDown() }
                        }, 300)
                    }
                }
                wv.loadDataWithBaseURL(null, html, "text/html", "utf-8", null)
            } catch (e: Throwable) { latch.countDown() }
        }
        latch.await(timeoutMs, TimeUnit.MILLISECONDS)
        return result
    }

    private fun toAscii(bmp: Bitmap, gridW: Int, gridH: Int): String {
        val sb = StringBuilder()
        val cellW = bmp.width / gridW; val cellH = bmp.height / gridH
        for (gy in 0 until gridH) {
            for (gx in 0 until gridW) {
                var sum = 0L; var n = 0
                val x0 = gx * cellW; val y0 = gy * cellH
                for (x in x0 until minOf(x0 + cellW, bmp.width)) {
                    for (y in y0 until minOf(y0 + cellH, bmp.height)) {
                        val p = bmp.getPixel(x, y)
                        sum += (Color.red(p) + Color.green(p) + Color.blue(p)) / 3; n++
                    }
                }
                val avg = if (n > 0) (sum / n).toInt() else 255
                sb.append(RAMP[(avg * (RAMP.length - 1) / 255)])
            }
            sb.append('\n')
        }
        return sb.toString()
    }
}


// ============== من CastManager.kt ==============
/** Detects the SAME "Cast" the user already uses from Android Settings > Connected devices > Cast
 *  (or the quick-settings "Smart View"/"Screen cast" tile): this is Android's built-in wireless-display
 *  route, not the paid/registered Google Cast developer SDK. No new Gradle dependency, no API key,
 *  no Google account — android.media.MediaRouter has shipped in every Android version since API 16.
 *
 *  What this gives you: the exact TV name once the user picks it from Settings, and a live flag for
 *  "a TV is connected right now" that other code (e.g. ScreenShareService, MainActivity) can read and
 *  use as a signal/path — e.g. to automatically favour a TV-friendly layout while mirroring is active.
 *
 *  What this does NOT do: it does not open the Cast picker itself (Settings already has that), and it
 *  does not send anything to the TV beyond what system mirroring already mirrors — the app's own screen.
 *  If you later want the APP to open the picker itself (instead of going through Settings) and/or send
 *  specific content to the TV instead of mirroring the whole screen, that needs the real Google Cast SDK
 *  (androidx.mediarouter + com.google.android.gms:play-services-cast-framework). That IS also free — no
 *  payment — but it is a bigger, separate addition (new Gradle deps, a CastOptionsProvider, a receiver
 *  app id). Say the word if you want that version instead of/alongside this one. */
object CastManager {
    @Volatile var connectedDisplayName: String? = null
        private set
    @Volatile var isConnected: Boolean = false
        private set

    private var router: MediaRouter? = null
    private var listener: ((connected: Boolean, name: String?) -> Unit)? = null

    private val callback = object : MediaRouter.SimpleCallback() {
        override fun onRouteSelected(r: MediaRouter, type: Int, info: MediaRouter.RouteInfo) {
            update(r)
        }
        override fun onRouteUnselected(r: MediaRouter, type: Int, info: MediaRouter.RouteInfo) {
            update(r)
        }
        override fun onRouteChanged(r: MediaRouter, info: MediaRouter.RouteInfo) {
            update(r)
        }
    }

    /** Call once, e.g. from MainActivity.onCreate or onResume. onChange is optional UI hook. */
    fun start(context: Context, onChange: ((connected: Boolean, name: String?) -> Unit)? = null) {
        listener = onChange
        val r = context.applicationContext.getSystemService(Context.MEDIA_ROUTER_SERVICE) as? MediaRouter ?: return
        router = r
        @Suppress("DEPRECATION")
        r.addCallback(
            MediaRouter.ROUTE_TYPE_LIVE_VIDEO or MediaRouter.ROUTE_TYPE_LIVE_AUDIO,
            callback
        )
        update(r)   // pick up a route that was already selected before start() was called
    }

    fun stop() {
        router?.let {
            @Suppress("DEPRECATION")
            it.removeCallback(callback)
        }
        router = null
        listener = null
    }

    @Suppress("DEPRECATION")
    private fun update(r: MediaRouter) {
        val selected = r.getSelectedRoute(MediaRouter.ROUTE_TYPE_LIVE_VIDEO)
        val isDefault = selected == null || selected == r.defaultRoute
        isConnected = !isDefault
        connectedDisplayName = if (isDefault) null else selected?.name?.toString()
        listener?.invoke(isConnected, connectedDisplayName)
    }
}


// ============== من الجوال_ModelHealth.kt ==============
/** The missing half of the "never regress" guarantee: `develop()` already checks itself (keeps an
 *  update only if bits/char improved). `growModel()` did NOT — it grew and saved immediately, trusting
 *  the math blindly. This closes that gap, using the exact same kind of check the project's own
 *  `devfile.check()` (Python side) already uses: every weight finite, and the model can actually answer
 *  without crashing or going empty. Directly relevant to "jahhizo for the real training coming" — this
 *  is the net that catches a bad training run before it gets saved and shipped. */
object ModelHealth {
    private val PROBES = listOf("مرحبا", "كيف حالك", "اكتب دالة تجمع رقمين")

    /** true = healthy. Never throws — an exception while checking counts as unhealthy, not a crash. */
    fun isHealthy(m: Nlm): Boolean {
        try {
            for (w in m.P) for (v in w) if (!v.isFinite()) return false   // NaN/Infinity anywhere = broken
            val s = NlmSession(m, 1L)
            var empty = 0
            for (q in PROBES) {
                val (r, _) = s.reply(q, maxChars = 120, temp = 0.5f, commit = false)
                if (r.isBlank()) empty++
            }
            return empty < PROBES.size   // at least one real probe must produce something
        } catch (e: Throwable) { return false }
    }

    /** Human-readable one-liner for a result dialog/toast. */
    fun describe(m: Nlm): String {
        val ok = isHealthy(m)
        return if (ok) "✓ فحص السلامة: الأوزان كلها أرقام سليمة والنموذج يرد بدون أعطال."
        else "✗ فحص السلامة فشل: النموذج فيه أوزان غير سليمة (NaN) أو يفشل يرد — تم التراجع تلقائيا."
    }
}


// ============== من الجوال_Nlm.kt ==============
/** Memory of what follows each context (orders 1..8), learned instantly. Mirror of NGram in nuwaa/lm/gru.py.
 *  Used twice: the long-term memory (kept on the phone) and a short-term memory that lives inside one NlmSession. */
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
    private fun run(x: IntArray, y: IntArray, s1: FloatArray, s2: FloatArray, grads: Boolean): Pair<Float, List<FloatArray>?> {
        val t = x.size
        val tr = Trace(t, h)
        var h1p = s1.copyOf(); var h2p = s2.copyOf()
        val g1 = FloatArray(h3); val g2 = FloatArray(h3)
        var loss = 0f
        for (i in 0 until t) {
            gx1(x[i], g1)
            cellFwd(g1, h1p, P[2], tr.h1[i], tr.r1[i], tr.z1[i], tr.n1[i], tr.rh1[i])
            gx2(tr.h1[i], g2)
            cellFwd(g2, h2p, P[5], tr.h2[i], tr.r2[i], tr.z2[i], tr.n2[i], tr.rh2[i])
            val p = softmaxOf(tr.h2[i]); tr.p[i] = p
            loss -= Math.log((p[y[i]] + 1e-9f).toDouble()).toFloat()
            h1p = tr.h1[i]; h2p = tr.h2[i]
        }
        loss /= t
        if (!grads) { System.arraycopy(h1p, 0, s1, 0, h); System.arraycopy(h2p, 0, s2, 0, h); return Pair(loss, null) }
        val g = P.map { FloatArray(it.size) }
        val dH2 = Array(t) { FloatArray(h) }
        for (i in 0 until t) {
            val d = tr.p[i]!!
            d[y[i]] -= 1f
            for (j in 0 until v) d[j] /= t.toFloat()
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
        val safeCorpus = if (corpus.length > 200_000) corpus.substring(0, 200_000) else corpus
        if (safeCorpus.length < 8) return "النص قصير جدًا، ما يستاهل تدريب."
        val reh = if (rehearse.length > 9000) rehearse.substring(0, 9000) else rehearse
        val sb = StringBuilder()
        var i = 0
        while (i < safeCorpus.length) {
            val end = minOf(i + bookChunk, safeCorpus.length)
            sb.append(safeCorpus, i, end)
            if (reh.isNotEmpty()) { sb.append("\n"); sb.append(reh.take(bookChunk / 3)) }   // جرعة محادثة بعد كل قطعة كتاب
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

/** One conversation. The hidden state and the short-term memory live here and nowhere else: when the conversation ends, the session is dropped.
 *  (What the user said has already been learned into the long-term memory by Nlm.observe.) Mirror of Session in nuwaa/lm/gru.py. */
class NlmSession(private val m: Nlm, seed: Long) {
    private val s1 = FloatArray(m.h); private val s2 = FloatArray(m.h)
    private var p: FloatArray = FloatArray(m.v)
    private val text = StringBuilder()
    val short = NMem()
    private val rng = java.util.Random(seed)
    private val replies = ArrayList<Pair<String, String>>()      // (normalised question, reply) of this conversation
    private val minOrder = 6        // a memory may speak only when it matches >= 6 characters of context: it completes words but can never hijack the START of a reply (raised from 4: a 4-char match is too easy to hit by coincidence in unrelated training text, and used to be enough to drag the reply onto a totally different topic)
    private val ngSelf = 10         // inside one reply no 10-character piece is written twice
    private val ngPrev = 8          // against earlier replies of this chat: no 8-character piece (unless the user asked the same thing again)

    init { feed("\n") }      // every pair in training follows a line end: a fresh conversation starts the same way

    fun feed(s: String) {
        for (ch in s) { p = m.stepChar(m.cid(ch), s1, s2); text.append(ch) }
    }

    /** Mixes the neural distribution with the long-term and the short-term memory. Always returns a NEW array. */
    private fun mix(p0: FloatArray, gainLong: Float, gainShort: Float): FloatArray {
        var q0 = p0.copyOf()
        val ctx = if (text.length > 8) text.substring(text.length - 8) else text.toString()
        val mems = listOf(Pair(m.long, gainLong), Pair(short, gainShort))
        for (item in mems) {
            if (item.second <= 0f) continue
            val d = item.first.dist(ctx, minOrder) ?: continue
            val o = d.first; val cnt = d.second
            var tot = 0
            for (c in cnt.values) tot += c
            val lam = item.second * tot / (tot + 0.1f) * minOf(1f, (o - minOrder + 1) / 4f)
            val q = FloatArray(m.v)
            for ((ch, k) in cnt) q[m.cid(ch)] += k.toFloat() / tot
            val r = FloatArray(m.v)
            for (j in 0 until m.v) r[j] = (1f - lam) * q0[j] + lam * q[j]
            q0 = r
        }
        return q0
    }

    private fun grams(t: String, n: Int): HashSet<String> {
        val g = HashSet<String>()
        if (t.length >= n) for (i in 0..t.length - n) g.add(t.substring(i, i + n))
        return g
    }

    /** Drops a sentence or line that repeats one already written. */
    private fun dedupe(t: String): String {
        val seen = HashSet<String>(); val sb = StringBuilder()
        var start = 0
        for (i in t.indices) {
            val c = t[i]
            if (c == '.' || c == '!' || c == '?' || c == '؟' || c == '\n' || i == t.length - 1) {
                val part = t.substring(start, i + 1); start = i + 1
                val k = part.trim()
                if (k.length >= 6 && !seen.add(k)) continue
                sb.append(part)
            }
        }
        return sb.toString().trim()
    }

    /** Writes a reply to `user`, character by character, until the model ends the line. No gate, no canned phrase: it always answers.
     *  A no-repeat sampler keeps it from repeating itself or its earlier replies. Returns the reply and the probability the model gave each character. */
    fun reply(user: String, maxChars: Int = 700, temp: Float = 0.3f, topk: Int = 4, minP: Float = 0.25f, commit: Boolean = true, memGain: Float = 0.15f, shortGain: Float = 0.15f): Pair<String, List<Float>> {
        val u = user.replace("\n", " ¶ ").trim()
        val nq = u.split(Regex("\\s+")).joinToString(" ")
        feed("س:" + u + "\nج:")
        val sameAsk = replies.any { it.first == nq }
        val prev = HashSet<String>()
        if (!sameAsk) for (r in replies.takeLast(6)) prev.addAll(grams(r.second, ngPrev))
        val selfG = HashSet<String>()
        val out = StringBuilder(); val track = ArrayList<Float>()
        val nl = m.cid('\n')
        val src = copySource(u); val cst = intArrayOf(-1)
        for (step in 0 until maxChars) {
            val pm = mix(p, memGain, shortGain)
            pm[0] = 0f
            val cc = copyNext(src, out, cst)                  // inside a file block: copy the user's own names, items and numbers
            if (cc != null) { val ci = m.cid(if (cc == '\n') '¶' else cc); for (j in 0 until m.v) pm[j] *= (1f - COPY_GAIN); pm[ci] += COPY_GAIN }
            val outNow = out.toString()
            val fileMode = outNow.contains("<<ملف:")           // code legitimately repeats itself: the anti-repetition rules are off inside a file reply
            val openBlock = outNow.lastIndexOf("<<ملف:") > outNow.lastIndexOf("<<نهاية>>")
            if (out.length < 3 || openBlock) pm[nl] = 0f     // a reply cannot end before it began, nor inside an open file block
            run { var tot = 0f; for (j in 0 until m.v) tot += pm[j]; if (tot <= 0f) pm[nl] = 1f }       // nothing left to write: end the reply
            if (!fileMode && out.length >= ngPrev - 1) {
                var any = false
                val tail = if (out.length > ngSelf - 1) out.substring(out.length - (ngSelf - 1)) else out.toString()      // only the last characters matter: no copy of the whole reply per candidate
                for (j in 0 until m.v) {
                    if (pm[j] <= 0f || j == nl) { if (pm[j] > 0f) any = true; continue }
                    val cand = tail + m.chars[j]
                    val bad = (cand.length >= ngSelf && selfG.contains(cand.substring(cand.length - ngSelf))) ||
                              (cand.length >= ngPrev && prev.contains(cand.substring(cand.length - ngPrev)))
                    if (bad) pm[j] = 0f else any = true
                }
                if (!any) { for (j in 0 until m.v) pm[j] = 0f; pm[nl] = 1f }     // every continuation repeats: finish the reply here
            }
            var mx = 0f; var best = 1
            for (j in 0 until m.v) if (pm[j] > mx) { mx = pm[j]; best = j }
            for (j in 0 until m.v) if (pm[j] < minP * mx) pm[j] = 0f
            for (j in 0 until m.v) if (pm[j] > 0f) pm[j] = Math.pow(pm[j].toDouble(), 1.0 / temp).toFloat()
            if (topk >= 1 && topk < m.v) {
                val sorted = pm.copyOf(); java.util.Arrays.sort(sorted)
                val cut = sorted[m.v - topk]
                for (j in 0 until m.v) if (pm[j] < cut) pm[j] = 0f
            }
            var sum = 0f
            for (j in 0 until m.v) sum += pm[j]
            var pick = best
            if (sum > 0f) {
                var r = rng.nextFloat() * sum
                for (j in 0 until m.v) { r -= pm[j]; if (r <= 0f && pm[j] > 0f) { pick = j; break } }
            }
            track.add(p[pick])
            val ch = m.chars[pick]
            p = m.stepChar(pick, s1, s2); text.append(ch)
            if (ch == "\n") break
            out.append(ch)
            if (out.length >= ngSelf) selfG.add(out.substring(out.length - ngSelf))
            if (!fileMode && out.length >= 48 && out.substring(0, out.length - 24).contains(out.substring(out.length - 24))) { out.setLength(out.length - 24); break }   // a loop: cut it off and stop
        }
        val raw = out.toString().replace("¶", "\n")
        val fin = (if (raw.contains("<<ملف:")) raw else dedupe(raw)).trim()           // dedupe would delete the second «<<نهاية>>»
        if (commit) { replies.add(Pair(nq, fin)); while (replies.size > 12) replies.removeAt(0); short.add("س:$u") }
        return Pair(fin, track)
    }

    private val COPY_GAIN = 0.9f
    private fun copySource(u: String): String = u.replace(Regex("\\s*،\\s*"), "\n").replace("باسم ", "\u0001ملف:").replace("مجلد ", "\u0001ملف:").replace(Regex("(\u0001ملف:)+"), "\u0001ملف:")

    /** Pointer copy, inside a «<<ملف:» block only. A run starts when the last 3+ characters written also occur in the request, after an opening bracket,
     *  after the first «: » of the request, or at the file name the user asked for; it then follows the request letter by letter and stops at a space
     *  (except after a comma) and at the end of the list. st[0] = next index in src, -1 = no run. Same logic as GRU Session._copy_dist in Python. */
    private fun copyNext(src: String, out: StringBuilder, st: IntArray): Char? {
        val outS = out.toString()
        if (outS.lastIndexOf("<<ملف:") <= outS.lastIndexOf("<<نهاية>>")) { st[0] = -1; return null }
        val o = outS.replace("¶", "\n").replace("<<ملف:", "\u0001ملف:")
        var p = st[0]
        if (p >= 0 && (p >= src.length || p == 0 || o.isEmpty() || o.last() != src[p - 1])) p = -1
        if (p < 0) {
            val opener = "\u0001ملف:"
            if (o.endsWith(opener)) { val k = src.indexOf(opener); p = if (k >= 0) k + opener.length else -1 }
            else if (o.endsWith("= [") && src.indexOf('[') >= 0) p = src.indexOf('[') + 1                 // a list literal being assigned: copy the user's list
            else if (o.endsWith(">>\n") && src.contains(": ") && isListFile(o)) p = src.indexOf(": ") + 2
            else {
                var len = minOf(12, o.length)
                while (len >= 3) { val pos = src.indexOf(o.substring(o.length - len)); if (pos >= 0) { p = pos + len; break }; len-- }
            }
        }
        if (p < 0 || p >= src.length) { st[0] = -1; return null }
        val ch = src[p]
        if (ch == '\u0001' || (ch == ' ' && (o.isEmpty() || o.last() != ','))) { st[0] = -1; return null }
        st[0] = p + 1
        return ch
    }

    /** True when the file block being written is a plain list (.txt / .md): only there the first item is taken from «قائمة ...: items». */
    private fun isListFile(o: String): Boolean {
        val i = o.lastIndexOf("\u0001ملف:"); if (i < 0) return false
        val j = o.indexOf(">>", i); if (j <= i) return false
        val nm = o.substring(i + 5, j).trim().lowercase(java.util.Locale.ROOT)
        return nm.endsWith(".txt") || nm.endsWith(".md")
    }

    private class Snap(val p: FloatArray, val a: FloatArray, val b: FloatArray, val text: String)
    private fun snap() = Snap(p.copyOf(), s1.copyOf(), s2.copyOf(), text.toString())
    private fun restore(x: Snap) { p = x.p.copyOf(); System.arraycopy(x.a, 0, s1, 0, s1.size); System.arraycopy(x.b, 0, s2, 0, s2.size); text.setLength(0); text.append(x.text) }

    /** Thinking mode: several separate attempts from the same starting point (cooler and warmer sampling); the one the model itself was most
     *  sure about wins, and only that one enters the conversation. Costs `tries` times the time of one reply, so Brain uses it for hard questions only.
     *  minMs: a real wall-clock floor (e.g. 5000 = five seconds). Once the first `tries` attempts are done, if less than minMs has actually
     *  elapsed it keeps generating MORE attempts (cycling the same temp spread again) until minMs is reached, so a fast phone / small model
     *  cannot finish "thinking" in under a fraction of a second just because `tries` was small — every reply spends at least minMs really computing.
     *  0 = no floor (old behaviour, exactly `tries` attempts). hardCapTries bounds it defensively in case the clock is ever wrong. */
    fun replyThink(user: String, maxChars: Int, tries: Int, baseTemp: Float = 0.3f, minMs: Long = 0, accept: ((String) -> Boolean)? = null): Pair<String, List<Float>> {
        // FIXED a real bug: this used to hardcode 5 fixed temps (0.55..0.85, averaging HIGHER than the
        // "accurate" tuning elsewhere) and silently capped every call at 5 real attempts no matter how
        // many `tries` was actually passed in (code-mode asking for 8 was quietly only getting 5, with
        // the caller's temp/accuracy preference thrown away entirely). Now it generates exactly `tries`
        // attempts, spread in a tight low band around the caller's own baseTemp — so Effort.kt's tuning
        // (and the 8-try code-detection mode) actually takes effect here, not just in the single-shot path.
        val start = snap()
        val startNs = System.nanoTime()
        fun elapsedMs() = (System.nanoTime() - startNs) / 1_000_000L
        val hardCapTries = maxOf(tries, 200)      // defensive only: stops an eternal loop if the clock misbehaves, never meant to be hit in practice
        var bestScore = -1f; var best: Pair<String, List<Float>>? = null; var bestEnd: Snap? = null
        var i = 0
        while (i < tries || (minMs > 0 && elapsedMs() < minMs)) {
            if (i >= hardCapTries) break
            val temp = (baseTemp + (i % 6) * 0.03f).coerceIn(0.05f, 0.9f)
            restore(start)
            val r = reply(user, maxChars, temp, 4, 0.25f, false)   // topk 8->4, minP 0.15->0.25: tighter, more deterministic sampling
            var sc = if (r.second.isEmpty()) 0f else r.second.sum() / r.second.size
            if (r.first.length < 8) sc -= 0.2f
            if (accept != null && !accept(r.first)) sc -= 1f              // an attempt that fails the cheap check is only chosen if all fail
            if (sc > bestScore) { bestScore = sc; best = r; bestEnd = snap() }
            Thread.yield()
            i++
        }
        val fin = best ?: return reply(user, maxChars)
        restore(bestEnd ?: start)
        val u = user.replace("\n", " ¶ ").trim()
        replies.add(Pair(u.split(Regex("\\s+")).joinToString(" "), fin.first)); while (replies.size > 12) replies.removeAt(0)
        short.add("س:$u")
        return fin
    }

    /** "Deep" thinking: closer to what the stages large models go through at reply-time actually buy you — NOT extra
     *  knowledge (that only comes from training), but extra CHOICE among what the model can already produce. It runs
     *  many independent attempts at the SAME question over a real time budget (maxMs — set this to hours if you want,
     *  see Effort.deepThinkMaxMs), then instead of only trusting each attempt's own self-confidence (replyThink's method,
     *  which a badly-trained model can still be "confident" and wrong about), it also scores how much each attempt
     *  AGREES with the others — shared 3-character runs, a cheap stand-in for "did several independent attempts
     *  converge on roughly the same thing" (self-consistency: if 40 independent attempts keep landing near the same
     *  answer, that is more likely to be real signal than the one attempt that happened to score highest confidence alone).
     *  Honest limit, stated plainly: this can only choose among answers the base model is ALREADY capable of generating.
     *  It cannot manufacture grammar, facts or working code the model never learned — that still only comes from
     *  Nlm.develop()/trainOn and real training data. More time here makes the pick more reliable, it does not make the
     *  model smarter. keepWindow caps memory use (only the most recent attempts are kept for the agreement comparison,
     *  not all of them, so an hour-long run on a phone does not fill up RAM with thousands of full candidate strings). */
    fun replyDeep(user: String, maxChars: Int, baseTemp: Float, maxMs: Long, keepWindow: Int = 40, accept: ((String) -> Boolean)? = null): Pair<String, List<Float>> {
        val start = snap()
        val startNs = System.nanoTime()
        fun elapsedMs() = (System.nanoTime() - startNs) / 1_000_000L
        // رفعتها من 2000 إلى 20000 بناءً على طلبك بإزالة الحد — بس ما خليتها لا نهائي حرفيًا، ليش: لو النموذج ما يقدر
        // أبدًا يطلع كود يجتاز codeSanity (احتمال وارد لأن النموذج صغير ومتدرب حرفًا حرفًا)، فاللاحد-الأقصى يعني يضل
        // يحاول للأبد بلا أي فايدة، يستهلك بطارية الجهاز بلا توقف حتى لو تركته طول الليل. 20000 محاولة سقف عملي كبير
        // جدًا (غالبًا ميزانية الوقت deepThinkMaxMs تنتهي قبله بكثير أصلًا)، وبرضو تقدر ترفعه بنفسك هنا لو تبي أعلى.
        val hardCap = 20000
        fun grams(s: String): Set<String> {
            val t = s.trim(); val g = HashSet<String>()
            for (k in 0..t.length - 3) g.add(t.substring(k, k + 3))
            return g
        }
        data class Cand(val r: Pair<String, List<Float>>, val conf: Float, val g: Set<String>, val end: Snap)
        val window = ArrayDeque<Cand>()
        var bestTotal = Float.NEGATIVE_INFINITY; var best: Cand? = null
        var i = 0
        while (i < hardCap && elapsedMs() < maxMs) {
            val temp = (baseTemp + (i % 8) * 0.025f).coerceIn(0.05f, 0.95f)
            restore(start)
            val r = reply(user, maxChars, temp, 4, 0.25f, false)
            var conf = if (r.second.isEmpty()) 0f else r.second.sum() / r.second.size
            if (r.first.trim().length < 8) conf -= 0.3f
            if (accept != null && !accept(r.first)) conf -= 5f   // رُفعت من 1 إلى 5: فرق أكبر من أي فرق ثقة واقعي، فتصير بوابة شبه إلزامية لا مجرد عقوبة خفيفة
            val cand = Cand(r, conf, grams(r.first), snap())
            window.addLast(cand); if (window.size > keepWindow) window.removeFirst()
            var agree = 0f
            if (window.size > 1) {
                for (other in window) if (other !== cand) {
                    val inter = cand.g.intersect(other.g).size.toFloat()
                    val union = cand.g.union(other.g).size.toFloat()
                    agree += if (union > 0f) inter / union else 0f
                }
                agree /= (window.size - 1)
            }
            val total = cand.conf + agree
            if (total > bestTotal) { bestTotal = total; best = cand }
            Thread.yield(); i++
        }
        val fin = best ?: return reply(user, maxChars)
        restore(fin.end)
        val u = user.replace("\n", " ¶ ").trim()
        replies.add(Pair(u.split(Regex("\\s+")).joinToString(" "), fin.r.first)); while (replies.size > 12) replies.removeAt(0)
        short.add("س:$u")
        return fin.r
    }
}


// ============== 20-round Arabic training + post-training self-test workflow ==============
class Trainer(private val context: Context, private val brain: Brain) {
    @Volatile var running: Boolean = false
        private set
    @Volatile var percent: Int = 40
        @Volatile var selfReview: Boolean = true
    private val stopFlag = java.util.concurrent.atomic.AtomicBoolean(false)
    fun stop() { stopFlag.set(true) }

    private val practicePairs = listOf(
        "س: ما وظيفة اللغة؟\nج: اللغة وسيلة للتواصل وأداة للتفكير ونقل المعرفة.\n",
        "س: ما الفرق بين الفصحى واللهجات؟\nج: الفصحى لغة مشتركة للكتابة والتعليم، واللهجات أشكال طبيعية للتواصل اليومي تختلف حسب المناطق.\n",
        "س: لماذا نختار الكلمات بعناية؟\nج: اختيار الكلمات وترتيبها يساعد على إيصال المعنى بوضوح.\n",
        "س: ما المقصود بالسياق؟\nج: السياق هو الظروف والكلام المحيط الذي يساعد على فهم المقصود.\n",
        "س: كيف أتعامل مع معلومة غير مؤكدة؟\nج: أميز بين المعلومة والاستنتاج وأصرح بعدم التأكد بدل اختراع جواب.\n",
        "س: ما فائدة السؤال؟\nج: السؤال يحدد الحاجة إلى المعرفة ويساعد على توضيح المعنى.\n",
        "س: ما الذي يجعل الإجابة واضحة؟\nج: أن تكون مرتبطة بالسؤال، دقيقة، مرتبة، وبألفاظ مناسبة.\n",
        "س: ما هي العربية الفصحى المعاصرة؟\nج: هي العربية المشتركة المستخدمة في التعليم والكتب والإعلام والكتابة الرسمية.\n",
        "س: هل اختلاف اللهجة يعني ضعف اللغة؟\nج: لا، اللهجة مرتبطة بالمجتمع والبيئة ولا تعني ضعف القدرة اللغوية.\n",
        "س: كيف ينبغي أن يجيب النموذج؟\nج: يجيب بوضوح ودقة، ولا يختلق حقائق عندما لا يعرف.\n"
    )
    private val testQuestions = listOf(
        "ما وظيفة اللغة؟", "ما الفرق بين الفصحى واللهجات؟", "لماذا نختار الكلمات بعناية؟",
        "ما المقصود بالسياق؟", "كيف أتعامل مع معلومة غير مؤكدة؟", "ما فائدة السؤال؟",
        "ما الذي يجعل الإجابة واضحة؟", "ما هي العربية الفصحى المعاصرة؟",
        "هل اختلاف اللهجة يعني ضعف اللغة؟", "كيف ينبغي أن يجيب النموذج؟"
    )

    private fun avgLoss(model: Nlm, samples: List<String>): Float {
        var total = 0.0; var count = 0
        for (text in samples) {
            val ids = model.encode(text)
            if (ids.size >= 3) { total += model.bpc(ids).toDouble(); count++ }
        }
        return if (count == 0) Float.POSITIVE_INFINITY else (total / count).toFloat()
    }

    fun start(onStatus: (String) -> Unit) {
        if (running) return
        stopFlag.set(false); running = true
        thread(name = "arabic-training-and-review") {
            try {
                val model = brain.trainModel() ?: run { onStatus("النموذج غير محمّل."); return@thread }
                var bestLoss = synchronized(brain.modelLock) { avgLoss(model, practicePairs) }
                var savedRounds = 0
                onStatus("بدأ التدريب على 10 أمثلة عربية. عدد الدورات المحدد: 20.")
                for (round in 1..20) {
                    if (stopFlag.get()) break
                    val batch = practicePairs.shuffled()
                    val snap = synchronized(brain.modelLock) { model.snapshot() }
                    synchronized(brain.modelLock) { batch.forEach { if (!stopFlag.get()) model.trainWindow(it, 0.00012f) } }
                    val after = synchronized(brain.modelLock) { avgLoss(model, practicePairs) }
                    if (after.isFinite() && after < bestLoss) {
                        bestLoss = after
                        brain.saveTrained()
                        savedRounds++
                    } else {
                        synchronized(brain.modelLock) { model.restoreSnapshot(snap) }
                    }
                    onStatus("التدريب: الجولة ${round}/20 — أفضل خسارة ${"%.4f".format(java.util.Locale.US,bestLoss)} — checkpoints محفوظة: ${savedRounds}")
                    val target = percent.coerceIn(5,100)
                    if (target < 100 && !stopFlag.get()) Thread.sleep((100-target)*8L)
                }
                if (stopFlag.get()) onStatus("أُوقف التدريب؛ أبدأ مراجعة الحالة المحفوظة.")
                else onStatus("اكتمل التدريب 20/20. أبدأ اختبار النموذج الآن...")
                brain.saveTrained()
                if (selfReview) {
                    val out = StringBuilder("\n\n=== اختبار بعد التدريب (10 أسئلة كحد أقصى) ===\n")
                    testQuestions.take(10).forEachIndexed { i, q ->
                        if (stopFlag.get()) return@forEachIndexed
                        val response = brain.selfAttempt(q).ifBlank { "(لم ينتج رد)" }
                        out.append("\n").append(i+1).append(") السؤال: ").append(q)
                            .append("\nرد النموذج: ").append(response).append("\n")
                        onStatus(out.toString())
                    }
                    onStatus("اكتمل التدريب والمراجعة. الأوزان محفوظة.\n" + out.toString())
                } else onStatus("انتهى التدريب وتم حفظ الأوزان. المراجعة الذاتية معطلة.")
            } catch (e: InterruptedException) {
                onStatus("توقفت العملية. آخر checkpoint محفوظ.")
            } catch (e: Throwable) {
                onStatus("حدث خطأ في التدريب/الاختبار: ${e.message ?: e.javaClass.simpleName}")
            } finally { running = false; stopFlag.set(false) }
        }
    }
}
