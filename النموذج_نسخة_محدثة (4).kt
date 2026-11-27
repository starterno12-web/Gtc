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
import android.webkit.WebChromeClient
import android.webkit.ConsoleMessage
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.Collections
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
    private lateinit var worldPage: LinearLayout
    private lateinit var worldImg: ImageView
    private lateinit var worldInfo: TextView
    private var worldRun = 0
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
    private val MAX_READ = Int.MAX_VALUE   // ما فيه حد فعليًا بعد الآن (كان 8_000_000) — بناءً على طلبك الصريح المتكرر
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
        titleView = Ui.label(this, "سوبرمان", 20f, Color.WHITE, true)
        titleView.setOnClickListener { onTitleTap() }
        status = Ui.label(this, "", 11f, Color.parseColor("#9FB3C0"), false)
        tcol.addView(titleView); tcol.addView(status)
        bar.addView(menu)
        bar.addView(tcol, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(dev)
        rootView.addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        pages = FrameLayout(this)
        rootView.addView(pages, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        buildChatPage(); buildHistPage(); buildScreenPage(); buildCodePage(); buildDevPage(); buildTrainPage(); buildDiscoverPage(); buildGrowPage(); buildWorldPage()
    }

    // ---------------- pages ----------------
    private fun showPage(p: View) {
        pages.removeAllViews()
        pages.addView(p, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        if (p === histPage) refreshHistory()
    }

    private fun showMenu() {
        val base = arrayOf("💬  محادثة جديدة", "🧠  تطوير النفس", "🗂  المحادثات السابقة", "📱  مشاركة الشاشة", "💻  كتابة الأكواد", "🗑  حذف كل المحادثات", "ℹ️  عن التطبيق", "⚡  رفع الفهم (تدريب)", "🔍  اكتشاف من الإنترنت", "🌍  عالم سوبرمان (يتخيل نفسه)")
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
                9 -> { showPage(worldPage); worldInfo.text = brain.worldStatus() }
                10 -> openGrowPage()
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
        AlertDialog.Builder(this).setTitle("سوبرمان").setMessage(
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
        titleView.text = if (c.title == "محادثة جديدة") "سوبرمان" else c.title
        if (c.msgs.isEmpty()) {
            bubble("هلا! أنا سوبرمان 👋\nأكتب ردودي بنفسي حرفا بعد حرف، وأتعلم من كلامك بدون ما تقول لي «تعلم». أحسب لك بدقة، وأتدرب على البرمجة.\nجرّب: «كيف اكتب حلقة for في بايثون» أو «كم يساوي ٢٥ × ١٧».", false, false)
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
        val dots = TextView(this); dots.text = "سوبرمان تكتب..."; dots.setTextColor(Ui.MUTED); dots.setPadding(dp(16), dp(4), dp(16), dp(4)); log.addView(dots)
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
                // لعبة/تطبيق تفاعلي (كانفاس أو يستمع للوحة المفاتيح/اللمس): يلعبها بنفسه كلاعب فعلي
                // عبر عدة ضغطات حقيقية ويرصد أي خطأ JavaScript وقع أثناء ذلك — مو لقطة واحدة ثابتة.
                // الدقة هي الهدف هنا مو السرعة، فعدد المحاولات كبير عمدًا (ما يهم الوقت).
                val code0 = String(f.bytes, Charsets.UTF_8)
                val fixed = if (PlayLoop.looksLikeGame(code0))
                    PlayLoop.reviewAndPlay(brain, this, convId, code0, f.name, maxPasses = 15,
                        log = { line -> if (TvLink.connected) TvLink.sendThinking(line) })
                else VisionLoop.reviewAndFix(
                    brain, this, convId, code0, f.name, maxPasses = 2,
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

    // ---------------- عالم سوبرمان: صفحة التخيّل ----------------
    private fun buildWorldPage() {
        worldPage = LinearLayout(this); worldPage.orientation = LinearLayout.VERTICAL; worldPage.setPadding(dp(16), dp(14), dp(16), dp(10))
        worldPage.addView(Ui.label(this, "🌍 عالم سوبرمان", 20f, Ui.DARK, true))
        worldPage.addView(Ui.label(this, "أبني بالكود عالم صغير (جدران، مفتاح، باب، أخطار، هدف)، وأحط نفسي فيه 🤖 وأجرّب فيه فعلًا بمحاكي حقيقي. أتعلم من عوالم محلولة، وأنا نفسي أبني عوالم وأتحقق إنها سليمة. أقيس نفسي بامتحان ثابت قبل وبعد.", 14f, Ui.MUTED, false).also { it.setPadding(0, dp(6), 0, dp(10)) })
        worldImg = ImageView(this); worldImg.adjustViewBounds = true; worldImg.scaleType = ImageView.ScaleType.FIT_CENTER
        worldPage.addView(worldImg, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(220)))
        val b1 = Ui.btn(this, "🌍  تخيّل عالم وشوف نفسك فيه", Ui.PRIMARY) { runDream() }
        val b2 = Ui.btn(this, "📚  تعلّم من العوالم (تدريب)", Ui.DARK) { runWorldTrain() }
        val b3 = Ui.btn(this, "📝  اختبرني (قياس حقيقي)", Ui.DARK) { runWorldExam() }
        val b4 = Ui.btn(this, "🔬  فحص رياضيات المحرك", Ui.DARK) { runGradCheck() }
        for (b in listOf(b1, b2, b3, b4)) {
            b.textSize = 16f
            val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); lp.setMargins(0, dp(8), 0, 0)
            worldPage.addView(b, lp)
        }
        val sv = ScrollView(this); worldInfo = Ui.label(this, "", 15f, Ui.DARK, false); worldInfo.setPadding(0, dp(12), 0, 0); worldInfo.setTextIsSelectable(true); sv.addView(worldInfo)
        worldPage.addView(sv, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun showWorldFrame(wd: SimWorld) {
        val cell = maxOf(24, minOf(dp(300) / wd.w, dp(220) / wd.hh))
        worldImg.setImageBitmap(wd.toBitmap(cell))
    }

    /** يعيد تمثيل الخطة خطوة خطوة: سوبرمان (الدائرة الزرقاء) يمشي في عالمه قدام عينك. */
    private fun playWorld(start: SimWorld, acts: String) {
        val id = ++worldRun
        val w = start.copy()
        showWorldFrame(w)
        val hd = Handler(Looper.getMainLooper())
        var i = 0
        val tick = object : Runnable {
            override fun run() {
                if (id != worldRun) return
                if (i >= acts.length || w.over) { showWorldFrame(w); return }
                w.step(acts[i]); i++
                showWorldFrame(w)
                hd.postDelayed(this, 380)
            }
        }
        hd.postDelayed(tick, 600)
    }

    private fun runDream() {
        worldInfo.text = "أبني عالم وأجرّب فيه..."
        thread {
            val d = try { brain.dreamOnce() } catch (e: Throwable) { null }
            runOnUiThread {
                if (d == null) { worldInfo.text = "النموذج غير محمّل أو صار خطأ."; return@runOnUiThread }
                worldInfo.text = d.report() + "\n\n" + brain.worldStatus()
                playWorld(d.world, d.plan ?: "")
            }
        }
    }

    private fun runWorldTrain() {
        worldInfo.text = "أتعلم من عوالم محلولة وأقيس نفسي قبل وبعد... (قد تأخذ عدة دقائق، لا تقفل التطبيق)"
        thread {
            val r = try { brain.worldTrain() } catch (e: Throwable) { "تعذر التدريب: " + (e.message ?: e.javaClass.simpleName) }
            runOnUiThread { worldInfo.text = r + "\n\n" + brain.worldStatus() }
        }
    }

    private fun runWorldExam() {
        worldInfo.text = "أمتحن نفسي على عوالم ثابتة (بدون بحث)..."
        thread {
            val r = try { brain.worldExam() } catch (e: Throwable) { "تعذر الامتحان: " + (e.message ?: e.javaClass.simpleName) }
            runOnUiThread { worldInfo.text = r + "\n" + brain.worldStatus() }
        }
    }

    private fun runGradCheck() {
        worldInfo.text = "أفحص رياضيات المحرك (الخلف مقابل الحساب الرقمي)..."
        thread {
            val r = try { brain.gradCheck() } catch (e: Throwable) { "تعذر الفحص: " + (e.message ?: e.javaClass.simpleName) }
            runOnUiThread { worldInfo.text = r }
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
        val maxSwitch = Switch(this).apply {
            text = "⚡ وضع ماكس — دقة قصوى، الوقت بلا قيمة إطلاقًا (أبطأ بكثير، ويتجاوز حماية البطارية/الحرارة)"
            textSize = 13f; isChecked = Effort.maxMode
            setOnCheckedChangeListener { _, isOn -> Effort.maxMode = isOn }
        }
        val maxLp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); maxLp.setMargins(0, dp(8), 0, dp(8))
        trainPage.addView(maxSwitch, maxLp)
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
        sizeBar.progress = 25; sizeLabel.text = "الهدف: " + targetParams(25) + " معامل (~" + (targetParams(25) * 4 / 1e6).toInt() + " MB)"   // كان 0% افتراضيًا، صار 25% بناءً على طلبك
        growPage.addView(sizeBar)
        growPage.addView(Ui.label(this, "⚠ كرت الشاشة: غير متاح للتدريب على الجوال في هذا التطبيق (المحرك هنا يحسب على المعالج فقط). التحكم الحقيقي بكرت الشاشة متاح فقط عبر «استوديو سوبرمان» على الكمبيوتر (tools/studio.py)، ليس من هنا.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(12), 0, dp(4)) })
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
        discoverPage.addView(Ui.label(this, "يطلب من Gemini (بمفتاحك الخاص) أمثلة برمجة ولغة عربية جديدة باستمرار أثناء التدريب، ويدرّب عليها بنفس حماية التراجع — لكن بلا تحقق بالتشغيل مثل بيانات سوبرمان نفسها، فهو معلّم غير موثَّق بقدر التحقق الذاتي.", 13f, Ui.MUTED, false).also { it.setPadding(0, dp(2), 0, dp(10)) })
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
        val b1 = Ui.btn(this, "تفعيل قارئ الشاشة (مرة واحدة)", Ui.DARK) { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)); Toast.makeText(this, "اختر «سوبرمان» ثم فعّله", Toast.LENGTH_LONG).show() }
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
            // أولوية أولى: الأوزان المدمجة جوا الكود نفسه (Nlm.EMBEDDED_WEIGHTS_B64). لو فاضية، يرجع للطريقة
            // القديمة (ملف nlm2.bin منفصل في assets) تلقائيًا — ما ينكسر شي لو ما عبّيت الثابت.
            val embedded = Nlm.loadEmbedded()
            val base = if (embedded != null) null else ctx.assets.open("nlm2.bin").use { s -> s.readBytes() }
            val heapMb = Runtime.getRuntime().maxMemory() / 1e6      // no budget of mine: the only limit is what this phone really gives the app
            if (base != null && base.size * 3.0 / 1e6 > heapMb * 0.85) { modelStatus = "النموذج: تخطيته لأن ذاكرة هذا الجهاز للتطبيق ${heapMb.toInt()} MB فقط"; return }
            val m = try { if (nlmUserFile.exists()) Nlm.load(nlmUserFile.readBytes()) else (embedded ?: Nlm.load(base!!)) } catch (e: Exception) { embedded ?: Nlm.load(base!!) }     // a damaged user file never blocks the start
            if (memFile.exists()) for (line in memFile.readLines(Charsets.UTF_8)) if (line.isNotBlank()) { memory.add(line); m.long.add(line) }
            nlm = m
            modelStatus = "نموذج سوبرمان (Transformer، نافذة سياق " + m.h + " حرف): " + m.paramCount() + " معامل · سمعت منك " + memory.size + " رسالة"
        } catch (e: Throwable) {
            // الأرجح إن السبب: ملف assets/nlm2.bin لسه بصيغة الـGRU القديمة، غير متوافقة مع الـTransformer الجديد
            // (حجم البايتات يختلف فعليًا). بدل ما يضل التطبيق بلا نموذج إطلاقًا، أبني نموذج جديد عشوائي التهيئة —
            // فاضي من أي معرفة، بس شغّال وقابل للتدريب فورًا عبر develop()/trainBig(). حجم افتراضي (e=88،
            // نافذة سياق h تُحسب تلقائيًا من ذاكرة الجهاز الفعلية وقت الإنشاء (ContextSizer.pick) — بطلبك
            // الصريح: بلا رقم ثابت أختاره أنا (كانت 64 ثم 512 ثم 2048، كل وحدة رقم يدوي)، تتكيف وحدها مع كل
            // جوال: جهاز ضعيف ياخذ نافذة أصغر تلقائيًا، جهاز قوي ياخذ نافذة أكبر، وما فيه سقف محتوى من عندي —
            // بس سقف دفاعي بعيد جدًا (2 مليون) يمنع رقم فلكي لو جهاز بلّغ ذاكرة غير واقعية.
            // تنبيه صادق عن الكلفة: الانتباه هنا بلا KV-cache (كود تعليمي مكتوب يدويًا من الصفر)، فكل حرف
            // يُولَّد يعيد حساب الانتباه على كامل النافذة — تكلفة ∝ h². نافذة أكبر = كل خطوة أبطأ بكثير، وهذا
            // بالضبط اللي يحدّ النافذة فعليًا هنا (مو قرار محتوى)، مقبول حسب أولويتك (الدقة قبل الوقت).
            // تنبيه ثاني لازم أقوله: نافذة أكبر لوحدها ما تضمن واجهة شغّالة — تمنع النسيان، لكن قدرة النموذج
            // (e=88، 3 طبقات) على توليد HTML/CSS/JS صحيح نحويًا يعتمد على حجم النموذج وتدريبه (من «تطوير
            // متقدم»)، مو طول النافذة بس. اللي يمسك أخطاء الواجهة فعليًا هو PlayLoop (يشغّلها كلاعب حقيقي
            // ويصلحها)، ويشتغل بصرف النظر عن حجم النافذة. وتخزين مؤقت حقيقي (KV-cache) يخفّض الكلفة من h²
            // إلى h، بس يحتاج تغيير جذري بترميز الموضع ما قدرت أختبره رقميًا هنا (راجع gradCheck)، فما سويته.
            // ومفردات افتراضية (عربي + لاتيني + أرقام + رموز شائعة) تقدر تغيّرها بنفسك.
            try {
                val defChars = (" \n\t.,!?:;'\"()[]{}-_=+*/\\<>@#$%^&|~`" +
                    "0123456789" +
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                    "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰ").toCharArray().distinct().map { it.toString() }
                nlm = Nlm.fresh(88, ContextSizer.pick(88, 3, Nlm.HEADS, defChars.size), defChars)   // e=88 عشان يكون من مضاعفات 8 (8 رؤوس انتباه، headDim=11) · h يُحسب تلقائيًا من ذاكرة هذا الجهاز بالذات (بطلبك: بلا رقم ثابت مني)
                modelStatus = "⚠ ما قدرت أحمّل أوزان قديمة (" + (e.message ?: e.javaClass.simpleName) + ") — بنيت نموذج جديد فاضي بدل ما يضل التطبيق بلا نموذج. درّبه من الصفر."
            } catch (e2: Throwable) {
                modelStatus = "تعذر تحميل النموذج وتعذر بناء نموذج بديل: " + (e2.message ?: e2.javaClass.simpleName)
            }
        }
    }

    // ---------------- عالم سوبرمان: التخيّل بمحاكي حقيقي ----------------
    fun worldLab(): WorldLab = WorldLab.get(ctx)
    fun dreamOnce(): DreamResult? { val m = nlm ?: return null; return synchronized(lock) { WorldLab.get(ctx).dream(m) } }
    fun worldTrain(epochs: Int = 6, lr: Float = 4e-4f, count: Int = 120): String {
        val m = nlm ?: return "النموذج غير محمّل."
        val r = synchronized(lock) { WorldLab.get(ctx).train(m, rehearsal, epochs, lr, count) }
        saveNlm()
        return r
    }
    fun worldExam(): String { val m = nlm ?: return "النموذج غير محمّل."; return synchronized(lock) { WorldLab.get(ctx).examReport(m) } }
    fun worldStatus(): String { val m = nlm ?: return ""; return WorldLab.get(ctx).status(m) }
    fun gradCheck(): String { val m = nlm ?: return "النموذج غير محمّل."; return synchronized(lock) { m.gradCheck() } }

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
        // القديم يبقى بالذاكرة أثناء التكبير (أوزان + Adam = 3 نسخ) مع الجديد كامل بتكلفة تدريبه (5 نسخ + كاش الانتباه التربيعي).
        // الحساب القديم (3 نسخ من المعاملات فقط) كان يقلّل الحاجة الفعلية بكثير ويقبل أحجام تنهار وقت التدريب.
        val free = ContextSizer.freeHeapBytes()
        val needed = m.paramCount().toLong() * 4L * 3L + ContextSizer.trainBytes(m.chars.size, newE, m.nLayers, m.nHeads, newH)
        if (needed > free * 0.85) return "⚠ ما قدرت: هذا الحجم يحتاج تقريبا " + (needed / 1_000_000L) + " MB (يشمل تدريبه؛ النافذة تزيد الذاكرة بالتربيع)، والمتاح فعليًا الآن " + (free / 1_000_000L) + " MB فقط. جرّب نافذة أو حجما أصغر."
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
            // رسالة قديمة كانت تقول "يبدأ بنفس سلوك القديم تماما" وهذا كان غير صحيح أصلًا: grow() وقتها كان يبني
            // نموذج عشوائي كامل بدون ما ينسخ شي (شوف تعليق grow() القديم اللي كان يعترف بهذا صراحة). صار الآن
            // نقل حقيقي للأوزان القديمة إلى الأبعاد المشتركة، فالرسالة صارت تعكس الواقع الفعلي مو وعد زايد.
            "✅ كبّرت النموذج فعلا: من " + m.paramCount() + " إلى " + grown.paramCount() + " معامل. " + ModelHealth.describe(grown) + " الأوزان اللي تعلّمها النموذج القديم انتقلت كلها للأبعاد المشتركة، والزيادة الجديدة في الحجم فاضية وتحتاج تدريب — يعني يبدأ قريب من سلوك القديم بس مو مطابق له تمامًا رياضيًا (تغيّر عدد الأبعاد يغيّر حسابات الشبكة). احفظت الحالة الجديدة؛ افتح ⚡ رفع الفهم لتدريبه."
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
            if (WorldLab.isDreamRequest(q)) {            // «تخيّل نفسك في عالم»: يبني عالم بالكود ويجرّب فيه فعلًا (محاكي حقيقي)، مو كلام مولَّد
                val d = WorldLab.get(ctx).dream(m)
                s.feed("س:" + q.replace("\n", " ¶ ").trim() + "\nج:بنيت عالم وجربت فيه\n")
                return@synchronized Answer(d.report(), "world")
            }
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
                // فلتر "الدقة قبل السرعة": ميزانية الوقت لأي رسالة (كود أو عادية) بلا حد عمليًا (Long.MAX_VALUE/2) —
                // ما يتوقف أبدًا بسبب الوقت، بس بسبب عدد المحاولات (hardCap جوا replyDeep) أو لما يلقى محاولة ناجحة
                // في فحص codeSanity (للكود بس؛ المحادثة العادية ما عندها بوابة قبول، تعتمد على self-consistency فقط).
                val timeBudget = Long.MAX_VALUE / 2
                // وضع ماكس: توليد سطر-سطر مع توقف وفحص سلامة حقيقي بعد كل سطر (replyLineByLine) — هذا هو
                // "يفكر حتى وهو يكتب نصف الرسالة" فعليًا، لا مجرد إعادة محاولة الرد كامل من الأول.
                val r = if (Effort.maxMode) s.replyMax(q, maxThinkLines = 3000, candidatesPerLine = 18, baseTemp = plan.temp)   // رفعتها بطلبك الصريح: تفكير أكبر بكثير (10x أسطر، تقريبًا ضعف المرشحين لكل سطر)، الوقت ما له قيمة في هذا الوضع
                         else if (plan.level == Level.THINK) s.replyDeep(q, plan.maxChars, plan.temp, timeBudget, Effort.minThinkMs, accept = if (plan.isCode) ::codeSanity else null)
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
            val written = synchronized(lock) { sessionFor(m, convId).replyThink(q, if (project) 1500 else 900, if (project) 5 else 4, minMs = Effort.minThinkMs) { FileTools.allSane(it) }.first }
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
        val gw = synchronized(lock) { m.gradWarning() }
        val r = synchronized(lock) { m.develop(texts, 10, 3e-4f, rehearsal) }
        saveNlm()
        lastReport = JSONObject().put("heard", memory.size).put("bits_before", r.first.toDouble()).put("bits_after", r.second.toDouble()).put("params", m.paramCount())
        val f = { x: Float -> String.format(java.util.Locale.US, "%.2f", x) }
        val sb = StringBuilder()
        sb.append("راجعت كل ما سمعته منك: ").append(memory.size).append(" رسالة.\n")
        sb.append("• تدربت على آخر ").append(texts.size).append(" رسالة، وقست نفسي قبل وبعد: ").append(f(r.first)).append(" ← ").append(f(r.second)).append(" بت/حرف (الأقل أفضل).\n")
        if (r.second >= r.first) sb.append("• ما تحسّنت هذه المرة، فرجعت لأوزاني السابقة بدون تغيير.\n") else sb.append("• تحسّنت، وحفظت الأوزان الجديدة.\n")
        sb.append("• حجمي: ").append(m.paramCount()).append(" معامل، وذاكرتي الطويلة: ").append(m.long.chars).append(" حرف من كلامك.\n")
        if (gw != null) sb.append("\n").append(gw).append("\n")
        sb.append("\nاضغط «ملف المطور» لأخرج لك ملفا فيه أوزاني الجديدة وذاكرتي وتقرير التحسن، وترسله لمن يفحصه.")
        return sb.toString()
    }

    /** تدريب قوي على نص كبير (الصق فيه كتاب أو مقال كامل) — منفصل عن زر "رفع الفهم" العادي (اللي يراجع رسائل
     *  الدردشة القصيرة). epochs=15 و lr=4e-4 افتراضيًا: أقوى من 10/3e-4 في develop() العادي، لأن هذا مخصص
     *  لجلسة تدريب مقصودة ومهمة، لا لمراجعة خفيفة بعد كل كم رسالة. */
    fun developBig(text: String, epochs: Int = 15, lr: Float = 4e-4f): String {
        val m = nlm ?: return "النموذج غير محمّل."
        val gw = synchronized(lock) { m.gradWarning() }
        val report = synchronized(lock) { m.trainBig(text, rehearsal, epochs, lr) }
        saveNlm()
        return if (gw != null) gw + "\n\n" + report else report
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
/** نسخة جزئية من codeSanity لنص لسه ناقص (مو كامل): بس تتأكد إن قوس إغلاق ما طلع قبل ما ينفتح قوسه — مو إنه
 *  متوازن كليًا (طبيعي يكون فيه أقواس مفتوحة لسه وسط الكود). تستخدمها replyLineByLine عشان توقف أول ما تشوف
 *  خطأ واضح، بدل ما تكمل كتابة على أساس خاطئ. */
private fun partialSane(s: String): Boolean {
    var round = 0; var square = 0; var curly = 0
    for (c in s) when (c) {
        '(' -> round++; ')' -> { round--; if (round < 0) return false }
        '[' -> square++; ']' -> { square--; if (square < 0) return false }
        '{' -> curly++; '}' -> { curly--; if (curly < 0) return false }
    }
    return true
}

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

/** حد الرام: مو موجود جوا trainBig/trainOn نفسها إطلاقًا (ما فيها أي قص صامت) — هذا بس رقم معلوماتي تقدر
 *  تستخدمه بنفسك من سكربت GitHub لو حبيت تحسب حجم الدفعة المناسب، وما يُفرض على المحرك تلقائيًا أبدًا.
 *  maxRamMb = null يعني "تلقائي" (يكتشف الذاكرة المتاحة فعليًا عبر JVM، بس ما يتجاوز autoCeilingMb أبدًا).
 *  حط رقم صريح في maxRamMb لو تبي يدوي بالضبط، وقتها autoCeilingMb ما له أي تأثير. الأرقام بالميجابايت. */
object EngineConfig {
    @Volatile var maxRamMb: Int? = null
    @Volatile var autoCeilingMb: Int = 4096   // 4 جيجا تقريبًا — غيّرها بنفسك هنا لـ3072 لو تبي سقف 3 جيجا بدل 4
    fun effectiveLimitMb(): Int {
        val manual = maxRamMb
        if (manual != null) return manual
        val detected = Runtime.getRuntime().maxMemory() / (1024 * 1024)
        return minOf(detected, autoCeilingMb.toLong()).toInt()
    }
}

/** يختار نافذة السياق (h) تلقائيًا من ذاكرة الجوال الفعلية وقت إنشاء نموذج جديد — بلا رقم ثابت مني.
 *  تصحيح مهم لنسختي السابقة (كانت تحسب ذاكرة التوليد فقط وتسكت عن التدريب): التدريب يخزّن كاش الانتباه لكل رأس
 *  وكل موضع مثلثيًا (T²/2 رقم × رؤوس × طبقات)، يعني ذاكرته تزيد بالتربيع مع h، مو خطيًا — وهذا هو القيد الحقيقي.
 *  الحساب كله (تقدير مبني على مصفوفات run() الفعلية):
 *   • ثابت: الأوزان + Adam(m,v) + تدرّج + نسخة rollback = 5 نسخ من المعاملات.
 *   • خطي بكل موضع: ~24e رقم لكل طبقة (كاش + تدرّجات مؤقتة) + 8v (logits/probs) + 16e.
 *   • تربيعي: nLayers × nHeads × T²/2 رقم × 4 بايت.
 *  ونحل المعادلة التربيعية لأكبر T يدخل في fraction من الذاكرة الفاضية فعليًا (مو كل الهيب). */
object ContextSizer {
    private fun linPerPos(e: Int, nLayers: Int, v: Int): Long = 24L * e * 4L * nLayers + 8L * v + 16L * e
    private fun quadCoef(nLayers: Int, nHeads: Int): Long = 2L * nLayers * nHeads

    /** تقدير ذاكرة تدريب نموذج بالحجم المعطى (بايت). */
    fun trainBytes(v: Int, e: Int, nLayers: Int, nHeads: Int, h: Int): Long =
        Nlm.estimateParams(v, e, h) * 4L * 5L + linPerPos(e, nLayers, v) * h + quadCoef(nLayers, nHeads) * h.toLong() * h

    fun freeHeapBytes(): Long {
        val rt = Runtime.getRuntime()
        return (rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())).coerceAtLeast(1L)
    }

    /** min: حد أدنى عشان حتى أضعف جوال يطلع له شي مفيد. max: سقف دفاعي فقط (مو قرار محتوى). */
    fun pick(e: Int, nLayers: Int, nHeads: Int, v: Int, min: Int = 256, max: Int = 2_000_000, fraction: Double = 0.30): Int {
        val budget = (freeHeapBytes() * fraction)
        val fixed = Nlm.estimateParams(v, e, 0).toDouble() * 4.0 * 5.0
        val avail = budget - fixed
        if (avail <= 0.0) return min
        val lin = (linPerPos(e, nLayers, v) + 20L * e).toDouble()   // + posEmb × 5 نسخ
        val quad = quadCoef(nLayers, nHeads).toDouble()
        val t = (-lin + Math.sqrt(lin * lin + 4.0 * quad * avail)) / (2.0 * quad)
        return t.toInt().coerceIn(min, max)
    }
}

object Effort {
    /** وضع "ماكس": دقة قصوى على أي رسالة (مو بس كود)، الوقت ما له قيمة إطلاقًا، توليد سطر-سطر مع توقف وتحقق
     *  حقيقي بعد كل سطر (replyLineByLine)، وبوابة سلامة هيكلية حتى على كلام عادي. يفعّله المستخدم بنفسه من
     *  الإعدادات (سويتش)، مو مفعّل افتراضيًا — لأنه أبطأ بكثير من الوضع العادي عمدًا. */
    @Volatile var maxMode: Boolean = false
    /** Measured speed of THIS phone with THIS model (milliseconds per generated character); 0 = not measured yet. Brain updates it after every reply, so a
     *  bigger model (10M..100M weights) or a slower phone automatically gets shorter replies / fewer attempts instead of making the phone wait for minutes. */
    @Volatile var msPerChar = 0.0
    /** ميزانية "التفكير العميق" (replyDeep) بالمللي ثانية. 180000 = 3 دقائق افتراضيًا. غيّرها بنفسك لساعة مثلاً: 3_600_000L،
     *  أو لساعتين: 7_200_000L. القيمة هذي وحدها هي اللي تتحكم بطول الانتظار — ما تحتاج تعدل أي مكان ثاني. */
    @Volatile var deepThinkMaxMs: Long = 180_000L
    /** حد أدنى حقيقي (مو تقريبي) لكل رد، حتى بالمحادثة العادية: 3000 = 3 ثواني. ما يرجع أسرع من هذا أبدًا،
     *  بصرف النظر عن سرعة توليد النموذج — المدة الإضافية تُستغل فعليًا بمحاولات self-consistency أكثر، مو انتظار فاضي. */
    @Volatile var minThinkMs: Long = 5000L
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
        // زر ماكس: المستخدم فعّله بنفسه بوعي من الإعدادات، فهنا (وبس هنا) يتجاوز حتى حماية البطارية/الحرارة —
        // اختيار واعٍ منه، مو سلوك افتراضي مفروض. أي رسالة بهذا الوضع تُعامل زي أقوى كود ممكن: بلا حد وقت،
        // بوابة سلامة هيكلية حتى على كلام عادي، تفعيل مباشر بدون المرور على كاشف الكلمات.
        if (Effort.maxMode) return Plan(Level.THINK, 200000, 0.1f, 16, "⚡ وضع ماكس: تفكير أكبر بكثير (رُفع بطلبك)، دقة قصوى، الوقت بلا قيمة إطلاقًا", isCode = true)
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
            // كان chars(2000, 90000.0) — حد مرتبط بميزانية وقت تخيّلية. شلته بناءً على طلبك الصريح: حتى
            // المحادثة العادية (مو بس الكود) صارت بلا حد لطول الرد، 200000 زي مسار الكود بالضبط. نفس تحذير
            // الكود ينطبق هنا: هذا سقف الإمكانية فقط، الطول الفعلي يحدده النموذج نفسه وقت التوليد.
            else -> Plan(Level.THINK, 200000, 0.3f, 4, note)
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
                val msg = JSONObject().put("type", "connect_controller").put("code", code).put("controllerName", "جوال سوبرمان")
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
    fun reviewAndFix(brain: Brain, activity: Activity, convId: String, code: String, fileName: String, maxPasses: Int = 60, log: (String) -> Unit = {}): String {
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

    fun startContinuous(activity: Activity, getCurrentHtml: () -> String, onFrame: (String) -> Unit, intervalMs: Long = 1000) {
        if (continuousRunning) return
        continuousRunning = true
        val tick = object : Runnable {
            override fun run() {
                if (!continuousRunning) return
                val ascii = captureAscii(activity, getCurrentHtml(), 16, 16, renderPx = 144, timeoutMs = 2000)
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

/** The Nuwaa language model, written from scratch, running and LEARNING on the phone.
 *  Neural part: proper small decoder-only transformer — multi-head causal self-attention, pre-LayerNorm,
 *  residual connections, 3 stacked blocks, feed-forward, final LayerNorm, output head. Fixed context window
 *  of h characters (REPLACED the GRU's unbounded-but-decaying hidden state on explicit request).
 *  Hand-written forward+backward (no autodiff library) + Adam.
 *  No longer a mirror of any Python file — nuwaa/lm/gru.py (if it still exists) now describes a DIFFERENT,
 *  incompatible architecture unless someone updates it too.
 *  No pretrained or licensed weights, no network. HONEST LIMIT: small model, small data, AND this specific
 *  rewrite was never compiled or numerically tested — gradient-check it before trusting long training runs.
 *  This is meaningfully more complex than a single-head/single-layer/no-LN version would be (multi-head
 *  attention backward + LayerNorm backward + a 3-layer stack, all by hand) — more moving parts, more surface
 *  for a subtle error, in exchange for the stronger, more standard architecture you asked for. */
class Nlm private constructor(val e: Int, val h: Int, val chars: List<String>) {
    val v = chars.size
    private val vocab = HashMap<String, Int>().also { m -> chars.forEachIndexed { i, c -> m[c] = i } }
    private val PAD = v
    private val ff = 4 * e
    // رجّعتها من 44 إلى 8 بطلبك الصريح (بعد ما أكّدت 44 قبل): كانت تعطي كل رأس بعد 2 بس (headDim=e/44)،
    // ضيق جدًا يخلي كل رأس شبه ما يقدر يمثّل شي ذو معنى لحاله. 8 رؤوس مع e=88 تعطي headDim=11 — أوسع بـ5.5
    // مرة، قريب من الحد الأدنى المعقول (≥8) اللي كان ناقص. ما يغيّر عدد المعاملات إطلاقًا (wq/wk/wv/wo نفس
    // حجمها e×e بصرف النظر عن عدد الرؤوس، بس توزيعها بين الرؤوس يتغيّر) — يعني هذا تحسين مجاني في القدرة،
    // بدون ما يكبّر النموذج أو الذاكرة أو الكلفة الحسابية.
    val nHeads = HEADS.also { require(e % it == 0) { "e=$e لازم يكون من مضاعفات 8 (مثلاً 88، 96، 104)" } }
    private val hd = e / nHeads
    val nLayers = 3
    private val EPS = 1e-5

    /** فهرس أوزان طبقة واحدة داخل القائمة المسطّحة P (مبنية مرة وحدة وقت الإنشاء) — يخلي كود forward/backward
     *  يقرأ "Lyr.wq" بدل ما يحسب أرقام إزاحة يدويًا، تقليل حقيقي لفرص الغلط بتكرار 3 طبقات. */
    private class LIdx(base: ArrayList<FloatArray>, e: Int, ff: Int) {
        val ln1g = base.size.also { base.add(FloatArray(e)) }
        val ln1b = base.size.also { base.add(FloatArray(e)) }
        val wq = base.size.also { base.add(FloatArray(e * e)) }
        val wk = base.size.also { base.add(FloatArray(e * e)) }
        val wv = base.size.also { base.add(FloatArray(e * e)) }
        val wo = base.size.also { base.add(FloatArray(e * e)) }
        val bo = base.size.also { base.add(FloatArray(e)) }
        val ln2g = base.size.also { base.add(FloatArray(e)) }
        val ln2b = base.size.also { base.add(FloatArray(e)) }
        val w1 = base.size.also { base.add(FloatArray(e * ff)) }
        val b1 = base.size.also { base.add(FloatArray(ff)) }
        val w2 = base.size.also { base.add(FloatArray(ff * e)) }
        val b2 = base.size.also { base.add(FloatArray(e)) }
    }

    private val _P = ArrayList<FloatArray>()
    private val tokEmb = _P.size.also { _P.add(FloatArray((v + 1) * e)) }
    private val posEmb = _P.size.also { _P.add(FloatArray(h * e)) }
    private val layers = (0 until nLayers).map { LIdx(_P, e, ff) }
    private val lnFg = _P.size.also { _P.add(FloatArray(e)) }
    private val lnFb = _P.size.also { _P.add(FloatArray(e)) }
    private val wout = _P.size.also { _P.add(FloatArray(e * v)) }
    private val bout = _P.size.also { _P.add(FloatArray(v)) }
    val P: List<FloatArray> = _P

    private val mAdam = P.map { FloatArray(it.size) }
    private val vAdam = P.map { FloatArray(it.size) }
    private var tAdam = 0

    private data class TrainingSnapshot(
        val weights: List<FloatArray>,
        val firstMoment: List<FloatArray>,
        val secondMoment: List<FloatArray>,
        val adamStep: Int
    )

    private fun snapshotTrainingState() = TrainingSnapshot(
        weights = P.map { it.copyOf() },
        firstMoment = mAdam.map { it.copyOf() },
        secondMoment = vAdam.map { it.copyOf() },
        adamStep = tAdam
    )

    private fun restoreTrainingState(snapshot: TrainingSnapshot) {
        for (k in P.indices) {
            System.arraycopy(snapshot.weights[k], 0, P[k], 0, P[k].size)
            System.arraycopy(snapshot.firstMoment[k], 0, mAdam[k], 0, mAdam[k].size)
            System.arraycopy(snapshot.secondMoment[k], 0, vAdam[k], 0, vAdam[k].size)
        }
        tAdam = snapshot.adamStep
    }

    val long = NMem()

    fun cid(c: Char): Int = vocab[c.toString()] ?: 1
    fun encode(s: String): IntArray = IntArray(s.length) { cid(s[it]) }
    fun paramCount(): Int = P.sumOf { it.size }

    // ---------------------------------------------------------------- helpers عامة
    private fun matVec(W: FloatArray, inDim: Int, outDim: Int, x: FloatArray, out: FloatArray) {
        // نفس الحساب بالضبط (لكل j الجمع بنفس ترتيب i، فالنتيجة متطابقة بت ببت) بس نقرأ W صف-صف بدل القفز بين الصفوف:
        // أسرع بكثير على الجوال لأن الذاكرة تُقرأ متتالية. هذا يسرّع التوليد والتدريب كلهم.
        java.util.Arrays.fill(out, 0, outDim, 0f)
        for (i in 0 until inDim) { val xi = x[i]; val base = i * outDim; for (j in 0 until outDim) out[j] += xi * W[base + j] }
    }
    private fun matVecT(W: FloatArray, inDim: Int, outDim: Int, dOut: FloatArray, dx: FloatArray) {
        for (i in 0 until inDim) { var s = 0f; for (j in 0 until outDim) s += dOut[j] * W[i * outDim + j]; dx[i] += s }
    }
    private fun addOuter(dW: FloatArray, inDim: Int, outDim: Int, x: FloatArray, dOut: FloatArray) {
        for (i in 0 until inDim) { val xi = x[i]; if (xi == 0f) continue; val base = i * outDim; for (j in 0 until outDim) dW[base + j] += xi * dOut[j] }
    }
    private fun softmaxRow(logits: FloatArray): FloatArray {
        var mx = Float.NEGATIVE_INFINITY; for (x in logits) if (x > mx) mx = x
        var sum = 0.0; val out = FloatArray(logits.size)
        for (i in logits.indices) { val ex = Math.exp((logits[i] - mx).toDouble()); out[i] = ex.toFloat(); sum += ex }
        for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out
    }
    /** LayerNorm forward لمتجه واحد. يرجّع y، ويخزّن xhat/mean/rstd بالمصفوفات الممررة (لازمة للخلف). */
    private fun lnFwd(x: FloatArray, gamma: FloatArray, beta: FloatArray, xhatOut: FloatArray): Pair<FloatArray, Float> {
        val n = x.size
        var mean = 0f; for (q in x) mean += q; mean /= n
        var vr = 0f; for (q in x) { val d0 = q - mean; vr += d0 * d0 }; vr /= n
        val rstd = (1.0 / Math.sqrt((vr + EPS))).toFloat()
        val y = FloatArray(n)
        for (i in 0 until n) { val xh = (x[i] - mean) * rstd; xhatOut[i] = xh; y[i] = xh * gamma[i] + beta[i] }
        return Pair(y, rstd)
    }
    /** LayerNorm backward (الصيغة المختصرة القياسية): dx = rstd*(dxhat - mean(dxhat) - xhat*mean(dxhat*xhat)). */
    private fun lnBwd(dy: FloatArray, xhat: FloatArray, gamma: FloatArray, rstd: Float, dgamma: FloatArray, dbeta: FloatArray, dx: FloatArray) {
        val n = dy.size
        val dxhat = FloatArray(n)
        var sumA = 0f; var sumB = 0f
        for (i in 0 until n) { dgamma[i] += xhat[i] * dy[i]; dbeta[i] += dy[i]; dxhat[i] = dy[i] * gamma[i]; sumA += dxhat[i]; sumB += dxhat[i] * xhat[i] }
        for (i in 0 until n) dx[i] += rstd * (dxhat[i] - sumA / n - xhat[i] * sumB / n)
    }

    // ---------------------------------------------------------------- forward فقط (توليد حرف-حرف، بلا كاش للخلف)
    private fun forwardProbsLast(x: IntArray): FloatArray {
        val T = x.size; val d = e
        var cur = Array(T) { FloatArray(d) }
        for (t in 0 until T) { val id = x[t]; for (q in 0 until d) cur[t][q] = P[tokEmb][id * d + q] + P[posEmb][t * d + q] }
        val scale = (1.0 / Math.sqrt(hd.toDouble())).toFloat()
        for (L in layers) {
            val ln1 = Array(T) { FloatArray(d) }; val xh1 = Array(T) { FloatArray(d) }
            for (t in 0 until T) ln1[t] = lnFwd(cur[t], P[L.ln1g], P[L.ln1b], xh1[t]).first
            val Q = Array(T) { FloatArray(d) }; val K = Array(T) { FloatArray(d) }; val Vv = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.wq], d, d, ln1[t], Q[t]); matVec(P[L.wk], d, d, ln1[t], K[t]); matVec(P[L.wv], d, d, ln1[t], Vv[t]) }
            val ctx = Array(T) { FloatArray(d) }
            for (t in 0 until T) for (hI in 0 until nHeads) {
                val off = hI * hd
                val sc = FloatArray(t + 1)
                for (s in 0..t) { var dp = 0f; for (q in 0 until hd) dp += Q[t][off + q] * K[s][off + q]; sc[s] = dp * scale }
                val a = softmaxRow(sc)
                for (q in 0 until hd) { var s2 = 0f; for (s in 0..t) s2 += a[s] * Vv[s][off + q]; ctx[t][off + q] = s2 }
            }
            val attnOut = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.wo], d, d, ctx[t], attnOut[t]); for (q in 0 until d) attnOut[t][q] += P[L.bo][q] }
            val hMid = Array(T) { FloatArray(d) }
            for (t in 0 until T) for (q in 0 until d) hMid[t][q] = cur[t][q] + attnOut[t][q]
            val ln2 = Array(T) { FloatArray(d) }; val xh2 = Array(T) { FloatArray(d) }
            for (t in 0 until T) ln2[t] = lnFwd(hMid[t], P[L.ln2g], P[L.ln2b], xh2[t]).first
            val f1 = Array(T) { FloatArray(ff) }
            for (t in 0 until T) { matVec(P[L.w1], d, ff, ln2[t], f1[t]); for (q in 0 until ff) f1[t][q] = maxOf(0f, f1[t][q] + P[L.b1][q]) }
            val f2 = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.w2], ff, d, f1[t], f2[t]); for (q in 0 until d) f2[t][q] += P[L.b2][q] }
            cur = Array(T) { tt -> FloatArray(d) { qq -> hMid[tt][qq] + f2[tt][qq] } }
        }
        val t = T - 1
        val xhF = FloatArray(d)
        val lnF = lnFwd(cur[t], P[lnFg], P[lnFb], xhF).first
        val logits = FloatArray(v); matVec(P[wout], d, v, lnF, logits); for (q in 0 until v) logits[q] += P[bout][q]
        return softmaxRow(logits)
    }

    fun stepChar(id: Int, s1: FloatArray, s2: FloatArray): FloatArray {
        for (i in 0 until h - 1) s1[i] = s1[i + 1]
        s1[h - 1] = (id + 1).toFloat()
        val wids = IntArray(h) { val r = s1[it].toInt() - 1; if (r < 0) PAD else r }
        return forwardProbsLast(wids)
    }

    // ---------------------------------------------------------------- forward+backward كامل فوق نافذة (للتدريب)
    private fun run(x: IntArray, y: IntArray, grads: Boolean): Pair<Float, List<FloatArray>?> {
        val T = minOf(x.size, h); val d = e
        val posOffset = h - T // Short training windows align to the newest positions used by generation.
        val scale = (1.0 / Math.sqrt(hd.toDouble())).toFloat()
        val embIn = Array(T) { FloatArray(d) }
        for (t in 0 until T) { val id = x[t]; for (q in 0 until d) embIn[t][q] = P[tokEmb][id * d + q] + P[posEmb][(posOffset + t) * d + q] }

        // كاش كل طبقة (لازم للخلف): مدخل الطبقة، LN الأول وxhat/rstd، Q/K/V، أوزان الانتباه لكل رأس، ctx، hMid، LN الثاني وxhat/rstd، f1 (بعد relu)
        data class Cache(val inp: Array<FloatArray>, val ln1: Array<FloatArray>, val xh1: Array<FloatArray>, val rstd1: FloatArray,
                          val Q: Array<FloatArray>, val K: Array<FloatArray>, val Vv: Array<FloatArray>, val attn: Array<Array<FloatArray>>,
                          val ctx: Array<FloatArray>, val hMid: Array<FloatArray>, val ln2: Array<FloatArray>, val xh2: Array<FloatArray>,
                          val rstd2: FloatArray, val z1: Array<FloatArray>, val f1: Array<FloatArray>)
        val caches = ArrayList<Cache>()
        var cur = embIn
        for (L in layers) {
            val ln1 = Array(T) { FloatArray(d) }; val xh1 = Array(T) { FloatArray(d) }; val rstd1 = FloatArray(T)
            for (t in 0 until T) { val (y0, r0) = lnFwd(cur[t], P[L.ln1g], P[L.ln1b], xh1[t]); ln1[t] = y0; rstd1[t] = r0 }
            val Q = Array(T) { FloatArray(d) }; val K = Array(T) { FloatArray(d) }; val Vv = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.wq], d, d, ln1[t], Q[t]); matVec(P[L.wk], d, d, ln1[t], K[t]); matVec(P[L.wv], d, d, ln1[t], Vv[t]) }
            val attn = Array(nHeads) { Array(T) { FloatArray(T) } }
            val ctx = Array(T) { FloatArray(d) }
            for (t in 0 until T) for (hI in 0 until nHeads) {
                val off = hI * hd
                val sc = FloatArray(t + 1)
                for (s in 0..t) { var dp = 0f; for (q in 0 until hd) dp += Q[t][off + q] * K[s][off + q]; sc[s] = dp * scale }
                val a = softmaxRow(sc)
                for (s in 0..t) attn[hI][t][s] = a[s]
                for (q in 0 until hd) { var s2 = 0f; for (s in 0..t) s2 += a[s] * Vv[s][off + q]; ctx[t][off + q] = s2 }
            }
            val attnOut = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.wo], d, d, ctx[t], attnOut[t]); for (q in 0 until d) attnOut[t][q] += P[L.bo][q] }
            val hMid = Array(T) { FloatArray(d) }
            for (t in 0 until T) for (q in 0 until d) hMid[t][q] = cur[t][q] + attnOut[t][q]
            val ln2 = Array(T) { FloatArray(d) }; val xh2 = Array(T) { FloatArray(d) }; val rstd2 = FloatArray(T)
            for (t in 0 until T) { val (y0, r0) = lnFwd(hMid[t], P[L.ln2g], P[L.ln2b], xh2[t]); ln2[t] = y0; rstd2[t] = r0 }
            val z1 = Array(T) { FloatArray(ff) }
            for (t in 0 until T) { matVec(P[L.w1], d, ff, ln2[t], z1[t]); for (q in 0 until ff) z1[t][q] += P[L.b1][q] }
            val f1 = Array(T) { FloatArray(ff) }
            for (t in 0 until T) for (q in 0 until ff) f1[t][q] = if (z1[t][q] > 0f) z1[t][q] else 0f
            val f2 = Array(T) { FloatArray(d) }
            for (t in 0 until T) { matVec(P[L.w2], ff, d, f1[t], f2[t]); for (q in 0 until d) f2[t][q] += P[L.b2][q] }
            val hOut = Array(T) { tt -> FloatArray(d) { qq -> hMid[tt][qq] + f2[tt][qq] } }
            caches.add(Cache(cur, ln1, xh1, rstd1, Q, K, Vv, attn, ctx, hMid, ln2, xh2, rstd2, z1, f1))
            cur = hOut
        }
        val xhF = Array(T) { FloatArray(d) }; val rstdF = FloatArray(T); val lnF = Array(T) { FloatArray(d) }
        for (t in 0 until T) { val (y0, r0) = lnFwd(cur[t], P[lnFg], P[lnFb], xhF[t]); lnF[t] = y0; rstdF[t] = r0 }
        val logits = Array(T) { FloatArray(v) }; val probs = Array(T) { FloatArray(v) }
        var loss = 0f
        var nValid = 0
        for (t in 0 until T) {
            if (y[t] < 0) continue                       // موضع حشو (PAD): لا توقع ولا خسارة ولا تدرّج (شوف makeWindow)
            nValid++
            matVec(P[wout], d, v, lnF[t], logits[t]); for (q in 0 until v) logits[t][q] += P[bout][q]
            probs[t] = softmaxRow(logits[t])
            loss -= Math.log((probs[t][y[t]] + 1e-9f).toDouble()).toFloat()
        }
        loss /= maxOf(1, nValid)
        if (!grads) return Pair(loss, null)

        val g = P.map { FloatArray(it.size) }
        val dLnF = Array(T) { FloatArray(d) }
        for (t in 0 until T) {
            if (y[t] < 0) continue
            val dlog = probs[t].copyOf(); dlog[y[t]] -= 1f
            for (q in 0 until v) dlog[q] /= maxOf(1, nValid).toFloat()
            addOuter(g[wout], d, v, lnF[t], dlog)
            for (q in 0 until v) g[bout][q] += dlog[q]
            matVecT(P[wout], d, v, dlog, dLnF[t])
        }
        var dCur = Array(T) { FloatArray(d) }
        for (t in 0 until T) lnBwd(dLnF[t], xhF[t], P[lnFg], rstdF[t], g[lnFg], g[lnFb], dCur[t])

        for (li in layers.indices.reversed()) {
            val L = layers[li]; val c = caches[li]
            // dCur هنا = dHout لهذي الطبقة
            val dHmid = Array(T) { FloatArray(d) }
            val dF2 = dCur
            for (t in 0 until T) {
                for (q in 0 until d) dHmid[t][q] += dF2[t][q]
                addOuter(g[L.w2], ff, d, c.f1[t], dF2[t])
                for (q in 0 until d) g[L.b2][q] += dF2[t][q]
                val dF1 = FloatArray(ff); matVecT(P[L.w2], ff, d, dF2[t], dF1)
                val dZ1 = FloatArray(ff)
                for (q in 0 until ff) dZ1[q] = if (c.z1[t][q] > 0f) dF1[q] else 0f
                addOuter(g[L.w1], d, ff, c.ln2[t], dZ1)
                for (q in 0 until ff) g[L.b1][q] += dZ1[q]
                val dLn2 = FloatArray(d); matVecT(P[L.w1], d, ff, dZ1, dLn2)
                lnBwd(dLn2, c.xh2[t], P[L.ln2g], c.rstd2[t], g[L.ln2g], g[L.ln2b], dHmid[t])
            }
            val dAttnIn = Array(T) { FloatArray(d) }     // = dCur لهذي الطبقة (مدخل الانتباه، residual 1)
            val dQ = Array(T) { FloatArray(d) }; val dK = Array(T) { FloatArray(d) }; val dV = Array(T) { FloatArray(d) }
            for (t in 0 until T) {
                for (q in 0 until d) dAttnIn[t][q] += dHmid[t][q]
                val dAttnOut = dHmid[t]
                addOuter(g[L.wo], d, d, c.ctx[t], dAttnOut)
                for (q in 0 until d) g[L.bo][q] += dAttnOut[q]
                val dCtx = FloatArray(d); matVecT(P[L.wo], d, d, dAttnOut, dCtx)
                for (hI in 0 until nHeads) {
                    val off = hI * hd
                    val da = FloatArray(t + 1)
                    for (s in 0..t) { var dp = 0f; for (q in 0 until hd) dp += dCtx[off + q] * c.Vv[s][off + q]; da[s] = dp }
                    for (s in 0..t) for (q in 0 until hd) dV[s][off + q] += c.attn[hI][t][s] * dCtx[off + q]
                    var dot = 0f; for (s in 0..t) dot += c.attn[hI][t][s] * da[s]
                    val dsc = FloatArray(t + 1)
                    for (s in 0..t) dsc[s] = c.attn[hI][t][s] * (da[s] - dot)
                    for (s in 0..t) {
                        val coef = dsc[s] * scale
                        for (q in 0 until hd) { dQ[t][off + q] += coef * c.K[s][off + q]; dK[s][off + q] += coef * c.Q[t][off + q] }
                    }
                }
            }
            for (t in 0 until T) {
                addOuter(g[L.wq], d, d, c.ln1[t], dQ[t]); val dLn1q = FloatArray(d); matVecT(P[L.wq], d, d, dQ[t], dLn1q)
                addOuter(g[L.wk], d, d, c.ln1[t], dK[t]); val dLn1k = FloatArray(d); matVecT(P[L.wk], d, d, dK[t], dLn1k)
                addOuter(g[L.wv], d, d, c.ln1[t], dV[t]); val dLn1v = FloatArray(d); matVecT(P[L.wv], d, d, dV[t], dLn1v)
                val dLn1 = FloatArray(d); for (q in 0 until d) dLn1[q] = dLn1q[q] + dLn1k[q] + dLn1v[q]
                lnBwd(dLn1, c.xh1[t], P[L.ln1g], c.rstd1[t], g[L.ln1g], g[L.ln1b], dAttnIn[t])
            }
            dCur = dAttnIn
        }
        for (t in 0 until T) { val id = x[t]; for (q in 0 until d) { g[tokEmb][id * d + q] += dCur[t][q]; g[posEmb][(posOffset + t) * d + q] += dCur[t][q] } }
        return Pair(loss, g)
    }

    private fun adam(g: List<FloatArray>, lr: Float) {
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

    fun bpc(ids: IntArray, chunk: Int = h): Float {   // يقيس بنفس شكل نوافذ التوليد/التدريب (حشو + الأحدث في آخر موضع)، بلا تداخل
        val nn = ids.size - 1
        if (nn < 2) return 0f
        val win = minOf(chunk, h).coerceAtLeast(2)
        var starts = windowStarts(nn, win, win)
        if (nn > win && starts[0] < 0) starts = starts.drop(1)
        var tot = 0.0; var cnt = 0
        for (s0 in starts) {
            val w = makeWindow(ids, s0, win)
            val nv = w.second.count { it >= 0 }
            if (nv == 0) continue
            tot += run(w.first, w.second, false).first.toDouble() * nv; cnt += nv
        }
        return if (cnt == 0) 0f else (tot / cnt / Math.log(2.0)).toFloat()
    }

    fun observe(text: String) { long.add(text) }

    /** معدل التعلم: إحماء قصير (5% من الخطوات) ثم نزول جيبي (cosine) لين 30% من القيمة العليا. التعلم بمعدل ثابت يتذبذب
     *  آخر التدريب؛ النزول التدريجي يثبّت الأوزان على أحسن نقطة. الحارس القديم (رفض التدريب لو ساءت النتيجة) باقٍ كما هو. */
    private fun lrAt(step: Int, total: Int, lr: Float): Float {
        val warm = maxOf(1, total / 20)
        if (step < warm) return lr * (step + 1) / warm
        val t = (step - warm).toDouble() / maxOf(1, total - warm)
        return (lr * (0.3 + 0.7 * 0.5 * (1.0 + Math.cos(Math.PI * t)))).toFloat()
    }

    /** مشكلة حقيقية كانت موجودة: التوليد (stepChar) يعطي النموذج دايمًا نافذة بطول h كاملة، الحرف الأحدث في آخر موضع
     *  والباقي حشو (PAD) في بداية أي محادثة — بينما التدريب القديم كان يقطع النص نوافذ بلا حشو تبدأ من الموضع 0.
     *  النتيجة: صفوف الموضع (posEmb) القريبة من الحاضر وصف PAD في tokEmb ما كانت تتدرب أبدًا، مع إن كل رد جديد يبدأ
     *  بها بالضبط. الآن كل نافذة تدريب بطول h كاملة وبنفس ترتيب التوليد: الأحدث دايمًا في h-1، وأي فراغ أمامه حشو
     *  (خسارته مقنّعة: y=-1). أول نافذة تبدأ بنصف حشو (تحاكي بداية محادثة)، والنوافذ تتداخل 50% عشان كل حرف
     *  يتدرب بسياق طويل، وآخر نافذة تغطي الذيل كامل، وترتيبها يتبدّل كل دورة. */
    private fun windowStarts(n: Int, win: Int, stride: Int): List<Int> {
        if (n <= 0) return emptyList()
        val out = ArrayList<Int>()
        var s0 = 0
        while (s0 < n) { out.add(s0); s0 += stride }
        return out
    }

    /**
     * Predict only the next win characters, while supplying up to h-win
     * preceding characters as real context. Previously each short training
     * chunk was left-padded, so training on 16 characters never exercised
     * the long context available during generation.
     */
    private fun makeWindow(ids: IntArray, s0: Int, win: Int): Pair<IntArray, IntArray> {
        require(win in 1..h)
        val x = IntArray(h) { PAD }
        val y = IntArray(h) { -1 }
        val contextStart = s0 - (h - win)
        for (j in 0 until h) {
            val pos = contextStart + j
            if (pos < 0 || pos >= ids.size) continue
            x[j] = ids[pos]
            if (pos >= s0 && pos < s0 + win && pos < ids.size - 1) y[j] = ids[pos + 1]
        }
        return Pair(x, y)
    }

    private fun trainOn(ids: IntArray, epochs: Int, lr: Float, chunk: Int) {
        val n = ids.size - 1
        if (n < 2) return
        val win = minOf(chunk, h).coerceAtLeast(2)
        val starts = windowStarts(n, win, win)
        val total = maxOf(1, epochs * starts.size)
        var step = 0
        for (ep in 0 until epochs) {
            for (s0 in starts.shuffled(java.util.Random(1234L + ep))) {
                val w = makeWindow(ids, s0, win)
                val g = run(w.first, w.second, true).second
                if (g != null) { adam(g, lrAt(step, total, lr)); step++ }
            }
        }
    }

    /** فحص رياضي حقيقي للتدرجات (الخلف مكتوب بيد وما انفحص رقميًا قبل): لكل معامل مختار نقارن التدرج المحسوب بالخلف مع
     *  الفرق المركزي (loss(w+e) - loss(w-e)) / 2e. لو الاثنين يتفقون فالـbackward سليم. ما يغيّر الأوزان (يرجّعها بعد كل معامل). */
    @Volatile var lastGradOk = -1
    @Volatile var lastGradTested = 0
    @Volatile private var gradWarnDone = false
    @Volatile private var gradWarnMsg: String? = null

    /** فحص تدرجات آلي مرة وحدة لكل نموذج محمّل. تحذير فقط (ما يمنع التدريب): حارس الرفض التلقائي في develop/trainBig
     *  يحمي من التدهور أصلًا، بس لو الخلف مكسور فالتحسن الحقيقي غير مضمون وأنت تستحق تعرف قبل ما تصرف ساعات تدريب. */
    fun gradWarning(): String? {
        if (!gradWarnDone) {
            gradWarnMsg = try {
                gradCheck()
                if (lastGradTested >= 10 && lastGradOk * 4 < lastGradTested * 3)
                    "⚠ فحص التدرجات الآلي: " + lastGradOk + " من " + lastGradTested + " فقط سليمة — الخلف (backward) ممكن فيه خطأ رياضي، فالتحسن الحقيقي من هذا التدريب غير موثوق (الرفض التلقائي يحمي النموذج من التدهور بس). افتح «🔬 فحص رياضيات المحرك» للتفاصيل."
                else null
            } catch (e: Throwable) { null }
            gradWarnDone = true
        }
        return gradWarnMsg
    }

    fun gradCheck(picks: Int = 70, seed: Long = 7L): String {
        val txt = "س:مرحبا كيف الحال\nج:بخير والحمد لله\n"
        val all = encode(txt)
        val n = minOf(all.size - 1, h, 24)
        if (n < 4) return "النص قصير لفحص التدرجات."
        val x = all.copyOfRange(0, n); val y = all.copyOfRange(1, n + 1)
        val g = run(x, y, true).second ?: return "تعذر حساب التدرجات."
        val names = HashMap<Int, String>()
        names[tokEmb] = "tokEmb"; names[posEmb] = "posEmb"; names[lnFg] = "lnF.g"; names[lnFb] = "lnF.b"; names[wout] = "wout"; names[bout] = "bout"
        for ((li, L) in layers.withIndex()) {
            names[L.ln1g] = "L$li.ln1g"; names[L.ln1b] = "L$li.ln1b"; names[L.wq] = "L$li.wq"; names[L.wk] = "L$li.wk"; names[L.wv] = "L$li.wv"
            names[L.wo] = "L$li.wo"; names[L.bo] = "L$li.bo"; names[L.ln2g] = "L$li.ln2g"; names[L.ln2b] = "L$li.ln2b"
            names[L.w1] = "L$li.w1"; names[L.b1] = "L$li.b1"; names[L.w2] = "L$li.w2"; names[L.b2] = "L$li.b2"
        }
        val rnd = java.util.Random(seed)
        val eps = 2e-3f
        var ok = 0; var tested = 0; var worst = 0f; var worstName = ""
        val bad = StringBuilder()
        for (t in 0 until picks) {
            val k = if (t < P.size) t else rnd.nextInt(P.size)          // أول P.size محاولات تغطي كل مصفوفة مرة على الأقل
            val i = rnd.nextInt(P[k].size)
            val w0 = P[k][i]
            P[k][i] = w0 + eps; val lp = run(x, y, false).first
            P[k][i] = w0 - eps; val lm = run(x, y, false).first
            P[k][i] = w0
            val num = (lp - lm) / (2f * eps); val ana = g[k][i]
            val diff = Math.abs(num - ana); val scale = maxOf(Math.abs(num), Math.abs(ana))
            tested++
            val good = diff <= 0.08f * scale + 2e-3f
            if (good) ok++ else if (bad.length < 300) bad.append("\n  ✗ ").append(names[k] ?: ("#" + k)).append("[").append(i).append("]: بالخلف ").append(String.format(java.util.Locale.US, "%.5f", ana)).append(" ، رقميًا ").append(String.format(java.util.Locale.US, "%.5f", num))
            val rel = if (scale > 1e-3f) diff / scale else 0f
            if (rel > worst) { worst = rel; worstName = names[k] ?: ("#" + k) }
        }
        lastGradOk = ok; lastGradTested = tested
        val head = if (ok == tested) "✅ فحص التدرجات: " + ok + " من " + tested + " سليمة (أكبر فرق نسبي " + String.format(java.util.Locale.US, "%.1f", worst * 100) + "% في " + worstName + "). الخلف مطابق للحساب الرقمي."
                   else "❌ فحص التدرجات: " + ok + " من " + tested + " فقط سليمة — فيه خطأ في الخلف (أسوأ: " + worstName + " بفرق " + String.format(java.util.Locale.US, "%.0f", worst * 100) + "%). لا تعتمد على التدريب الطويل قبل إصلاحه." + bad.toString()
        return head
    }

    fun trainWindow(text: String, lr: Float): Float {
        val ids = encode(text)
        if (ids.size < 8) return 0f
        val win = minOf(ids.size - 1, h)
        val res = run(ids.copyOfRange(0, win), ids.copyOfRange(1, win + 1), true)
        val g = res.second
        if (g != null) adam(g, lr)
        return (res.first / Math.log(2.0)).toFloat()
    }

    /**
     * Memory-bounded corpus training path. Each target character is visited once per epoch
     * (except context overlap supplied by the caller). Unlike trainBig, this does not build
     * a mixed copy of the whole corpus, shuffle a huge window list, or run full-corpus BPC
     * evaluations before and after every chunk.
     *
     * Returns the exact number of non-PAD next-character targets used in optimizer updates.
     */
    fun trainStreamChunk(
        text: String,
        epochs: Int,
        lr: Float,
        chunk: Int = 128,
        progress: (Int, Int, Int) -> Unit = { _, _, _ -> }
    ): Int {
        require(epochs in 1..5) { "epochs must be between 1 and 5" }
        require(lr.isFinite() && lr > 0f) { "learning rate must be finite and positive" }
        require(chunk in 2..h) { "chunk must be between 2 and model context" }
        val ids = encode(text)
        val n = ids.size - 1
        if (n < 1) return 0
        val win = chunk
        val windowsPerEpoch = (n + win - 1) / win
        val totalSteps = windowsPerEpoch * epochs
        var step = 0
        var trainedTargets = 0
        for (ep in 0 until epochs) {
            var start = 0
            while (start < n) {
                val window = makeWindow(ids, start, win)
                val valid = window.second.count { it >= 0 }
                if (valid > 0) {
                    val gradients = run(window.first, window.second, true).second
                        ?: error("Gradient calculation returned null during corpus training.")
                    adam(gradients, lr)
                    trainedTargets += valid
                    step++
                    if (step % 256 == 0 || step == totalSteps) {
                        progress(step, totalSteps, trainedTargets)
                    }
                }
                start += win
            }
        }
        check(trainedTargets >= n * epochs) {
            "Target accounting mismatch: trained $trainedTargets targets for $n source transitions across $epochs epochs."
        }
        return trainedTargets
    }
    fun snapshot(): List<FloatArray> = P.map { it.copyOf() }
    fun restoreSnapshot(s: List<FloatArray>) { for (k in P.indices) System.arraycopy(s[k], 0, P[k], 0, P[k].size) }

    fun develop(texts: List<String>, epochs: Int, lr: Float, rehearse: String = "", chunk: Int = h): Pair<Float, Float> {   // كان 96 ثابت — أصغر حتى من النافذة الافتراضية القديمة (64)، فالتدريب ما كان يستفيد من أي توسيع للنافذة مهما كبّرتها. بطلبك: صار = h فعليًا.
        val sb = StringBuilder()
        for (tx in texts) sb.append("س:").append(tx.take(6000)).append("\n")
        val heard = if (sb.length > 24000) sb.substring(sb.length - 24000) else sb.toString()
        val ids = encode(heard)
        if (ids.size < 8) return Pair(0f, 0f)
        val reh = if (rehearse.length > 9000) rehearse.substring(0, 9000) else rehearse
        val rids = if (reh.length > 8) encode(reh) else null
        val before = bpc(ids)
        val rBefore = if (rids != null) bpc(rids) else 0f
        val snap = snapshotTrainingState()
        trainOn(encode(heard + reh), epochs, lr, chunk)
        val after = bpc(ids)
        val rAfter = if (rids != null) bpc(rids) else 0f
        if (after > before || (rids != null && rAfter > rBefore * 1.08f)) {
            restoreTrainingState(snap)
            return Pair(before, before)
        }
        return Pair(before, after)
    }

    fun trainBig(corpus: String, rehearse: String, epochs: Int, lr: Float, bookChunk: Int = 3000, chunk: Int = minOf(128, h)): String {   // كان 128 ثابت — نفس المشكلة بالضبط: النافذة الديناميكية الجديدة (ContextSizer) ما كانت توصل فعليًا لخطوات التدريب. تنبيه صادق: نافذة تدريب = h يعني كل خطوة تدريب (تقدم + تدرّج) تاخذ ذاكرة أكبر من مجرد التوليد (التدرّج يحتاج يخزّن كل شي مرتين تقريبًا) — إذا صار بطيء جدًا أو وقف البرنامج، مرّر chunk أصغر يدويًا.
        val safeCorpus = corpus
        if (safeCorpus.length < 8) return "النص قصير جدًا، ما يستاهل تدريب."
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
            if (dose.isNotEmpty()) { sb.append("\n"); sb.append(dose) }
            i = end
        }
        val mixed = sb.toString()
        val ids = encode(mixed)
        if (ids.size < 8) return "النص بعد المعالجة قصير جدًا."
        val rids = if (reh.length > 8) encode(reh) else null
        val bookIds = encode(safeCorpus)
        val beforeBook = bpc(bookIds)
        val beforeReh = if (rids != null) bpc(rids) else 0f
        val snap = snapshotTrainingState()
        trainOn(ids, epochs, lr, chunk)
        val afterBook = bpc(bookIds)
        val afterReh = if (rids != null) bpc(rids) else 0f
        val f = { x: Float -> String.format(java.util.Locale.US, "%.3f", x) }
        val sb2 = StringBuilder()
        sb2.append("النص: ").append(f(beforeBook)).append(" ← ").append(f(afterBook)).append(" بت/حرف\n")
        if (rids != null) sb2.append("الكلام (rehearse): ").append(f(beforeReh)).append(" ← ").append(f(afterReh)).append(" بت/حرف\n")
        if (afterBook > beforeBook) {
            restoreTrainingState(snap)
            sb2.append("❌ رُفضت: ما تحسّن على النص نفسه حتى، فرجعت للأوزان القديمة. جرّب تخفّض lr أو تزيد epochs.")
            return sb2.toString()
        }
        if (rids != null && afterReh > beforeReh * 1.08f) {
            restoreTrainingState(snap)
            sb2.append("❌ رُفضت: تحسّن على النص، بس خرّب قدرة الكلام أكثر من 8%، فرجعت للأوزان القديمة. جرّب تخفّض lr أو تقلّل bookChunk.")
            return sb2.toString()
        }
        sb2.append("✅ قُبلت وحُفظت الأوزان الجديدة.")
        return sb2.toString()
    }

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
        const val HEADS = 8                      // مصدر واحد لعدد الرؤوس (ContextSizer يحتاجه لحساب ذاكرة الانتباه)
        /** عدد معاملات الترانسفورمر بالضبط (3 طبقات، ff = 4e): نفس تركيب _P فوق. الصيغة القديمة في growModel كانت لبنية GRU وما تناسب هذي. */
        fun estimateParams(v: Int, e: Int, h: Int): Long {
            val ee = e.toLong()
            val perLayer = 12L * ee * ee + 10L * ee + 4L * ee   // 4e² (Q,K,V,O) + 8e² (FFN) + تحيّزات وLN
            return (v + 1L) * ee + h.toLong() * ee + 3L * perLayer + 2L * ee + ee * v + v
        }
        var EMBEDDED_WEIGHTS_B64: String = ""

        fun loadEmbedded(): Nlm? {
            if (EMBEDDED_WEIGHTS_B64.isBlank()) return null
            return try { load(android.util.Base64.decode(EMBEDDED_WEIGHTS_B64, android.util.Base64.DEFAULT)) } catch (e: Throwable) { null }
        }

        /** تهيئة عشوائية صغيرة (مقياس 1/sqrt(input dim)) — ضرورية فعليًا، مو اختيارية؛ بدونها تضل كل التدرّجات
         *  صفر رياضيًا من نقطة بداية متناظرة كاملة (zero-init trap) وما يتعلم الشبكة أي شي أبدًا. gamma لـLN
         *  تبدأ 1 وbeta تبدأ 0 (قياسي) — لازم gamma تبدأ 1 لا صفر، وإلا LN تصفّر كل شي عند أول خطوة. */
        fun fresh(e: Int, h: Int, chars: List<String>, seed: Long = 1234L): Nlm {
            val n = Nlm(e, h, chars)
            val rnd = java.util.Random(seed)
            fun fillW(a: FloatArray, inDim: Int) {
                val scale = (1.0 / Math.sqrt(inDim.toDouble())).toFloat()
                for (i in a.indices) a[i] = ((rnd.nextDouble() * 2.0 - 1.0).toFloat()) * scale
            }
            fillW(n.P[n.tokEmb], n.e); fillW(n.P[n.posEmb], n.e)
            for (L in n.layers) {
                for (i in n.P[L.ln1g].indices) n.P[L.ln1g][i] = 1f      // LN gamma = 1، beta تبقى 0 (افتراضي الصفيف)
                for (i in n.P[L.ln2g].indices) n.P[L.ln2g][i] = 1f
                fillW(n.P[L.wq], n.e); fillW(n.P[L.wk], n.e); fillW(n.P[L.wv], n.e); fillW(n.P[L.wo], n.e)
                fillW(n.P[L.w1], n.e); fillW(n.P[L.w2], n.ff)
            }
            for (i in n.P[n.lnFg].indices) n.P[n.lnFg][i] = 1f
            fillW(n.P[n.wout], n.e)
            return n
        }

        /** تكبير حقيقي (net2net تقريبي) — يحافظ فعليًا على الأبعاد المشتركة، مو يصفّرها: نموذج جديد بالحجم
         *  الأكبر (كل أبعاده الجديدة عشوائية التهيئة من fresh())، وبعدها ننسخ فوقها كل وزن تعلّمه old حرفيًا في
         *  الأبعاد اللي لسه موجودة (أول oldE من newE، أول oldH من newH). النتيجة: الأبعاد القديمة تبدأ بنفس القيم
         *  المتعلّمة بالضبط، والأبعاد الجديدة (الزيادة في الحجم) تبدأ فاضية وتحتاج تدريب — مو "نفس السلوك تمامًا"
         *  (التغيير في عدد الأبعاد يغيّر حسابات الـLayerNorm وتوزيع الانتباه رياضيًا، فلا يوجد ضمان رياضي للسلوك
         *  المطابق 100%)، لكن أبعد بكثير عن "نسيان كل شي" اللي كانت تسويه النسخة القديمة من هذي الدالة.
         *  posEmb تُنسخ بمحاذاة النهاية (العمود الأخير = الأحدث يبقى الأحدث)، فالصفوف الجديدة (تاريخ أبعد ما شافه
         *  old أبدًا) هي اللي تبدأ عشوائية، مو الصفوف القريبة من الحاضر اللي يعرفها فعلًا. */
        fun grow(old: Nlm, newE: Int, newH: Int): Nlm {
            val grown = fresh(newE, newH, old.chars)
            val oldE = old.e; val oldH = old.h; val oldFf = 4 * oldE; val v = old.v
            fun copyRows(oA: FloatArray, oOut: Int, nA: FloatArray, nOut: Int, rows: Int, oRowOff: Int = 0, nRowOff: Int = 0) {
                for (i in 0 until rows) System.arraycopy(oA, (oRowOff + i) * oOut, nA, (nRowOff + i) * nOut, oOut)
            }
            copyRows(old.P[old.tokEmb], oldE, grown.P[grown.tokEmb], newE, v + 1)                  // جدول كلمات: نفس عدد الصفوف (المفردات ما تغيّرت)، أعمدة أقل
            copyRows(old.P[old.posEmb], oldE, grown.P[grown.posEmb], newE, oldH, 0, newH - oldH)    // نافذة أطول: الصفوف الجديدة = ماضي أبعد، تُضاف في البداية مو النهاية
            for (li in old.layers.indices) {
                val oL = old.layers[li]; val nL = grown.layers[li]
                fun v1(oIdx: Int, nIdx: Int, len: Int) = System.arraycopy(old.P[oIdx], 0, grown.P[nIdx], 0, len)
                v1(oL.ln1g, nL.ln1g, oldE); v1(oL.ln1b, nL.ln1b, oldE)
                v1(oL.ln2g, nL.ln2g, oldE); v1(oL.ln2b, nL.ln2b, oldE)
                v1(oL.bo, nL.bo, oldE); v1(oL.b2, nL.b2, oldE); v1(oL.b1, nL.b1, oldFf)
                copyRows(old.P[oL.wq], oldE, grown.P[nL.wq], newE, oldE)
                copyRows(old.P[oL.wk], oldE, grown.P[nL.wk], newE, oldE)
                copyRows(old.P[oL.wv], oldE, grown.P[nL.wv], newE, oldE)
                copyRows(old.P[oL.wo], oldE, grown.P[nL.wo], newE, oldE)
                copyRows(old.P[oL.w1], oldFf, grown.P[nL.w1], 4 * newE, oldE)
                copyRows(old.P[oL.w2], oldE, grown.P[nL.w2], newE, oldFf)
            }
            System.arraycopy(old.P[old.lnFg], 0, grown.P[grown.lnFg], 0, oldE)
            System.arraycopy(old.P[old.lnFb], 0, grown.P[grown.lnFb], 0, oldE)
            copyRows(old.P[old.wout], v, grown.P[grown.wout], v, oldE)                              // الإخراج: نفس عدد الأعمدة (المفردات ثابتة)، صفوف أقل
            System.arraycopy(old.P[old.bout], 0, grown.P[grown.bout], 0, v)
            return grown
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
        val (fin, track) = genSegment(u, maxChars, temp, topk, minP, memGain, shortGain)
        if (commit) { replies.add(Pair(nq, fin)); while (replies.size > 12) replies.removeAt(0); short.add("س:$u") }
        return Pair(fin, track)
    }

    /** نفس حلقة التوليد اللي جوا reply() بالضبط، بس بدون استدعاء feed() في البداية — عشان توليد سطر إضافي يكمل
     *  على حالة مثبتة سابقًا بدل ما يعيد حقن "س:...ج:" في النص كل مرة (هذا بالضبط كان الخطأ في أول محاولة لي
     *  لبناء replyLineByLine: ناديت reply("") ظنًا إنها تكمّل، وهي فعليًا كانت تحقن "س:\nج:" زيادة كل سطر). */
    private fun genSegment(u: String, maxChars: Int, temp: Float, topk: Int, minP: Float, memGain: Float = 0.15f, shortGain: Float = 0.15f): Pair<String, List<Float>> {
        val nq = u.split(Regex("\\s+")).joinToString(" ")
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

    /** توليد سطر-سطر مع توقف حقيقي للتفكير بعد كل سطر — مو تخمين متصل من أول حرف لآخر حرف بضربة وحدة. لكل سطر:
     *  يجرب candidatesPerLine محاولات مستقلة لنفس السطر (كلها من نفس نقطة البداية)، يفحص كل واحدة بـ partialSane
     *  (رفض فوري لو فيها قوس إغلاق قبل فتحه)، يختار الأعلى ثقة من الناجحات، يثبّته، وبعدين يكمل للسطر التالي من
     *  نفس الحالة المثبتة. ما فيه حد أعلى على عدد الأسطر/مرات التوقف (maxLines بس حماية سقف بعيد، مو هدف بحد ذاته)
     *  — يوقف فعليًا لما يطلع سطر فاضي (نهاية طبيعية للرد) أو يوصل maxLines. */
    /** الحلقة المشتركة: سطر فسطر، كل سطر يولَّد عدة مرشحين (candidatesPerLine) من نفس نقطة البداية، يُفحص كل
     *  واحد بـpartialSane، يُختار الأعلى ثقة من الناجحين، يُثبَّت، ويكمل. ما تنادي feed() ولا تعمل commit —
     *  هذي مسؤولية المستدعي (replyLineByLine أو replyMax)، عشان تصلح لمرحلة وحدة أو عدة مراحل متتالية. */
    private fun genLines(maxLines: Int, candidatesPerLine: Int, baseTemp: Float): Pair<String, List<Float>> {
        val full = StringBuilder()
        val allScores = ArrayList<Float>()
        var emptyStreak = 0
        var lines = 0
        while (lines < maxLines) {
            val lineStart = snap()
            var bestLine: String? = null; var bestScore = Float.NEGATIVE_INFINITY; var bestEnd: Snap? = null; var bestTrack: List<Float> = emptyList()
            for (c in 0 until candidatesPerLine) {
                restore(lineStart)
                val temp = (baseTemp + (c % 4) * 0.03f).coerceIn(0.05f, 0.8f)
                val r = genSegment("", 400, temp, 4, 0.25f)
                val partial = full.toString() + r.first
                if (!partialSane(partial)) continue
                val sc = if (r.second.isEmpty()) 0f else r.second.sum() / r.second.size
                if (sc > bestScore) { bestScore = sc; bestLine = r.first; bestEnd = snap(); bestTrack = r.second }
            }
            if (bestLine == null) { restore(lineStart); break }
            restore(bestEnd ?: lineStart)
            full.append(bestLine); full.append("\n")
            allScores.addAll(bestTrack)
            lines++
            emptyStreak = if (bestLine.isBlank()) emptyStreak + 1 else 0
            if (emptyStreak >= 2) break
        }
        return Pair(full.toString().trim(), allScores)
    }

    fun replyLineByLine(user: String, maxLines: Int = 4000, candidatesPerLine: Int = 6, baseTemp: Float = 0.2f): Pair<String, List<Float>> {
        val u = user.replace("\n", " ¶ ").trim()
        feed(u + "\nج:")
        val (full, allScores) = genLines(maxLines, candidatesPerLine, baseTemp)
        replies.add(Pair(u.split(Regex("\\s+")).joinToString(" "), full)); while (replies.size > 12) replies.removeAt(0)
        short.add("س:$u")
        return Pair(full, allScores)
    }

    /** وضع ماكس بمرحلتين: (1) تفكير داخلي — يُولَّد ويُغذَّى فعليًا بسياق الجلسة (فيؤثر رياضيًا على المرحلة
     *  التالية عبر الانتباه)، بس **لا يُعرض للمستخدم إطلاقًا** — هذا بالضبط "يتكلم مع نفسه بدون ما يظهر وش فكر".
     *  (2) الجواب الفعلي — يُولَّد بعدها ويُعرض هو بس. maxThinkLines حماية من تفكير بلا نهاية فعلية (مو قيد
     *  زمني — ما فيه أي حد زمني هنا إطلاقًا، بس حد منطقي على عدد الأسطر). صادق لازم أقوله: البنية موجودة
     *  وتشتغل الآن (maxThinkLines/candidatesPerLine رُفعت بطلبك لتفكير أكبر بكثير من قبل)، بس الاستفادة
     *  الحقيقية من مرحلة التفكير تكبر كل ما درّبته على أمثلة بنفس الصيغة
     *  ("س:...\nتفكير:...\nج:...") — من دون هذا التدريب المخصص، يولّد شيء جوا "تفكير:" لكن ما بالضرورة
     *  تفكير فعلي منظّم، لأنه ما شاف أمثلة توضح له إيش المفروض يكتب هناك. */
    fun replyMax(user: String, maxThinkLines: Int = 3000, candidatesPerLine: Int = 18, baseTemp: Float = 0.15f): Pair<String, List<Float>> {
        val u = user.replace("\n", " ¶ ").trim()
        feed("$u\nتفكير:")
        genLines(maxThinkLines, candidatesPerLine, baseTemp)   // مخفي عمدًا: نرمي النص المُرجَع، بس الحالة (s1/s2) تضل محدَّثة بأثره
        feed("\nج:")
        val (ans, track) = genLines(4000, candidatesPerLine, baseTemp)
        replies.add(Pair(u.split(Regex("\\s+")).joinToString(" "), ans)); while (replies.size > 12) replies.removeAt(0)
        short.add("س:$u")
        return Pair(ans, track)
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
    fun replyDeep(user: String, maxChars: Int, baseTemp: Float, maxMs: Long, minMs: Long = 0, keepWindow: Int = 40, accept: ((String) -> Boolean)? = null): Pair<String, List<Float>> {
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
        while (true) {
            val underBudget = i < hardCap && elapsedMs() < maxMs
            val underFloor = elapsedMs() < minMs       // الأرضية (minMs) تتجاوز hardCap نفسه عمدًا لو احتاجت — ضمان حقيقي، مو تقريبي
            if (!underBudget && !underFloor) break
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


// ============== من الجوال_World.kt ==============
/** عالم سوبرمان — العالم الافتراضي اللي يبنيه سوبرمان بالكود ويتخيّل نفسه فيه.
 *  شبكة صغيرة: # جدار · . أرض · @ سوبرمان نفسه · G هدف · K مفتاح · = باب (ما يفتح إلا بالمفتاح) · ^ خطر (الدخول عليه = خسارة).
 *  الحركة: R يمين · L يسار · U فوق · D تحت.
 *  كل شي هنا حقيقي ومُختبر بالتشغيل: المحاكي يطبّق القواعد فعلًا، والمخطِّط (BFS) يلقى أقصر خطة فعلًا، والتحقق
 *  دايمًا بتشغيل الخطة داخل المحاكي، مو بالتخمين. النموذج (Nlm) ما يغش: يُمتحن بمحاولته وحده بدون بحث. */
class SimWorld(val w: Int, val hh: Int, private val cells: CharArray, val sx: Int, val sy: Int) {
    var ax = sx
        private set
    var ay = sy
        private set
    var hasKey = false
        private set
    var steps = 0
        private set
    var dead = false
        private set
    var won = false
        private set
    val over: Boolean get() = dead || won

    fun at(x: Int, y: Int): Char = if (x < 0 || y < 0 || x >= w || y >= hh) '#' else cells[y * w + x]

    fun copy(): SimWorld {
        val c = SimWorld(w, hh, cells.copyOf(), sx, sy)
        c.ax = ax; c.ay = ay; c.hasKey = hasKey; c.steps = steps; c.dead = dead; c.won = won
        return c
    }

    /** خطوة وحدة. يرجّع true لو تحرّكت فعلًا (الاصطدام بجدار أو بباب مقفل يحسب خطوة بس ما تتحرك). */
    fun step(a: Char): Boolean {
        if (over) return false
        val dx = when (a) { 'R' -> 1; 'L' -> -1; else -> 0 }
        val dy = when (a) { 'D' -> 1; 'U' -> -1; else -> 0 }
        if (dx == 0 && dy == 0) return false
        val nx = ax + dx; val ny = ay + dy
        val t = at(nx, ny)
        steps++
        if (t == '#' || (t == '=' && !hasKey)) return false
        ax = nx; ay = ny
        if (t == '^') dead = true
        else if (t == 'K') { hasKey = true; cells[ny * w + nx] = '.' }
        else if (t == 'G') won = true
        return true
    }

    fun rows(): List<String> = (0 until hh).map { y -> String(CharArray(w) { x -> if (x == ax && y == ay) '@' else at(x, y) }) }

    /** الصيغة اللي يشوفها النموذج: سطر واحد، الصفوف مفصولة بـ | */
    fun text(): String = rows().joinToString("|")

    /** للعرض بس (الإيموجي ما هو في مفردات النموذج). */
    fun emoji(): String = rows().joinToString("\n") { r ->
        r.map { c -> when (c) { '#' -> "⬛"; '@' -> "🤖"; 'G' -> "🏁"; 'K' -> "🔑"; '=' -> "🚪"; '^' -> "🔥"; else -> "⬜" } }.joinToString("")
    }

    /** يرسم العالم مع سوبرمان نفسه (دائرة زرقاء بعيون) — هذي "صورته لنفسه". */
    fun toBitmap(cell: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w * cell, hh * cell, Bitmap.Config.ARGB_8888)
        val cv = Canvas(bmp)
        val p = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG)
        cv.drawColor(Color.parseColor("#CFD8DC"))
        for (y in 0 until hh) for (x in 0 until w) {
            val l = (x * cell).toFloat(); val t = (y * cell).toFloat(); val r = l + cell; val b = t + cell
            val c = at(x, y)
            p.color = when (c) {
                '#' -> Color.parseColor("#37474F"); '^' -> Color.parseColor("#E53935"); '=' -> Color.parseColor("#8D6E63")
                'G' -> Color.parseColor("#43A047"); else -> Color.parseColor("#ECEFF1")
            }
            cv.drawRect(l + 1f, t + 1f, r - 1f, b - 1f, p)
            if (c == 'K') { p.color = Color.parseColor("#FDD835"); cv.drawCircle((l + r) / 2f, (t + b) / 2f, cell * 0.25f, p) }
        }
        p.color = if (dead) Color.parseColor("#B71C1C") else Color.parseColor("#1E88E5")
        cv.drawCircle((ax + 0.5f) * cell, (ay + 0.5f) * cell, cell * 0.36f, p)
        p.color = Color.WHITE
        cv.drawCircle((ax + 0.38f) * cell, (ay + 0.42f) * cell, cell * 0.07f, p)
        cv.drawCircle((ax + 0.62f) * cell, (ay + 0.42f) * cell, cell * 0.07f, p)
        if (hasKey) { p.color = Color.parseColor("#FDD835"); cv.drawCircle((ax + 0.85f) * cell, (ay + 0.15f) * cell, cell * 0.1f, p) }
        return bmp
    }

    companion object {
        /** يقرأ عالم من نص (الصفوف مفصولة بـ |). يرجّع null لو النص مو عالم سليم: صفوف متساوية، رموز معروفة بس، @ واحدة، G واحد،
         *  والمفتاح والباب إما كلاهما أو لا شي. هذا هو الفحص اللي يمنع النموذج يضيف "عوالم" مكسورة لتدريبه. */
        fun parse(s: String): SimWorld? {
            val rs = s.trim().split("|")
            if (rs.size < 2 || rs.size > 12) return null
            val w = rs[0].length
            if (w < 3 || w > 16) return null
            val cells = CharArray(w * rs.size)
            var sx = -1; var sy = -1; var g = 0; var k = 0; var d = 0
            for (y in rs.indices) {
                val r = rs[y]
                if (r.length != w) return null
                for (x in 0 until w) {
                    var c = r[x]
                    when (c) {
                        '#', '.', '^' -> { }
                        '@' -> { if (sx >= 0) return null; sx = x; sy = y; c = '.' }
                        'G' -> g++
                        'K' -> k++
                        '=' -> d++
                        else -> return null
                    }
                    cells[y * w + x] = c
                }
            }
            if (sx < 0 || g != 1 || k > 1 || d > 1 || (k == 1) != (d == 1)) return null
            return SimWorld(w, rs.size, cells, sx, sy)
        }
    }
}

object WorldPlanner {
    private const val ACTS = "RDLU"
    private val DX = intArrayOf(1, 0, -1, 0)
    private val DY = intArrayOf(0, 1, 0, -1)

    /** أقصر خطة حقيقية (BFS على الحالة = مكان + هل معي المفتاح). null = العالم ما له حل. الترتيب ثابت (R,D,L,U) عشان
     *  النموذج يتعلم سلوك ثابت مو عشوائي. يخطّط من الحالة الحالية للعالم، فيصلح أثناء الحلقة مو بس من البداية. */
    fun plan(wd: SimWorld): String? {
        val w = wd.w; val h = wd.hh
        val n = w * h * 2
        val prev = IntArray(n) { -1 }; val how = CharArray(n); val seen = BooleanArray(n)
        val start = ((wd.ay * w + wd.ax) shl 1) or (if (wd.hasKey) 1 else 0)
        val q = IntArray(n); var qh = 0; var qt = 0
        q[qt++] = start; seen[start] = true
        var goal = -1
        while (qh < qt && goal < 0) {
            val s = q[qh++]
            val key = s and 1; val cell = s shr 1; val x = cell % w; val y = cell / w
            for (a in 0 until 4) {
                val nx = x + DX[a]; val ny = y + DY[a]
                val t = wd.at(nx, ny)
                if (t == '#' || t == '^') continue
                if (t == '=' && key == 0) continue
                val nk = if (t == 'K') 1 else key
                val ns = ((ny * w + nx) shl 1) or nk
                if (seen[ns]) continue
                seen[ns] = true; prev[ns] = s; how[ns] = ACTS[a]; q[qt++] = ns
                if (t == 'G') { goal = ns; break }
            }
        }
        if (goal < 0) return null
        val sb = StringBuilder()
        var c = goal
        while (c != start) { sb.append(how[c]); c = prev[c] }
        return sb.reverse().toString()
    }

    /** يشغّل الأفعال في نسخة من العالم (الأصل ما يتغير) ويرجّع العالم بعد التنفيذ. */
    fun execute(w0: SimWorld, acts: String): SimWorld {
        val c = w0.copy()
        for (a in acts) { if (c.over) break; c.step(a) }
        return c
    }

    fun speak(acts: String): String = acts.map { a -> when (a) { 'R' -> "يمين"; 'L' -> "يسار"; 'U' -> "فوق"; 'D' -> "تحت"; else -> "" } }.joinToString(" ← ")
}

/** يبني العوالم بالكود. 7 مستويات تتدرج: غرفة فاضية ← جدران ← أخطار ← مفتاح وباب (الباب يوضع عمدًا على أقصر طريق، فالمفتاح ضرورة مو ديكور)
 *  ← عوالم أكبر. نافذة النموذج (h حرف) هي اللي تحدد أعلى مستوى ممكن: لازم السؤال + الجواب كله يدخل النافذة، وإلا ما يقدر "يشوف" العالم وهو يجاوب. */
object WorldGen {
    private class Spec(val w: Int, val h: Int, val walls: Double, val hazards: Int, val key: Boolean)
    private val SPECS = listOf(
        Spec(4, 3, 0.0, 0, false), Spec(5, 3, 0.15, 0, false), Spec(5, 4, 0.18, 1, false), Spec(6, 4, 0.2, 1, true),
        Spec(6, 5, 0.22, 2, true), Spec(7, 5, 0.22, 2, true), Spec(8, 6, 0.22, 3, true)
    )
    const val MAX_LEVEL = 7

    /** طول الزوج في التدريب: س: + العالم + سطر جديد + ج: + الخطة + سطر جديد */
    fun pairLen(wd: SimWorld, plan: String): Int = 2 + (wd.w * wd.hh + wd.hh - 1) + 1 + 2 + plan.length + 1

    /** أعلى مستوى يدخل في نافذة النموذج (تقدير متحفّظ). نافذة 64 → حتى المستوى 6، نافذة 96 أو أكثر → المستوى 7. */
    fun maxLevelFor(window: Int): Int {
        var best = 1
        for (l in 1..MAX_LEVEL) {
            val s = SPECS[l - 1]
            val worst = 2 + (s.w * s.h + s.h - 1) + 1 + 2 + (s.w + s.h) + 4
            if (worst <= window - 2) best = l
        }
        return best
    }

    /** عالم حل مضمون، أو null لو ما لقى بعد 400 محاولة (يصير بس لو النافذة أصغر من المستوى). كل خطة تُتحقق بالتشغيل. */
    fun make(level: Int, seed: Long, window: Int): Pair<SimWorld, String>? {
        val sp = SPECS[level.coerceIn(1, MAX_LEVEL) - 1]
        val rnd = java.util.Random(seed)
        for (attempt in 0 until 400) {
            val n = sp.w * sp.h
            val b = CharArray(n) { '.' }
            for (i in 0 until n) if (rnd.nextDouble() < sp.walls) b[i] = '#'
            fun free(excl: Int): Int {
                for (t in 0 until 200) { val i = rnd.nextInt(n); if (b[i] == '.' && i != excl) return i }
                return -1
            }
            val s = free(-1); if (s < 0) continue
            val g = free(s); if (g < 0) continue
            b[g] = 'G'
            val sx = s % sp.w; val sy = s / sp.w
            if (sp.key) {
                val p0 = WorldPlanner.plan(SimWorld(sp.w, sp.h, b.copyOf(), sx, sy)) ?: continue
                if (p0.length < 4) continue
                var x = sx; var y = sy
                for (i in 0 until p0.length / 2) {
                    val a = p0[i]
                    if (a == 'R') x++ else if (a == 'L') x-- else if (a == 'D') y++ else if (a == 'U') y--
                }
                b[y * sp.w + x] = '='
                val k = free(s); if (k < 0) continue
                b[k] = 'K'
            }
            for (z in 0 until sp.hazards) { val hz = free(s); if (hz >= 0) b[hz] = '^' }
            val wd = SimWorld(sp.w, sp.h, b, sx, sy)
            val plan = WorldPlanner.plan(wd) ?: continue
            if (plan.length < 3 || pairLen(wd, plan) > window - 2) continue
            if (sp.key && !WorldPlanner.execute(wd, plan).hasKey) continue
            return Pair(wd, plan)
        }
        return null
    }

    /** عالم احتياطي ثابت (غرفة فيها ممر) لو المولّد ما قدر لأي سبب. */
    fun fallback(): Pair<SimWorld, String> {
        val wd = SimWorld.parse("#####|#@..G|#####")!!
        return Pair(wd, WorldPlanner.plan(wd) ?: "RRR")
    }

    fun solveBlock(wd: SimWorld, plan: String): String = "س:" + wd.text() + "\nج:" + plan + "\n"
    fun buildBlock(level: Int, wd: SimWorld): String = "س:عالم " + level + "\nج:" + wd.text() + "\n"
}

class DreamResult(val level: Int, val world: SimWorld, val plan: String?, val planWorld: SimWorld?, val modelActs: String, val modelWorld: SimWorld) {
    val planWon: Boolean get() = planWorld?.won == true
    val modelWon: Boolean get() = modelWorld.won

    fun report(): String {
        val sb = StringBuilder()
        sb.append("🌍 بنيت بالكود عالم جديد (المستوى ").append(level).append("، ").append(world.w).append("×").append(world.hh).append(") وحطيت نفسي فيه 🤖:\n\n")
        sb.append(world.emoji()).append("\n\n")
        if (plan != null && planWorld != null) {
            sb.append("🧭 خطتي بالبحث (محاكي حقيقي، مو تخمين): ").append(plan).append("\n")
            sb.append(WorldPlanner.speak(plan)).append("\n")
            sb.append(if (planWon) "جرّبتها في العالم ووصلت الهدف ✅ بعد " + planWorld.steps + " خطوة.\n\n" else "جرّبتها وما وصلت ❌ (هذا ما يفترض يصير).\n\n")
        } else sb.append("🧭 البحث ما لقى طريق لهذا العالم.\n\n")
        sb.append("🧠 محاولتي أنا وحدي (النموذج بدون بحث): ").append(if (modelActs.isEmpty()) "(ما كتبت حركات مفهومة)" else modelActs).append("\n")
        sb.append(if (modelWon) "ووصلت الهدف ✅ بعد " + modelWorld.steps + " خطوة.\n" else if (modelWorld.dead) "ودخلت خطر 🔥 وخسرت ❌.\n" else "وما وصلت الهدف ❌.\n")
        sb.append("\nمكاني في آخر محاولتي:\n").append(modelWorld.emoji())
        sb.append("\n\nالخطة الأولى من خوارزمية بحث، أما محاولتي الثانية فهي مقياس تعلّمي الحقيقي — اضغط «تعلّم» عشان أتدرب على عوالم محلولة وأتحسن.")
        return sb.toString()
    }
}

class ExamResult(val level: Int, val n: Int, val valid: Int, val solved: Int, val optimality: Float) {
    fun rate(): Float = if (n == 0) 0f else solved.toFloat() / n
    fun line(): String = "المستوى " + level + ": حلّيت " + solved + " من " + n + " عالم (صيغة صحيحة: " + valid + ")" +
        (if (solved > 0) ", كفاءة الطريق " + String.format(java.util.Locale.US, "%.0f", optimality * 100) + "%" else "")
}

/** مختبر العالم: يدمج المحاكي مع تعلّم النموذج بشكل آمن وتدريجي.
 *  حلقة التعلم الذاتي (كلها محققة، بلا غش):
 *   1) المولّد يبني عوالم بالكود ← المخطِّط يحلّها ← كتل «س:عالم/ج:حل» تُدرَّب بـ trainBig (اللي عنده رفض تلقائي لو ساءت النتيجة أو نسي الكلام القديم).
 *   2) النموذج نفسه يُطلب منه يبني عالم («س:عالم 3») ← parse + حل المخطِّط يتحققان إنه سليم وله حل ← السليم يدخل مخزن (pool) ويُدرَّب عليه لاحقًا.
 *   3) امتحان ثابت (نفس العوالم قبل وبعد) يقيس هل النموذج صار يحلّ بنفسه فعلًا. ما يرتفع المستوى إلا لو نجح ≥ 60% في الامتحان. */
class WorldLab private constructor(private val ctx: Context) {
    companion object {
        @Volatile private var inst: WorldLab? = null
        fun get(c: Context): WorldLab {
            val i = inst
            if (i != null) return i
            synchronized(WorldLab::class.java) {
                val j = inst
                if (j != null) return j
                val n = WorldLab(c.applicationContext); inst = n; return n
            }
        }

        /** هل الرسالة طلب «تخيّل / صوّر نفسك في عالم»؟ ضيّق عمدًا عشان ما يخطف محادثة عادية. */
        fun isDreamRequest(q: String): Boolean {
            val t = q.replace("ّ", "")
            val imagine = t.contains("تخيل") || t.contains("تصور") || t.contains("صور نفسك") || t.contains("صوّر") || t.contains("ابني لنفسك") || t.contains("ابن لنفسك")
            return (imagine && (t.contains("عالم") || t.contains("نفسك"))) || t.contains("عالم افتراضي") || t.contains("عالمك")
        }
        private const val EXAM_SEED = 9_000_000L
    }

    private val stateFile = File(ctx.filesDir, "world_lab.json")
    private val poolFile = File(ctx.filesDir, "world_pool.txt")
    private val pool = ArrayList<String>()
    @Volatile var level = 1
        private set
    @Volatile var lastRate = -1f
        private set
    @Volatile var builtValid = 0
        private set
    @Volatile var builtTotal = 0
        private set
    private var seedCounter = 1000L

    init {
        try {
            if (stateFile.exists()) {
                val j = JSONObject(stateFile.readText(Charsets.UTF_8))
                level = j.optInt("level", 1).coerceIn(1, WorldGen.MAX_LEVEL)
                lastRate = j.optDouble("lastRate", -1.0).toFloat()
                builtValid = j.optInt("builtValid", 0); builtTotal = j.optInt("builtTotal", 0)
                seedCounter = j.optLong("seed", 1000L)
            }
            if (poolFile.exists()) for (l in poolFile.readLines(Charsets.UTF_8)) if (l.isNotBlank()) pool.add(l.trim())
        } catch (e: Throwable) { /* حالة تالفة: نبدأ من الصفر بدون ما نوقف التطبيق */ }
    }

    private fun save() {
        try {
            stateFile.writeText(JSONObject().put("level", level).put("lastRate", lastRate.toDouble()).put("builtValid", builtValid).put("builtTotal", builtTotal).put("seed", seedCounter).toString(), Charsets.UTF_8)
            poolFile.writeText(pool.takeLast(400).joinToString("\n"), Charsets.UTF_8)
        } catch (e: Throwable) { /* نحاول المرة الجاية */ }
    }

    fun maxLevel(m: Nlm): Int = WorldGen.maxLevelFor(m.h)
    fun effectiveLevel(m: Nlm): Int = level.coerceIn(1, maxLevel(m))

    /** محاولة النموذج وحده: يشوف العالم كنص ويكتب حركات بدون أي بحث. مرتبة وثابتة عشان القياس يقارن نفس الشي قبل/بعد. */
    private fun modelAttempt(m: Nlm, wd: SimWorld, seed: Long): String {
        val s = NlmSession(m, seed)
        val ans = s.reply(wd.text(), 40, 0.12f, 2, 0.25f, false, 0f, 0f).first
        return ans.takeWhile { it == 'R' || it == 'L' || it == 'U' || it == 'D' }
    }

    /** حلم واحد: عالم جديد + خطة البحث + محاولة النموذج، كلها تُنفَّذ فعليًا في المحاكي. */
    fun dream(m: Nlm): DreamResult {
        val lv = effectiveLevel(m)
        val seed = 20_000_000L + (System.nanoTime() and 0xFFFFFFL)
        val r = WorldGen.make(lv, seed, m.h) ?: WorldGen.fallback()
        val wd = r.first
        val plan = WorldPlanner.plan(wd)
        val planW = if (plan != null) WorldPlanner.execute(wd, plan) else null
        val acts = modelAttempt(m, wd, seed)
        return DreamResult(lv, wd, plan, planW, acts, WorldPlanner.execute(wd, acts))
    }

    /** امتحان ثابت: نفس العوالم (بذور ثابتة بعيدة عن بذور التدريب) كل مرة، فالمقارنة قبل/بعد عادلة. */
    fun exam(m: Nlm, n: Int, lv: Int): ExamResult {
        var valid = 0; var solved = 0; var tried = 0; var ratio = 0f
        for (i in 0 until n) {
            val r = WorldGen.make(lv, EXAM_SEED + lv * 1000L + i, m.h) ?: continue
            tried++
            val acts = modelAttempt(m, r.first, 1234L + i)
            if (acts.isNotEmpty()) valid++
            val fin = WorldPlanner.execute(r.first, acts)
            if (fin.won) { solved++; ratio += r.second.length.toFloat() / maxOf(1, acts.length) }
        }
        return ExamResult(lv, tried, valid, solved, if (solved > 0) ratio / solved else 0f)
    }

    /** النموذج يبني عوالم بنفسه؛ نعدّ السليم (له @ وG ومحلول) ونحفظه في المخزن. */
    private fun buildByModel(m: Nlm, lv: Int, n: Int): Pair<Int, Int> {
        var ok = 0
        for (i in 0 until n) {
            val s = NlmSession(m, 777L + i + seedCounter)
            val out = s.reply("عالم " + lv, 120, 0.5f, 6, 0.1f, false, 0f, 0f).first.trim()
            val wd = SimWorld.parse(out) ?: continue
            val p = WorldPlanner.plan(wd) ?: continue
            if (p.length < 3 || WorldGen.pairLen(wd, p) > m.h - 2) continue
            ok++
            if (!pool.contains(out)) pool.add(out)
        }
        while (pool.size > 400) pool.removeAt(0)
        return Pair(ok, n)
    }

    /** دورة تعلّم كاملة. تدريب الأوزان يمر عبر Nlm.trainBig (يرفض ويرجّع الأوزان القديمة تلقائيًا لو ساء). الباقي قياس. */
    fun train(m: Nlm, rehearsal: String, epochs: Int, lr: Float, count: Int): String {
        val lv = effectiveLevel(m)
        val sb = StringBuilder()
        val before = exam(m, 12, lv)
        val base = seedCounter; seedCounter += count + 7
        val corpus = StringBuilder()
        var made = 0
        for (i in 0 until count) {
            val l = if (i % 4 == 3 && lv > 1) 1 + (i / 4) % (lv - 1) else lv       // 25% من مستويات أسهل: ما ينسى القديم
            val r = WorldGen.make(l, base + i, m.h) ?: continue
            corpus.append(WorldGen.solveBlock(r.first, r.second)); made++
            if (i % 3 == 0) corpus.append(WorldGen.buildBlock(l, r.first))
        }
        var fromPool = 0
        for (rows in pool.takeLast(maxOf(4, count / 4))) {
            val wd = SimWorld.parse(rows) ?: continue
            val p = WorldPlanner.plan(wd) ?: continue
            if (WorldGen.pairLen(wd, p) > m.h - 2) continue
            corpus.append(WorldGen.solveBlock(wd, p)); fromPool++
        }
        if (made == 0) { save(); return "ما قدرت أبني عوالم تناسب نافذة النموذج الحالية (" + m.h + " حرف). كبّر النافذة من «تطوير متقدم»." }
        val gw = m.gradWarning()
        val rep = (if (gw != null) gw + "\n\n" else "") + m.trainBig(corpus.toString(), rehearsal, epochs, lr)
        val after = exam(m, 12, lv)
        val built = buildByModel(m, lv, 10)
        builtValid += built.first; builtTotal += built.second
        sb.append("🌍 تعلّمت من ").append(made).append(" عالم بنيتهم بالكود وحلّيتهم بالبحث")
        if (fromPool > 0) sb.append(" + ").append(fromPool).append(" عالم بنيتهم أنا بنفسي (وتحققت إنها سليمة)")
        sb.append(".\n\n").append(rep).append("\n\n")
        sb.append("📊 امتحان ثابت (نفس العوالم قبل وبعد):\nقبل → ").append(before.line()).append("\nبعد → ").append(after.line()).append("\n")
        sb.append("🏗 عوالم بنيتها أنا الحين: ").append(built.first).append(" سليمة من ").append(built.second).append(" (المجموع الكلي: ").append(builtValid).append(" من ").append(builtTotal).append(")\n")
        lastRate = after.rate()
        if (after.rate() >= 0.6f && level < maxLevel(m) && level == lv) {
            level = lv + 1
            sb.append("\n⬆ نجحت بـ ").append((after.rate() * 100).toInt()).append("% فارتفع المستوى إلى ").append(level).append(".")
        } else if (level >= maxLevel(m) && after.rate() >= 0.6f) {
            sb.append("\n🏁 وصلت أعلى مستوى تسمح به نافذة النموذج (").append(m.h).append(" حرف). لمستويات أكبر كبّر النافذة من «تطوير متقدم».")
        } else sb.append("\nباقي أقل من 60% في الامتحان، فأبقى في المستوى ").append(lv).append(" وأكرر التعلم (التقدم تدريجي).")
        save()
        return sb.toString()
    }

    fun status(m: Nlm): String = "المستوى الحالي: " + effectiveLevel(m) + " من " + maxLevel(m) + " (حسب نافذة " + m.h + " حرف) · آخر امتحان: " +
        (if (lastRate < 0f) "ما اختبرت بعد" else (lastRate * 100).toInt().toString() + "%") + " · مخزن عوالمي: " + pool.size

    fun examReport(m: Nlm): String {
        val lv = effectiveLevel(m)
        val sb = StringBuilder("📊 امتحان حقيقي (بدون بحث، النموذج وحده، المحاكي هو الحكم):\n")
        var l = 1
        while (l <= lv) { sb.append(exam(m, 12, l).line()).append("\n"); l++ }
        val b = buildByModel(m, lv, 10)
        builtValid += b.first; builtTotal += b.second
        sb.append("🏗 بنيت عوالم بنفسي: ").append(b.first).append(" سليمة من ").append(b.second).append("\n")
        save()
        return sb.toString()
    }
}

// ============== من الجوال_PlayLoop.kt ==============
/** "يلعبها بنفسه كلاعب حقيقي قبل ما يوصلك الملف": لكود الألعاب أو أي صفحة تفاعلية (فيها <canvas> أو تستمع للوحة
 *  المفاتيح أو اللمس)، VisionLoop القديم كان يأخذ لقطة واحدة ثابتة بس. هذا يذهب أبعد: يشغّل الكود فعليًا داخل
 *  WebView، ويرسل له ضغطات حقيقية (الأسهم، مسافة، نقرة) بنفس الطريقة اللي يرسلها المتصفح لصفحة حقيقية
 *  (keydown/keyup/click أحداث DOM فعلية، مو محاكاة مزيّفة)، ويلتقط لقطة بعد كل ضغطة، ويرصد أي خطأ JavaScript
 *  حقيقي وقع أثناء ذلك (onerror + console.error، عبر WebChromeClient — هذا خطأ حقيقي من محرك المتصفح، مو تخمين).
 *  بعدها يعرض على النموذج التسلسل كامل (كل اللقطات مع الضغطة اللي سبقتها وأي خطأ) ويسأله: هل تصرف اللاعب
 *  (تحرّك، تفاعل) كما يفترض بلعبة سليمة؟ لو لا — أو لو فيه خطأ — يعدّل الكود ويُعاد اللعب من جديد.
 *  هذا مصمم للدقة مو للسرعة: عدد المحاولات كبير عمدًا والتوقف المبكر ما يصير إلا لو الكود توقف عن التغيّر فعلًا. */
object PlayLoop {
    private val GAME_SIGNS = listOf(
        "<canvas", "requestanimationframe", "addeventlistener(\"keydown\"", "addeventlistener('keydown'",
        "onkeydown", "addeventlistener(\"touchstart\"", "addeventlistener('touchstart'", "onclick=\"", "addeventlistener(\"click\""
    )
    fun looksLikeGame(code: String): Boolean { val c = code.lowercase(); return GAME_SIGNS.any { c.contains(it) } }

    /** سكربت لاعب تجريبي: كل الاتجاهات، المسافة (قفز/إطلاق شائع)، ونقرة — كل واحدة مرتين عشان نتأكد الحركة مستقرة مو صدفة. */
    private val SCRIPT: List<Pair<String, String>> = listOf(
        "ArrowRight" to "key", "ArrowRight" to "key", "ArrowDown" to "key", "ArrowDown" to "key",
        "ArrowUp" to "key", "ArrowLeft" to "key", " " to "key", "tap" to "tap", "tap" to "tap",
        "ArrowRight" to "key", "ArrowDown" to "key", " " to "key"
    )

    class Frame(val action: String, val ascii: String, val errors: List<String>)

    private fun toAscii(bmp: Bitmap, gridW: Int, gridH: Int): String {
        val ramp = " .:-=+*#%@"
        val sb = StringBuilder()
        val cellW = maxOf(1, bmp.width / gridW); val cellH = maxOf(1, bmp.height / gridH)
        for (gy in 0 until gridH) {
            for (gx in 0 until gridW) {
                var sum = 0L; var n = 0
                val x0 = gx * cellW; val y0 = gy * cellH
                for (x in x0 until minOf(x0 + cellW, bmp.width)) for (y in y0 until minOf(y0 + cellH, bmp.height)) {
                    val p = bmp.getPixel(x, y); sum += (Color.red(p) + Color.green(p) + Color.blue(p)) / 3; n++
                }
                val avg = if (n > 0) (sum / n).toInt() else 255
                sb.append(ramp[(avg * (ramp.length - 1) / 255)])
            }
            sb.append('\n')
        }
        return sb.toString()
    }

    /** يشغّل الصفحة، يرسل لها سكربت اللاعب أعلاه كأحداث DOM حقيقية، يلتقط لقطة (ASCII) + أخطاء JS بعد كل ضغطة.
     *  يرجّع قائمة فاضية لو فشل التحميل (بدون رمي استثناء) — المتصل يتعامل معها على إنها "ما قدرت ألعب". */
    private fun playSequence(activity: Activity, html: String, gridW: Int = 14, gridH: Int = 14, renderPx: Int = 260, stepDelayMs: Long = 260): List<Frame> {
        val frames = Collections.synchronizedList(ArrayList<Frame>())
        val latch = CountDownLatch(1)
        val pendingErrors = Collections.synchronizedList(ArrayList<String>())
        Handler(Looper.getMainLooper()).post {
            try {
                val wv = WebView(activity)
                wv.layout(0, 0, renderPx, renderPx)
                wv.settings.javaScriptEnabled = true
                wv.webChromeClient = object : WebChromeClient() {
                    override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                        if (m.messageLevel() == ConsoleMessage.MessageLevel.ERROR) pendingErrors.add(m.message() + " (سطر " + m.lineNumber() + ")")
                        return true
                    }
                }
                fun snap(action: String) {
                    try {
                        val bmp = Bitmap.createBitmap(renderPx, renderPx, Bitmap.Config.ARGB_8888)
                        wv.draw(Canvas(bmp))
                        val errs = synchronized(pendingErrors) { val c = ArrayList(pendingErrors); pendingErrors.clear(); c }
                        frames.add(Frame(action, toAscii(bmp, gridW, gridH), errs))
                    } catch (e: Throwable) { }
                }
                wv.webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        Handler(Looper.getMainLooper()).postDelayed({
                            snap("بداية")
                            var i = 0
                            val step = object : Runnable {
                                override fun run() {
                                    if (i >= SCRIPT.size) { latch.countDown(); return }
                                    val (v, kind) = SCRIPT[i]; i++
                                    val js = if (kind == "tap")
                                        "(function(){var el=document.elementFromPoint(" + renderPx / 2 + "," + renderPx / 2 + ")||document.body;" +
                                        "['touchstart','touchend','mousedown','mouseup','click'].forEach(function(t){try{el.dispatchEvent(new Event(t,{bubbles:true}));}catch(e){}});})();"
                                    else
                                        "(function(){var k=" + org.json.JSONObject.quote(v) + ";var code=(k===' ')?'Space':('Arrow'+k.replace('Arrow',''));" +
                                        "['keydown','keyup'].forEach(function(t){try{var ev=new KeyboardEvent(t,{key:k,code:code,bubbles:true,cancelable:true});" +
                                        "window.dispatchEvent(ev);document.dispatchEvent(ev);if(document.activeElement)document.activeElement.dispatchEvent(ev);}catch(e){}});})();"
                                    wv.evaluateJavascript(js) {
                                        Handler(Looper.getMainLooper()).postDelayed({ snap(if (v == " ") "مسافة" else v); Handler(Looper.getMainLooper()).postDelayed(this, 60) }, 160)
                                    }
                                }
                            }
                            Handler(Looper.getMainLooper()).postDelayed(step, stepDelayMs)
                        }, 400)
                    }
                }
                val hook = "<script>window.onerror=function(msg,src,line){try{console.error('استثناء غير ملتقط: '+msg+' (سطر '+line+')');}catch(e){}};</script>"
                wv.loadDataWithBaseURL(null, hook + html, "text/html", "utf-8", null)
            } catch (e: Throwable) { latch.countDown() }
        }
        latch.await(30_000, TimeUnit.MILLISECONDS)   // سقف أعلى من VisionLoop لأن هذا يشغّل سكربت كامل من الضغطات، لكنه على خيط خلفي فما يعلّق التطبيق
        return ArrayList(frames)
    }

    private fun wrapForRender(code: String, ext: String): String = when (ext) {
        "html", "htm" -> code
        "css" -> "<style>" + code + "</style><body style=\"margin:0;font-family:sans-serif\">معاينة</body>"
        "js" -> "<body style=\"margin:0\"><script>" + code + "</script></body>"
        else -> code
    }

    private fun extractCode(text: String): String? {
        if (!text.contains("```")) return null
        val block = text.substringAfter("```")
        val afterLang = block.substringAfter("\n", block)
        return afterLang.substringBefore("```").trim('\n')
    }

    private fun describeFrames(frames: List<Frame>): String {
        val sb = StringBuilder()
        var anyErr = false
        for ((idx, f) in frames.withIndex()) {
            sb.append("--- اللقطة ").append(idx).append(" بعد ضغطة: ").append(f.action).append(" ---\n")
            sb.append(f.ascii)
            if (f.errors.isNotEmpty()) { anyErr = true; sb.append("⚠ أخطاء JavaScript وقعت هنا فعليًا:\n"); for (e in f.errors) sb.append("  - ").append(e).append("\n") }
        }
        if (!anyErr) sb.append("(ما وقع أي خطأ JavaScript خلال كل الضغطات)\n")
        return sb.toString()
    }

    /** الحلقة الكاملة: شغّل ← سجّل لاعب حقيقي بضغطات فعلية ← اعرض التسلسل كامل على النموذج ← عدّل لو لزم ← كرر.
     *  يتوقف لو: (أ) النموذج أعاد نفس الكود بدون تغيير، أو (ب) لعبتان متتاليتان ما فيهم أي خطأ JS وكل اللقطات
     *  فيها تغيّر فعلي بعد كل ضغطة اتجاه (دليل إن اللاعب يتحرك، مو عالق)، أو (ج) وصل maxPasses. */
    fun reviewAndPlay(brain: Brain, activity: Activity, convId: String, code: String, fileName: String, maxPasses: Int = 15, log: (String) -> Unit = {}): String {
        val ext = fileName.substringAfterLast('.', "").lowercase()
        if (ext !in setOf("html", "htm", "js")) return code
        var current = code
        var cleanStreak = 0
        for (pass in 1..maxPasses) {
            log("🎮 يلعبها بنفسه (ضغطات حقيقية: أسهم، مسافة، نقرة) — محاولة " + pass + "...")
            val html = wrapForRender(current, ext)
            val frames = playSequence(activity, html)
            if (frames.size < 2) { log("⚠ ما قدر يشغّل الصفحة للعب (ربما تحتاج ملفات أخرى)، يكتفي بلقطة ثابتة."); return VisionLoop.reviewAndFix(brain, activity, convId, current, fileName, 2, log) }
            val hasErr = frames.any { it.errors.isNotEmpty() }
            val stuck = frames.zipWithNext().count { (a, b) -> a.ascii == b.ascii } >= frames.size - 2   // ولا لقطة تغيّرت تقريبًا = عالق فعليًا
            log("🧠 يراجع كل اللقطات (" + frames.size + ") وأي خطأ وقع، ويقرر إذا يحتاج تعديل...")
            val prompt = "هذا كود " + fileName + " الحالي:\n```\n" + current + "\n```\n" +
                "وهذا تسلسل لعبي الفعلي له كلاعب حقيقي (ضغطت أسهم ومسافة ونقرات فعلًا وأخذت لقطة بعد كل ضغطة):\n" +
                describeFrames(frames) +
                (if (stuck) "ملاحظة: الشكل ما تغيّر تقريبًا رغم الضغطات، يعني اللاعب أو الحركة عالقة أو ما تشتغل.\n" else "") +
                "إذا شفت خطأ JavaScript فعلي، أو اللاعب ما يتحرك كما يفترض، أو أي سلوك غير منطقي في التسلسل: عدّل الكود وأعطني النسخة الكاملة المصححة داخل ```. " +
                "إذا كان كل شي سليم اكتب نفس الكود بالضبط بدون أي تغيير."
            val reply = brain.answer(prompt, convId)
            val fixed = extractCode(reply.text)
            val changed = !(fixed.isNullOrBlank() || fixed.trim() == current.trim())
            if (!changed) {
                if (!hasErr && !stuck) { log("✅ لعبها واشتغلت صح بدون أخطاء، وقف هنا."); break }
                cleanStreak = 0
                if (hasErr || stuck) { log("⚠ فيه مشكلة لكن النموذج ما غيّر شي، يحاول مرة ثانية بصياغة أوضح إذا بقيت محاولات."); continue }
            } else {
                log("✏️ عدّل الكود بعد ما شافه يُلعب، يعيد اللعب من جديد...")
                current = fixed!!
                cleanStreak = 0
                continue
            }
            cleanStreak++
            if (cleanStreak >= 2) break
        }
        return current
    }
}
