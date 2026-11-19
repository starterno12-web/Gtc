import java.util.Properties

plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

// Reads the Tavily key from a LOCAL, GITIGNORED file — never from source code, never committed.
// Create app/secrets.properties (NOT secrets.properties.example) with one line:
//   TAVILY_API_KEY=your_new_rotated_key_here
// If that file doesn't exist (e.g. a fresh clone, or GitHub Actions without the secret), the key is
// just empty and WebSearch.search() returns null — the app still runs fine, search is simply off.
val secrets = Properties().apply {
    val f = rootProject.file("app/secrets.properties")
    if (f.exists()) load(f.inputStream())
}

android {
    namespace = "ai.nuwaa"
    compileSdk = 34
    defaultConfig {
        applicationId = "ai.nuwaa"; minSdk = 24; targetSdk = 34; versionCode = 1; versionName = "0.1"
        buildConfigField("String", "TAVILY_API_KEY", "\"${secrets.getProperty("TAVILY_API_KEY", "")}\"")
    }
    buildFeatures { buildConfig = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "bin" }
}

dependencies {
    // Only new dependency in the whole app. Needed because plain Android has no built-in WebSocket
    // client — this is what TvLink.kt uses to talk to the pairing server (same protocol as server.ts).
    // Free, no account, no API key; one of the most widely used Android libraries there is.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
