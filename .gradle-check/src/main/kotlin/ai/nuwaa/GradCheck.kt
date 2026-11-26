package ai.nuwaa

internal fun runGradCheck() {
    val model = Nlm.fresh(e = 88, h = 24, chars = defaultVocabulary(), seed = 7L)
    val result = model.gradCheck(picks = 70, seed = 7L)
    println(result)
    check(result.startsWith("✅")) {
        "NLM2 gradient check FAILED. Do not start long training until backward pass is fixed."
    }
}
