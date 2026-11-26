package ai.nuwaa

internal fun runGradCheck() {
    val defaultChars = (" \n\t.,!?:;'\\\"()[]{}-_=+*/\\\\<>@#$%^&|~" + '\u0060' +
        "0123456789" +
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" +
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰ")
        .toCharArray().distinct().map { it.toString() }

    val model = Nlm.fresh(e = 88, h = 24, chars = defaultChars, seed = 7L)
    val result = model.gradCheck(picks = 70, seed = 7L)
    println(result)
    check(result.startsWith("✅")) {
        "NLM2 gradient check FAILED. Do not start long training until backward pass is fixed."
    }
}
