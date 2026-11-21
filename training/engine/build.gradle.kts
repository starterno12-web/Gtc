plugins {
    kotlin("jvm") version "2.0.21"
    application
}
repositories { mavenCentral() }
dependencies { implementation("org.json:json:20240303") }
kotlin { jvmToolchain(17) }
application { mainClass.set("ai.nuwaa.training.TrainingRunnerKt") }
