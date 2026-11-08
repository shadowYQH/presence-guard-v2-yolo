plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.presenceguard.yolo"
    compileSdk = 34
    buildToolsVersion = "34.0.0"

    defaultConfig {
        applicationId = "com.presenceguard.yolo"
        minSdk = 26
        targetSdk = 34
        versionCode = 2
        versionName = "2.0.0"

        ndk {
            // 只保留 arm64：x86_64 只对模拟器有意义，armeabi-v7a 是为 32 位老机型准备的。
            // GPU 委托又带了两个 ABI 的原生库，去掉 32 位可省约 35MB。
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    androidResources {
        // TFLite 需要通过 mmap 直接读取模型，必须保持未压缩
        noCompress += listOf("tflite", "lite")
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/AL2.0",
                "META-INF/LGPL2.1",
                "META-INF/*.kotlin_module"
            )
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // 唯一的推理引擎：YOLO（TFLite）。这里只要最朴素的运行时，
    // 不引入 task-vision / ML Kit —— 解码逻辑全部自己写，见 detect/YoloDetector.kt。
    // 好处是 APK 从 56MB 降到 8MB，且不受 Task 库那两个老问题（量化模型加载失败、
    // 原生库进程级卸载）影响。
    // 唯一的推理引擎：YOLO（TFLite）。
    //
    // 不引 tensorflow-lite-gpu：实测这个混合量化模型在 2.14 的 GPU 委托上一个内核都
    // 上不了（int8 张量不被支持），只白白多 13MB 原生库。详见 docs/PERFORMANCE.md 第 2 节。
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}
