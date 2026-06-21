#!/usr/bin/env dart
// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:archive/archive.dart';

// ANSI 颜色代码
const String _green = '\x1B[32m';
const String _cyan = '\x1B[36m';
const String _red = '\x1B[31m';
const String _yellow = '\x1B[33m';
const String _magenta = '\x1B[35m';
const String _reset = '\x1B[0m';

Process? _currentProcess; // 用于跟踪当前运行的子进程

/// 彩色打印
void _printColor(String text, String color) {
  print('$color$text$_reset');
}

/// 帮助函数：运行一个进程并流式传输其输出
///
/// [runInShell: true] 对于在 Windows 上正确执行 .bat 文件至关重要
/// 返回进程的退出代码
Future<int> _runCommand(
  String executable,
  List<String> arguments, {
  String? workingDirectory,
  Map<String, String>? environment,
}) async {
  final process = await Process.start(
    executable,
    arguments,
    workingDirectory: workingDirectory,
    environment: environment,
    runInShell: true, // 确保 .bat 文件能正确执行
  );
  _currentProcess = process; // 存储当前进程

  // 实时转发子进程的标准输出和错误
  process.stdout.transform(utf8.decoder).listen(stdout.write);
  process.stderr.transform(utf8.decoder).listen(stderr.write);

  final exitCode = await process.exitCode;
  _currentProcess = null; // 进程结束后清除
  return exitCode;
}

/// 跨平台强力杀进程函数
Future<void> _killProcessTree(Process process) async {
  final pid = process.pid;

  if (Platform.isWindows) {
    // Windows: 使用 taskkill 杀进程树
    try {
      await Process.run('taskkill', ['/F', '/T', '/PID', pid.toString()]);
    } catch (e) {
      _printColor('Windows 杀进程失败: $e', _red);
    }
  } else {
    // Unix (Linux/macOS):
    // runInShell: true 会启动一个 shell (sh/bash/zsh) 作为父进程。
    // 我们不仅要杀掉 shell，还要杀掉 shell 启动的子进程 (如 java/gradle)。
    try {
      // 1. 尝试使用 pkill -P <pid> 杀掉该 PID 的所有子进程
      await Process.run('pkill', ['-P', pid.toString()]);
    } catch (e) {
      // pkill 可能不存在或失败，忽略
      print('pkill 失败: $e');
    }

    // 2. 杀掉当前的 Shell 进程本身
    process.kill(ProcessSignal.sigkill);

    // 3. (可选保险措施) 如果知道 gradlew 会启动 java，
    // 有时候在极端情况下可能需要 killall java，但这样做太暴力，容易误伤，
    // 通常 pkill -P 配合 sigkill 已经足够。
  }
}

Future<Map<String, String>> _injectBindgenEnv() async {
  // 1. 获取完整的当前环境
  final Map<String, String> env = Map.from(Platform.environment);
  final ndkHome = env['ANDROID_NDK_HOME'];
  if (ndkHome == null || ndkHome.isEmpty) return env;

  String hostTag = Platform.isWindows
      ? 'windows-x86_64'
      : (Platform.isMacOS ? 'darwin-x86_64' : 'linux-x86_64');
  final String ndkBase = ndkHome.replaceAll('\\', '/');
  final String toolchainPath = '$ndkBase/toolchains/llvm/prebuilt/$hostTag';
  final String sysroot = '$toolchainPath/sysroot';
  final String ndkBin = '$toolchainPath/bin';

  // 2. 动态寻找 Clang (保持原样)
  String clangInclude = '';
  final clangLibDir = Directory('$toolchainPath/lib/clang');
  if (await clangLibDir.exists()) {
    final entities = await clangLibDir.list().toList();
    for (var entity in entities) {
      if (entity is Directory &&
          RegExp(
            r'^\d+',
          ).hasMatch(entity.path.split(Platform.pathSeparator).last)) {
        clangInclude = '${entity.path.replaceAll('\\', '/')}/include';
        break;
      }
    }
  }

  // 3. 注入 BINDGEN 参数 (保持正斜杠)
  env['BINDGEN_EXTRA_CLANG_ARGS'] = [
    if (clangInclude.isNotEmpty) '-isystem$clangInclude',
    '-isystem$sysroot/usr/include/arm-linux-androideabi',
    '-isystem$sysroot/usr/include',
    '--sysroot=$sysroot',
  ].join(' ');

  // 4. --- 重点修复：大小写无关地修改 Path ---
  // 遍历所有 key，找到那个叫 path 的（不管大小写）
  String? actualPathKey;
  for (var key in env.keys) {
    if (key.toLowerCase() == 'path') {
      actualPathKey = key;
      break;
    }
  }

  actualPathKey ??= Platform.isWindows ? 'Path' : 'PATH'; // 兜底
  final String oldPath = env[actualPathKey] ?? '';
  // 将 NDK bin 放在最前面，并保留原有的所有路径
  final String pathSeparator = Platform.isWindows ? ';' : ':';
  final String safeNdkBin = Platform.isWindows
      ? ndkBin.replaceAll('/', '\\')
      : ndkBin;
  env[actualPathKey] = '$safeNdkBin$pathSeparator$oldPath';

  // 5. 显式指定 LIBCLANG_PATH (不同操作系统 libclang 所在文件夹不同)
  String libClangPath = '';
  if (Platform.isWindows) {
    // Windows 下的 libclang.dll 大多位于 bin 目录
    libClangPath = ndkBin.replaceAll('/', '\\');
  } else if (Platform.isMacOS) {
    // macOS 下的 libclang.dylib 位于 lib 目录
    libClangPath = '$toolchainPath/lib';
  } else {
    // Linux 下的 libclang.so 位于 lib64 或 lib 目录
    final lib64Dir = Directory('$toolchainPath/lib64');
    if (await lib64Dir.exists()) {
      libClangPath = '$toolchainPath/lib64';
    } else {
      libClangPath = '$toolchainPath/lib';
    }
  }
  env['LIBCLANG_PATH'] = libClangPath;

  _printColor('✅ 环境注入完成 (已兼容所有操作系统的 Path 和 LIBCLANG_PATH):', _green);
  print('   NDK_BIN: $safeNdkBin');
  print('   LIBCLANG_PATH: $libClangPath');

  return env;
}

Future<Map<String, String>> _initializePaths() async {
  final sep = Platform.pathSeparator;
  final String scriptPath = Platform.script.toFilePath();
  final String scriptDir = Directory(scriptPath).parent.path;
  final String projectRoot = Directory(scriptDir).parent.path;

  const String flutterExecutable = 'flutter';

  final String symbolsDir = "$projectRoot${sep}build${sep}symbols";

  return {
    'sep': sep,
    'projectRoot': projectRoot,
    'flutterExecutable': flutterExecutable,
    'symbolsDir': symbolsDir,
  };
}

// ════════════════════════════════════════════════════════════════
//  ort / NNAPI 动态库（Android ort 后端，Real-CUGAN .onnx 推理）
// ════════════════════════════════════════════════════════════════

/// ort crate 2.0.0-rc.12 绑定 onnxruntime API 24 = onnxruntime 1.21.x（任意 patch 兼容）。
/// 与 build_windows.dart 的 _ortVersion 对齐。
const String _ortAndroidVersion = '1.21.0';

/// onnxruntime Android（Maven Central AAR，自带 NNAPI EP）。
/// 注意：Android 不发 GitHub release zip（404），官方经 Maven Central 分发 AAR
/// （https://onnxruntime.ai/docs/install/#install-on-android）。AAR 本质是 zip，内含 `jni/<abi>/libonnxruntime.so`。
final String _ortAndroidZipUrl =
    'https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$_ortAndroidVersion/onnxruntime-android-$_ortAndroidVersion.aar';

/// 确保 android/app/src/main/jniLibs/arm64-v8a/libonnxruntime.so 就绪（ort NNAPI 后端依赖）。
/// AAR 内 .so 形如 `jni/arm64-v8a/libonnxruntime.so`，按后缀匹配提取。AAR 缓存到 build/bin，
/// jniLibs 已在 .gitignore（不入库）。
Future<void> _ensureOrtAndroidLib(String projectRoot) async {
  final sep = Platform.pathSeparator;
  final jniLibsArm64 =
      '$projectRoot${sep}android${sep}app${sep}src${sep}main${sep}jniLibs${sep}arm64-v8a${sep}libonnxruntime.so';

  if (await File(jniLibsArm64).exists()) {
    _printColor('使用本地 libonnxruntime.so（arm64-v8a）', _green);
    return;
  }

  final binDir = '$projectRoot${sep}build${sep}bin';
  await Directory(binDir).create(recursive: true);
  final cachePath =
      '$binDir${sep}onnxruntime-android-$_ortAndroidVersion.aar';

  if (!await File(cachePath).exists()) {
    _printColor('下载 onnxruntime-android（NNAPI）: $_ortAndroidZipUrl', _cyan);
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(_ortAndroidZipUrl));
      final response = await request.close();
      if (response.statusCode != 200) {
        throw Exception(
          '下载 onnxruntime-android 失败，HTTP ${response.statusCode}',
        );
      }
      final sink = File(cachePath).openWrite();
      await response.pipe(sink);
      final mb =
          (await File(cachePath).length() / 1048576).toStringAsFixed(1);
      _printColor('下载完成: $mb MB', _green);
    } finally {
      client.close();
    }
  }

  _printColor('解压 libonnxruntime.so（arm64-v8a）...', _cyan);
  final zipBytes = await File(cachePath).readAsBytes();
  final archive = ZipDecoder().decodeBytes(zipBytes);
  ArchiveFile? soEntry;
  for (final f in archive) {
    if (f.name.endsWith('arm64-v8a/libonnxruntime.so')) {
      soEntry = f;
      break;
    }
  }
  if (soEntry == null) {
    throw Exception('onnxruntime-android AAR 中未找到 arm64-v8a/libonnxruntime.so');
  }
  await File(jniLibsArm64).parent.create(recursive: true);
  await File(jniLibsArm64).writeAsBytes(soEntry.content as List<int>);
  _printColor('已写入: $jniLibsArm64', _green);
}

Future<void> main(List<String> args) async {
  final Map<String, String> env = await _injectBindgenEnv();
  late final Map<String, String> paths;
  int exitCode = 0; // 初始化 exitCode

  bool isCleaningUp = false; // 防止重复清理

  bool isDebugMode = args.isNotEmpty;

  // 设置 SIGINT (Ctrl+C) 监听器
  late final StreamSubscription<ProcessSignal> sigintSubscription;
  sigintSubscription = ProcessSignal.sigint.watch().listen((_) async {
    if (isCleaningUp) return;
    isCleaningUp = true;

    _printColor('\n\n检测到 Ctrl+C。正在停止并清理...', _yellow);

    if (_currentProcess != null) {
      _printColor('正在强制终止子进程...', _yellow);
      await _killProcessTree(_currentProcess!);
      // 给文件系统一点喘息时间，释放 gradle 占用的锁
      await Future.delayed(Duration(milliseconds: 200));
    }

    _printColor('退出脚本。', _magenta);
    // 取消监听，防止多次触发
    await sigintSubscription.cancel();
    exit(130);
  });

  try {
    paths = await _initializePaths();

    final String projectRoot = paths['projectRoot']!;
    final String flutterExecutable = paths['flutterExecutable']!;
    final String symbolsDir = paths['symbolsDir']!;

    final String sentryDsn = env['SENTRY_DSN'] ?? '';
    if (sentryDsn.isEmpty) {
      _printColor('提示: 未找到 sentry_dsn 环境变量，将使用空字符串', _yellow);
    } else {
      _printColor('已读取 Sentry DSN (长度: ${sentryDsn.length})', _green);
    }

    _printColor('Flutter 命令: $flutterExecutable', _green);
    _printColor('工作目录: $projectRoot', _yellow);

    // 确保 ort NNAPI 动态库（arm64-v8a）入 jniLibs（ort 超分后端依赖）。
    await _ensureOrtAndroidLib(projectRoot);

    if (isDebugMode) {
      _printColor('\n⚡ 启动快速调试构建 (仅限 arm64 & x64)...', _magenta);

      exitCode = await _runCommand(
        flutterExecutable,
        [
          'build',
          'apk',
          '--debug',
          '--target-platform=android-arm64,android-x64',
          '--dart-define=sentry_dsn=$sentryDsn',
        ],
        workingDirectory: projectRoot,
        environment: env,
      );

      if (exitCode == 0) {
        _printColor('\n✅ Debug 构建成功 (arm64/x64)！', _green);
      } else {
        _printColor('\n❌ Debug 构建失败，请检查 Rust 代码错误。', _red);
      }
      await sigintSubscription.cancel();
      exit(exitCode);
    }

    _printColor('\n开始构建 APK（Impeller 由应用运行时判断）...', _cyan);
    exitCode = await _runCommand(
      flutterExecutable,
      [
        'build',
        'apk',
        '--split-per-abi',
        '--split-debug-info=$symbolsDir',
        '--dart-define=sentry_dsn=$sentryDsn',
      ],
      workingDirectory: projectRoot,
      environment: env,
    );
    if (exitCode != 0) {
      throw Exception('APK 构建失败！ (Exit code: $exitCode)');
    }
  } catch (e) {
    _printColor('\n构建过程中发生错误！: $e', _red);
    if (exitCode == 0) exitCode = 1;
  }

  await sigintSubscription.cancel();
  if (exitCode != 0) exit(exitCode);
  _printColor('\n构建流程全部完成！', _green);
}
