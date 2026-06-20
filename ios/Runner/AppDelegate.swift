import Flutter
import UIKit
import Darwin
import onnxruntime_objc
import CoreGraphics
import ImageIO

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let memoryChannelName = "memory_monitor"
  private var memoryChannel: FlutterMethodChannel?

  /// RealSR 超分通道名，与 Dart / Android 侧一致。
  private let realSrChannelName = "realsr_super_resolution"

  /// ONNX Runtime 环境（进程级单例，懒初始化）。
  private static var ortEnv: ORTEnv?

  /// ONNX Runtime Session（进程级单例，懒初始化，CoreML EP）。
  private static var ortSession: ORTSession?

  /// Real-CUGAN tile 参数（固定）。
  private static let cropSize: Int = 128        // 有效输入大小
  private static let prepadding: Int = 18       // 每边上下文（reflect）
  private static let tileSize: Int = 164        // cropSize + prepadding*2
  private static let outputSize: Int = 256       // 128 * 2

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    setupMemoryChannel(pluginRegistry: engineBridge.pluginRegistry)
    setupRealSrChannel(pluginRegistry: engineBridge.pluginRegistry)
  }

  /// 注册 RealSR 超分通道。
  ///
  /// Phase 3：完整 tile 推理实现（ORTSession + Real-CUGAN 164×164→256×256 tile）。
  private func setupRealSrChannel(pluginRegistry: FlutterPluginRegistry) {
    guard let registrar = pluginRegistry.registrar(forPlugin: realSrChannelName) else {
      return
    }

    let channel = FlutterMethodChannel(
      name: realSrChannelName,
      binaryMessenger: registrar.messenger()
    )

    channel.setMethodCallHandler { call, result in
      switch call.method {
      case "extractAssets":
        // 骨架阶段：iOS 无需从 assets 解压模型，直接标记成功。
        result(true)
      case "upscale":
        // Phase 3：真实 tile 推理（ORTSession + Real-CUGAN tile）。
        guard let args = call.arguments as? [String: Any],
              let inputPath = args["inputPath"] as? String,
              let outputPath = args["outputPath"] as? String else {
          result(FlutterError(
            code: "INVALID_ARGS",
            message: "缺少 inputPath 或 outputPath",
            details: nil
          ))
          return
        }

        DispatchQueue.global(qos: .userInitiated).async {
          self.doUpscale(inputPath: inputPath, outputPath: outputPath) { ret in
            DispatchQueue.main.async {
              result(ret)
            }
          }
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  private func setupMemoryChannel(pluginRegistry: FlutterPluginRegistry) {
    if memoryChannel != nil {
      return
    }

    guard let registrar = pluginRegistry.registrar(forPlugin: memoryChannelName) else {
      return
    }

    let channel = FlutterMethodChannel(
      name: memoryChannelName,
      binaryMessenger: registrar.messenger()
    )

    let handler: FlutterMethodCallHandler = { [weak self] call, result in
      guard let self = self else {
        result(FlutterError(code: "DEALLOCATED", message: "AppDelegate deallocated", details: nil))
        return
      }

      switch call.method {
      case "getMemoryInfo":
        result(self.getMemoryInfo())
      case "getDartMemoryInfo":
        result(self.getDartMemoryInfo())
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    channel.setMethodCallHandler(handler)

    memoryChannel = channel
  }

  private func getMemoryInfo() -> [String: Any] {
    let process = getProcessMemoryUsage()
    let totalMemory = ProcessInfo.processInfo.physicalMemory
    let availableMemory = getAvailableMemory()

    return [
      "totalMemory": toInt64(totalMemory),
      "availableMemory": toInt64(availableMemory),
      "nativeHeapSize": toInt64(process.virtualSize),
      "nativeHeapAllocatedSize": toInt64(process.residentSize),
      "nativeHeapFreeSize": 0,
    ]
  }

  private func getDartMemoryInfo() -> [String: Any] {
    let process = getProcessMemoryUsage()
    let totalMemory = ProcessInfo.processInfo.physicalMemory
    let usedMemory = process.residentSize
    let freeMemory = totalMemory > usedMemory ? totalMemory - usedMemory : 0

    return [
      "dartHeapUsed": toInt64(usedMemory),
      "dartHeapCapacity": toInt64(totalMemory),
      "dartHeapCommitted": toInt64(usedMemory),
      "externalMemory": toInt64(usedMemory),
      "maxMemory": toInt64(totalMemory),
      "totalMemory": toInt64(totalMemory),
      "freeMemory": toInt64(freeMemory),
      "usedMemory": toInt64(usedMemory),
      "nativeHeapSize": toInt64(process.virtualSize),
      "nativeHeapAllocated": toInt64(process.residentSize),
      "nativeHeapFree": 0,
      "processRss": toInt64(process.residentSize),
    ]
  }

  private func getAvailableMemory() -> UInt64 {
    var pageSize: vm_size_t = 0
    host_page_size(mach_host_self(), &pageSize)

    var vmStat = vm_statistics64()
    var count = mach_msg_type_number_t(
      MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
    )

    let result: kern_return_t = withUnsafeMutablePointer(to: &vmStat) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
        host_statistics64(mach_host_self(), HOST_VM_INFO64, reboundPointer, &count)
      }
    }

    if result != KERN_SUCCESS {
      return 0
    }

    let freePages = UInt64(vmStat.free_count)
    let inactivePages = UInt64(vmStat.inactive_count)
    let speculativePages = UInt64(vmStat.speculative_count)
    return (freePages + inactivePages + speculativePages) * UInt64(pageSize)
  }

  private func getProcessMemoryUsage() -> (residentSize: UInt64, virtualSize: UInt64) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
      MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size
    )

    let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPointer, &count)
      }
    }

    if result != KERN_SUCCESS {
      return (0, 0)
    }

    return (
      UInt64(info.resident_size),
      UInt64(info.virtual_size)
    )
  }

  private func toInt64(_ value: UInt64) -> Int64 {
    if value > UInt64(Int64.max) {
      return Int64.max
    }
    return Int64(value)
  }

  // MARK: - Real-CUGAN 超分推理

  /// 执行超分推理（异步）。
  private func doUpscale(inputPath: String, outputPath: String, completion: @escaping ([String: Any]?) -> Void) {
    do {
      // 1. 初始化 ORT 环境（懒加载）
      if AppDelegate.ortEnv == nil {
        AppDelegate.ortEnv = try ORTEnv(loggingLevel: .warning)
      }

      // 2. 初始化 ORTSession（懒加载，带 CoreML EP）
      if AppDelegate.ortSession == nil {
        guard let modelPath = findModelPath() else {
          completion(failureResult("模型文件不存在"))
          return
        }
        let opts = try ORTSessionOptions()
        try opts.appendExecutionProvider("coreml", providerOptions: [:])
        try opts.setGraphOptimizationLevel(.all)
        AppDelegate.ortSession = try ORTSession(env: AppDelegate.ortEnv!, modelPath: modelPath, sessionOptions: opts)
      }

      // 3. 读取输入图像
      guard let inputCGImage = loadImage(from: inputPath) else {
        completion(failureResult("无法读取输入图像"))
        return
      }

      let width = Int(inputCGImage.width)
      let height = Int(inputCGImage.height)
      let outputWidth = width * 2
      let outputHeight = height * 2

      // 4. 预处理：CGImage → Float RGB 0-1
      guard let inputFloats = cgImageToFloats(cgImage: inputCGImage) else {
        completion(failureResult("图像预处理失败"))
        return
      }

      // 5. 分块推理
      var outputFloats = [Float](repeating: 0, count: outputWidth * outputHeight * 3)

      for y in stride(from: 0, to: height, by: AppDelegate.cropSize) {
        for x in stride(from: 0, to: width, by: AppDelegate.cropSize) {
          let tileW = min(AppDelegate.cropSize, width - x)
          let tileH = min(AppDelegate.cropSize, height - y)

          guard let tileInput = extractTileWithReflectPad(
            floats: inputFloats, width: width, height: height, x: x, y: y, tileW: tileW, tileH: tileH
          ) else {
            completion(failureResult("tile 提取失败"))
            return
          }

          guard let tileOutput = runInference(input: tileInput) else {
            completion(failureResult("推理失败"))
            return
          }

          copyTileOutput(
            source: tileOutput, sourceW: AppDelegate.outputSize, sourceH: AppDelegate.outputSize,
            dest: &outputFloats, destW: outputWidth, destH: outputHeight,
            destX: x * 2, destY: y * 2, validW: tileW * 2, validH: tileH * 2
          )
        }
      }

      // 6. 后处理：Float → uint8 RGB → CGImage → PNG
      guard let outputCGImage = floatsToCGImage(floats: outputFloats, width: outputWidth, height: outputHeight),
            saveCGImage(cgImage: outputCGImage, to: outputPath) else {
        completion(failureResult("图像保存失败"))
        return
      }

      completion(successResult(outputPath))
    } catch {
      completion(failureResult("超分失败: \(error)"))
    }
  }

  /// 构建成功结果。
  private func successResult(_ outputPath: String) -> [String: Any] {
    ["success": true, "exitCode": 0, "outputPath": outputPath, "stdout": "", "stderr": ""]
  }

  /// 构建失败结果。
  private func failureResult(_ message: String) -> [String: Any] {
    ["success": false, "exitCode": 1, "stderr": message]
  }

  /// 查找模型文件路径。
  private func findModelPath() -> String? {
    let modelName = "super_resolution/realcugan-onnx/up2x-conservative-2x-tile.onnx"
    // Dart getFilePath() = ApplicationSupport/files；模型在 <AppSupport>/files/super_resolution/...
    let searchDirs: [FileManager.SearchPathDirectory] = [
      .applicationSupportDirectory, .cachesDirectory, .documentDirectory,
    ]

    for dir in searchDirs {
      guard let base = NSSearchPathForDirectoriesInDomains(dir, .userDomainMask, true).first else {
        continue
      }
      // Dart getFilePath = AppSupport/files，模型路径 = base/files/super_resolution/...
      let withFiles = (base as NSString).appendingPathComponent("files").appendingPathComponent(modelName)
      if FileManager.default.fileExists(atPath: withFiles) {
        return withFiles
      }
      // 兼容：直接 base/super_resolution/...
      let direct = (base as NSString).appendingPathComponent(modelName)
      if FileManager.default.fileExists(atPath: direct) {
        return direct
      }
    }
    return nil
  }

  /// CGImage → Float RGB HWC 0-1。
  private func cgImageToFloats(cgImage: CGImage) -> [Float]? {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let totalBytes = height * bytesPerRow

    guard let data = CFDataCreateMutable(nil, totalBytes),
          let context = CGContext(
            data: CFDataGetMutableBytePtr(data), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
          ) else {
      return nil
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    let ptr = CFDataGetMutableBytePtr(data)!
    var floats = [Float](repeating: 0, count: width * height * 3)
    let inv255: Float = 1.0 / 255.0

    for i in 0..<(width * height) {
      let srcIdx = i * 4
      let dstIdx = i * 3
      floats[dstIdx + 0] = Float(ptr[srcIdx + 0]) * inv255
      floats[dstIdx + 1] = Float(ptr[srcIdx + 1]) * inv255
      floats[dstIdx + 2] = Float(ptr[srcIdx + 2]) * inv255
    }

    return floats
  }

  /// 从图像提取 tile 并 reflect pad 到 164×164（NCHW 格式返回）。
  ///
  /// 算法：取以 为中心的 128×128 有效区 + 18 像素 reflect pad 上下文。
  /// 边界 tile 不足 128 时先 reflect 扩到 128，再加 18 上下文。
  private func extractTileWithReflectPad(
    floats: [Float],
    width: Int,
    height: Int,
    x: Int,
    y: Int,
    tileW: Int,
    tileH: Int
  ) -> [Float]? {
    let nchwStride = AppDelegate.tileSize * AppDelegate.tileSize
    var result = [Float](repeating: 0, count: 3 * nchwStride)

    // 中心 128×128 区域
    fillCenterRegion(floats, width, height, x, y, &result)

    // Reflect pad 四个边框
    fillPaddingBorders(floats, width, height, x, y, &result)

    return result
  }

  /// 填充中心 128×128 区域。
  private func fillCenterRegion(
    _ floats: [Float], _ width: Int, _ height: Int,
    _ x: Int, _ y: Int, _ result: inout [Float]
  ) {
    let nchwStride = AppDelegate.tileSize * AppDelegate.tileSize

    for dy in 0..<AppDelegate.cropSize {
      for dx in 0..<AppDelegate.cropSize {
        let srcX = reflectCoordinate(x + dx, size: width)
        let srcY = reflectCoordinate(y + dy, size: height)
        let srcIdx = (srcY * width + srcX) * 3

        let outX = AppDelegate.prepadding + dx
        let outY = AppDelegate.prepadding + dy
        let dstIdx = outY * AppDelegate.tileSize + outX

        result[dstIdx] = floats[srcIdx]
        result[nchwStride + dstIdx] = floats[srcIdx + 1]
        result[2 * nchwStride + dstIdx] = floats[srcIdx + 2]
      }
    }
  }

  /// 填充 reflect pad 边框。
  private func fillPaddingBorders(
    _ floats: [Float], _ width: Int, _ height: Int,
    _ x: Int, _ y: Int, _ result: inout [Float]
  ) {
    let nchwStride = AppDelegate.tileSize * AppDelegate.tileSize
    let pp = AppDelegate.prepadding
    let cs = AppDelegate.cropSize

    // 左边框 (18 × 128)
    for dy in 0..<cs {
      for dx in 0..<pp {
        let srcX = reflectCoordinate(x + pp - dx, size: width)
        let srcY = reflectCoordinate(y + dy, size: height)
        let dstIdx = (pp + dy) * AppDelegate.tileSize + dx
        copyPixel(floats, (srcY * width + srcX) * 3, &result, dstIdx, nchwStride)
      }
    }

    // 右边框 (18 × 128)
    for dy in 0..<cs {
      for dx in 0..<pp {
        let srcX = reflectCoordinate(x + cs - 1 - dx, size: width)
        let srcY = reflectCoordinate(y + dy, size: height)
        let dstIdx = (pp + dy) * AppDelegate.tileSize + (pp + cs + dx)
        copyPixel(floats, (srcY * width + srcX) * 3, &result, dstIdx, nchwStride)
      }
    }

    // 上边框 (164 × 18)
    for dy in 0..<pp {
      for dx in 0..<AppDelegate.tileSize {
        let srcX = reflectCoordinate(x + dx - pp, size: width)
        let srcY = reflectCoordinate(y + pp - dy, size: height)
        let dstIdx = dy * AppDelegate.tileSize + dx
        copyPixel(floats, (srcY * width + srcX) * 3, &result, dstIdx, nchwStride)
      }
    }

    // 下边框 (164 × 18)
    for dy in 0..<pp {
      for dx in 0..<AppDelegate.tileSize {
        let srcX = reflectCoordinate(x + dx - pp, size: width)
        let srcY = reflectCoordinate(y + cs - 1 - dy, size: height)
        let dstIdx = (pp + cs + dy) * AppDelegate.tileSize + dx
        copyPixel(floats, (srcY * width + srcX) * 3, &result, dstIdx, nchwStride)
      }
    }
  }

  /// 复制单个像素（HWC → NCHW）。
  private func copyPixel(_ src: [Float], _ srcIdx: Int, _ dst: inout [Float], _ dstIdx: Int, _ nchwStride: Int) {
    dst[dstIdx] = src[srcIdx]
    dst[nchwStride + dstIdx] = src[srcIdx + 1]
    dst[2 * nchwStride + dstIdx] = src[srcIdx + 2]
  }

  /// Reflect 坐标映射：i<0 → -i-1, i>=W → 2W-i-1
  private func reflectCoordinate(_ i: Int, size: Int) -> Int {
    if i < 0 { return -i - 1 }
    if i >= size { return 2 * size - i - 1 }
    return i
  }

  /// 运行 ONNX 推理（输入 NCHW [1,3,164,164]，输出 NCHW [1,3,256,256]）。
  private func runInference(input: [Float]) -> [Float]? {
    guard let session = AppDelegate.ortSession else { return nil }

    // 输入张量
    let inputCount = 3 * AppDelegate.tileSize * AppDelegate.tileSize
    let inputData = NSMutableData(length: inputCount * MemoryLayout<Float>.size)!
    inputData.mutableBytes.assumingMemoryBound(to: Float.self).assign(from: input, count: inputCount)

    // ORTValue + 推理
    guard let tensor = try? ORTValue(
      tensorData: inputData, elementType: .float,
      shape: [1, 3, AppDelegate.tileSize, AppDelegate.tileSize] as [NSNumber]
    ) else { return nil }

    guard let outputs = try? session.run(
      withInputs: ["input": tensor], outputNames: ["output"], runOptions: try ORTRunOptions()
    ) else { return nil }

    guard let out = outputs["output"],
          let outData = try? out.tensorData() else { return nil }

    let outPtr = outData.bytes.assumingMemoryBound(to: Float.self)
    let outCount = 3 * AppDelegate.outputSize * AppDelegate.outputSize
    return Array(UnsafeBufferPointer(start: outPtr, count: outCount))
  }

  /// 复制 tile 输出到目标图（直贴）。
  private func copyTileOutput(
    source: [Float], sourceW: Int, sourceH: Int,
    dest: inout [Float], destW: Int, destH: Int,
    destX: Int, destY: Int, validW: Int, validH: Int
  ) {
    for y in 0..<validH {
      let srcBaseY = y * sourceW * 3
      let dstBaseY = (destY + y) * destW * 3
      for x in 0..<validW {
        let srcIdx = srcBaseY + x * 3
        let dstIdx = dstBaseY + (destX + x) * 3
        dest[dstIdx] = source[srcIdx]
        dest[dstIdx + 1] = source[srcIdx + 1]
        dest[dstIdx + 2] = source[srcIdx + 2]
      }
    }
  }

  /// Float NCHW → CGImage（RGB 0-1 → clip 0-1 ×255 → uint8）。
  private func floatsToCGImage(floats: [Float], width: Int, height: Int) -> CGImage? {
    let bytesPerRow = width * 3
    guard let data = CFDataCreateMutable(nil, height * bytesPerRow) else { return nil }

    let ptr = CFDataGetMutableBytePtr(data)!
    for i in 0..<(floats.count) {
      ptr[i] = UInt8(max(0.0, min(1.0, floats[i])) * 255.0)
    }

    guard let provider = CGDataProvider(data: data),
          let cgImage = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
          ) else { return nil }

    return cgImage
  }

  /// 保存 CGImage 为 PNG 文件。
  private func saveCGImage(cgImage: CGImage, to path: String) -> Bool {
    guard let dest = CGImageDestinationCreateWithURL(
      URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil
    ) else { return false }

    CGImageDestinationAddImage(dest, cgImage, nil)
    return CGImageDestinationFinalize(dest)
  }

  /// 从文件加载 CGImage。
  private func loadImage(from path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
  }
}
