//! ONNX 超分（Real-CUGAN tile 模型，ort crate load-dynamic + EP）。
//!
//! iOS 不用此模块（走 onnxruntime-objc + CoreML EP，已验证）。
//! 桌面 / Android / macOS 用：load-dynamic 外挂含 EP 的 onnxruntime 动态库。

use anyhow::{Context, Result, anyhow};
use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

/// tile 几何常量（与 iOS AppDelegate.swift 一致）。
const CROP_SIZE: usize = 128;
const PREPADDING: usize = 18;
const TILE_SIZE: usize = CROP_SIZE + PREPADDING * 2; // 164
const OUTPUT_SIZE: usize = CROP_SIZE * 2; // 256

/// ort 超分：读 input_path 图片 → Real-CUGAN tile 推理 → 写 output_path（2x PNG）。
///
/// [dylib_path] 外挂 onnxruntime 动态库路径（load-dynamic）。
#[frb]
pub fn upscale(
    input_path: String,
    output_path: String,
    model_path: String,
    dylib_path: String,
) -> Result<()> {
    // load-dynamic init（外挂含 EP 的 onnxruntime 动态库）。
    ort::init_from(&dylib_path)
        .with_context(|| format!("ort init 失败（dylib={}）", dylib_path))?
        .commit();

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap_or_else(|e| e.recover())
        .commit_from_file(&model_path)
        .with_context(|| format!("ort session 创建失败（model={}）", model_path))?;

    // 读图 → HWC f32 0-1
    let img = image::open(&input_path).with_context(|| format!("读图失败: {}", input_path))?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let input_floats: Vec<f32> = rgb.iter().map(|&v| v as f32 / 255.0).collect();

    // 输出 buffer（2x，HWC）
    let (out_w, out_h) = (width * 2, height * 2);
    let mut output_floats = vec![0f32; out_w * out_h * 3];

    // tile 循环（步进 CROP_SIZE，无 overlap 直贴）
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let tile_w = CROP_SIZE.min(width - x);
            let tile_h = CROP_SIZE.min(height - y);
            let tile_input = extract_tile_with_reflect_pad(&input_floats, width, height, x, y);
            let tile_output = run_inference(&mut session, tile_input)?;
            copy_tile_output(
                &tile_output,
                &mut output_floats,
                out_w,
                x * 2,
                y * 2,
                tile_w * 2,
                tile_h * 2,
            );
            x += CROP_SIZE;
        }
        y += CROP_SIZE;
    }

    // 写 PNG（HWC f32 → RGB u8）
    let mut out_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(out_w as u32, out_h as u32);
    for py in 0..out_h {
        for px in 0..out_w {
            let idx = (py * out_w + px) * 3;
            let pixel = out_img.get_pixel_mut(px as u32, py as u32);
            pixel[0] = (output_floats[idx] * 255.0).clamp(0.0, 255.0) as u8;
            pixel[1] = (output_floats[idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
            pixel[2] = (output_floats[idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out_img
        .save(&output_path)
        .with_context(|| format!("写 PNG 失败: {}", output_path))?;
    Ok(())
}

/// ort 推理：NCHW [1,3,164,164] → NCHW [1,3,256,256]。
fn run_inference(session: &mut Session, input: Vec<f32>) -> Result<Vec<f32>> {
    let shape = vec![1_i64, 3, TILE_SIZE as i64, TILE_SIZE as i64];
    let tensor = Tensor::from_array((shape, input))
        .map_err(|e| anyhow!("ort input tensor 失败: {}", e))?;
    let outputs = session
        .run(ort::inputs!["input" => tensor])
        .map_err(|e| anyhow!("ort run 失败: {}", e))?;
    let out = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("ort output 提取失败: {}", e))?;
    Ok(out.1.to_vec())
}

/// Reflect 坐标映射（对齐 iOS reflectCoordinate：i<0 → -i-1, i>=size → 2*size-i-1）。
fn reflect_coord(i: isize, size: usize) -> usize {
    let s = size as isize;
    let r = if i < 0 {
        -i - 1
    } else if i >= s {
        2 * s - i - 1
    } else {
        i
    };
    r.clamp(0, s - 1) as usize
}

/// 取 tile + reflect pad → NCHW [3, TILE_SIZE, TILE_SIZE]。
/// 中心 128×128 + 四边 reflect pad（18px），HWC 输入 → NCHW 输出。
fn extract_tile_with_reflect_pad(
    floats: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> Vec<f32> {
    let stride = TILE_SIZE * TILE_SIZE;
    let mut result = vec![0f32; 3 * stride];
    let xi = x as isize;
    let yi = y as isize;

    // 中心 CROP_SIZE×CROP_SIZE 区域
    for dy in 0..CROP_SIZE {
        for dx in 0..CROP_SIZE {
            let src_x = reflect_coord(xi + dx as isize, width);
            let src_y = reflect_coord(yi + dy as isize, height);
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = (PREPADDING + dy) * TILE_SIZE + (PREPADDING + dx);
            result[dst_idx] = floats[src_idx];
            result[stride + dst_idx] = floats[src_idx + 1];
            result[2 * stride + dst_idx] = floats[src_idx + 2];
        }
    }

    // 左边框 (PREPADDING × CROP_SIZE)
    for dy in 0..CROP_SIZE {
        for dx in 0..PREPADDING {
            let src_x = reflect_coord(xi + PREPADDING as isize - dx as isize, width);
            let src_y = reflect_coord(yi + dy as isize, height);
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = (PREPADDING + dy) * TILE_SIZE + dx;
            result[dst_idx] = floats[src_idx];
            result[stride + dst_idx] = floats[src_idx + 1];
            result[2 * stride + dst_idx] = floats[src_idx + 2];
        }
    }

    // 右边框 (PREPADDING × CROP_SIZE)
    for dy in 0..CROP_SIZE {
        for dx in 0..PREPADDING {
            let src_x = reflect_coord(xi + CROP_SIZE as isize - 1 - dx as isize, width);
            let src_y = reflect_coord(yi + dy as isize, height);
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = (PREPADDING + dy) * TILE_SIZE + (PREPADDING + CROP_SIZE + dx);
            result[dst_idx] = floats[src_idx];
            result[stride + dst_idx] = floats[src_idx + 1];
            result[2 * stride + dst_idx] = floats[src_idx + 2];
        }
    }

    // 上边框 (TILE_SIZE × PREPADDING)
    for dy in 0..PREPADDING {
        for dx in 0..TILE_SIZE {
            let src_x = reflect_coord(xi + dx as isize - PREPADDING as isize, width);
            let src_y = reflect_coord(yi + PREPADDING as isize - dy as isize, height);
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = dy * TILE_SIZE + dx;
            result[dst_idx] = floats[src_idx];
            result[stride + dst_idx] = floats[src_idx + 1];
            result[2 * stride + dst_idx] = floats[src_idx + 2];
        }
    }

    // 下边框 (TILE_SIZE × PREPADDING)
    for dy in 0..PREPADDING {
        for dx in 0..TILE_SIZE {
            let src_x = reflect_coord(xi + dx as isize - PREPADDING as isize, width);
            let src_y = reflect_coord(yi + CROP_SIZE as isize - 1 - dy as isize, height);
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = (PREPADDING + CROP_SIZE + dy) * TILE_SIZE + dx;
            result[dst_idx] = floats[src_idx];
            result[stride + dst_idx] = floats[src_idx + 1];
            result[2 * stride + dst_idx] = floats[src_idx + 2];
        }
    }

    result
}

/// 复制 tile 输出（NCHW [3, OUTPUT_SIZE, OUTPUT_SIZE]）到目标大图（HWC），直贴。
fn copy_tile_output(
    source: &[f32],
    dest: &mut [f32],
    dest_w: usize,
    dest_x: usize,
    dest_y: usize,
    valid_w: usize,
    valid_h: usize,
) {
    let channel_stride = OUTPUT_SIZE * OUTPUT_SIZE;
    for y in 0..valid_h {
        for x in 0..valid_w {
            let dst_idx = ((dest_y + y) * dest_w + (dest_x + x)) * 3;
            for c in 0..3 {
                dest[dst_idx + c] = source[c * channel_stride + y * OUTPUT_SIZE + x];
            }
        }
    }
}
