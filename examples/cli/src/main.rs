use std::clone::Clone;
use image::{GenericImageView, Rgb, RgbImage};
use ndarray::{s, ArrayView2};
use ndarray::{Array2};
use num_complex::{Complex, Complex64};
use palette::{FromColor, Lch, LinSrgb, Srgb};
use serde::{Deserialize, Serialize};
use summed_field_method::Field;
use summed_field_method::{resample_shape_min, sfm_asm_part_1, sfm_asm_part_2, sfm_asm_part_3};

#[derive(Clone, Serialize, Deserialize)]
struct Config {
    mask: String,
    size: f64,
    focal_length: f64,
    reduce_memory: bool,
    //auto_resampling: bool,
    oversample: f64,
    defocus: f64,
    lambda: f64,
    gamma: f64,
    //color_temp_k: f64,
}

pub fn main() {
    let file_path = "sfm.json".to_owned();
    let contents = std::fs::read_to_string(file_path).expect("Couldn't find or load sfm.json.");

    let c: Config = serde_json::from_str(&contents).unwrap();
    println!("{}", serde_json::to_string_pretty(&c).unwrap());

    let z = c.focal_length + c.defocus;
   
    /* load image */
    let (input_field, mask_shape) = load_image(c.clone());
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_real_image("test_cli_input.png", input_intensity.view(), 1.0, true).unwrap();

    let (resample_shape, _factor) = resample_shape_min(
        input_field.values.shape(),
        input_field.pitch,
        c.lambda,
        c.focal_length,
        c.oversample,
    );

    let gamma = (c.gamma, c.gamma);
    let tile_shape = [
        (mask_shape as f64 * c.oversample).ceil() as usize,
        (mask_shape as f64 * c.oversample).ceil() as usize,
    ];

    let mut input_spectrum = sfm_asm_part_1(
        input_field,
        c.focal_length,
        &[c.lambda, c.lambda * 1.1],
        tile_shape,
        resample_shape,
        c.reduce_memory,
    );
    let output_spectrum = sfm_asm_part_2(input_spectrum.swap_remove(0), z, c.lambda);
    save_complex_image("test_cli_spectrum.png", output_spectrum.values.view()).unwrap();

    let super_sample = 1;
    let mut super_sample_output =
        Array2::zeros([tile_shape[0] * super_sample, tile_shape[1] * super_sample]);

    for x in 0..super_sample {
        for y in 0..super_sample {
            let output = sfm_asm_part_3(
                &output_spectrum,
                gamma,
                (
                    y as f64 / super_sample as f64,
                    x as f64 / super_sample as f64,
                ),
            );
            super_sample_output
                .slice_mut(s![y..;super_sample, x..;super_sample])
                .assign(&output.values);
        }
    }

    let output_intensity = super_sample_output.map(|e| e.norm_sqr());
    let output_log_intensity = log_intensity(output_intensity.view(), 1e-10);

    save_real_image("test_cli_output.png", output_log_intensity.view(), 1.0, true).unwrap();
    save_complex_image("test_cli_outputc.png", super_sample_output.view()).unwrap();
}

pub fn save_grayscale_real_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<f64>,
    amp: f64,
    normalise: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let mut max: f64 = arr.iter().fold(0.0, |max, val| val.max(max));
        let sum = arr.iter().fold(0.0, |sum, val| val + sum);
        println!("h:{} w:{} max:{} sum:{} - {:?}", h, w, max, sum, file_name);

        let mut img = RgbImage::new(w as u32, h as u32);
        if !normalise {
            max = 1.0;
        }

        for (x, y, p) in img.enumerate_pixels_mut() {
            let value = arr[[y as usize, x as usize]] / max;
            let value = (value * amp).min(1.0).max(0.0);

            let para = (value - value * value) * 0.1;

            //let colour = Srgb::from(Hsl::new(360.0*(-value*0.65+0.65), 1.0, 0.01 + 0.99*value));
            let colour = Srgb::<f64>::from_linear(LinSrgb::new(
                value + para * ((value + 2.0 / 3.0) * std::f64::consts::PI * 2.0).sin(),
                value + para * ((value + 1.0 / 3.0) * std::f64::consts::PI * 2.0).sin(),
                value + para * (value * std::f64::consts::PI * 2.0).sin(),
            ));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}

pub fn save_real_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<f64>,
    amp: f64,
    normalise: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let mut max: f64 = arr.iter().fold(0.0, |max, val| val.max(max));
        let sum = arr.iter().fold(0.0, |sum, val| val + sum);
        println!("h:{} w:{} max:{} sum:{} - {:?}", h, w, max, sum, file_name);

        let mut img = RgbImage::new(w as u32, h as u32);
        if !normalise {
            max = 1.0;
        }

        for (x, y, p) in img.enumerate_pixels_mut() {
            let value = arr[[y as usize, x as usize]] / max;
            let value = (value * amp).min(1.0);

            //let colour = Srgb::from(Hsl::new(360.0*(-value*0.65+0.65), 1.0, 0.01 + 0.99*value));
            let colour =
                Srgb::from_color(Lch::new(value * 70.0, value * 128.0, 280.0 - 245.0 * value));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}

pub fn log_intensity(arr: ArrayView2<f64>, min: f64) -> Array2<f64> {
    let log_min = -min.ln();
    let max = arr.iter().fold(0.0, |max, e| e.max(max));
    arr.map(|e| ((e / max).ln() / log_min + 1.0).max(0.0).min(1.0))
}

pub fn save_complex_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<Complex<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let max_sqr: f64 = arr.iter().fold(0.0, |max, val| val.norm_sqr().max(max));
        let sum_sqr: f64 = arr.iter().fold(0.0, |sum, val| val.norm_sqr() + sum);
        println!(
            "h:{} w:{} max_sqr:{} sum_sqr:{} - {:?}",
            h, w, max_sqr, sum_sqr, file_name
        );

        let max = max_sqr.sqrt();

        let mut img = RgbImage::new(w as u32, h as u32);

        for (x, y, p) in img.enumerate_pixels_mut() {
            let (r, theta) = arr[[y as usize, x as usize]].to_polar();
            let r = r / max;

            //let colour = Srgb::from(Hsv::new(360.0*(theta/std::fxx::consts::TAU + 0.5), 1.0, r*0.9));
            let colour = Srgb::from_color(Lch::new(
                r * 100.0,
                r * 128.0,
                360.0 * (theta / ::std::f64::consts::PI + 1.0) * 0.5,
            ));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}

fn load_image(c : Config) -> (Field, usize) {
    let img = image::open(c.mask).expect("File not found!");

    let width: usize = img.width().try_into().unwrap();
    let height: usize = img.height().try_into().unwrap();

    let mut mask: Array2<Complex64> = Array2::zeros([width, height]);

    for (x, y, pixel) in img.pixels() {
        let x: usize = x.try_into().unwrap();
        let y: usize = y.try_into().unwrap();

        // PNG is big endian, right?
        let re: f64 = u32::from_be_bytes(pixel.0).into();

        mask[[x, y]] = Complex64::new(re, 0.0);
    
        }

    let p = c.size / width as f64;

    return (
        Field {
            values: mask,
            pitch: (p, p),
        },
        width,
    )
}

