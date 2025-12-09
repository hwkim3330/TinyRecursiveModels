//! Model exports and utilities

// Re-export everything from lib.rs
// This file is for additional model utilities

use crate::tensor::Tensor;
use wasm_bindgen::prelude::*;

/// Utility to create a Sudoku input tensor
#[wasm_bindgen]
pub fn create_sudoku_input(grid: &[u8]) -> Vec<u8> {
    // Convert 9x9 grid to flat array
    // 0 = empty, 1-9 = digits
    let mut input = vec![0u8; 81];
    for (i, &v) in grid.iter().enumerate().take(81) {
        input[i] = v;
    }
    input
}

/// Utility to create a Maze input tensor
#[wasm_bindgen]
pub fn create_maze_input(grid: &[u8], width: usize, height: usize) -> Vec<u8> {
    // 0 = path, 1 = wall, 2 = start, 3 = end
    let mut input = vec![0u8; width * height];
    for (i, &v) in grid.iter().enumerate().take(width * height) {
        input[i] = v;
    }
    input
}

/// Visualization helper: normalize activations to 0-255 range
#[wasm_bindgen]
pub fn normalize_activations(activations: &[f32]) -> Vec<u8> {
    if activations.is_empty() {
        return vec![];
    }

    let min_val = activations.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = activations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    if range < 1e-6 {
        return vec![128u8; activations.len()];
    }

    activations.iter()
        .map(|&v| ((v - min_val) / range * 255.0) as u8)
        .collect()
}

/// Compute attention-style heatmap from two activation vectors
#[wasm_bindgen]
pub fn compute_attention_heatmap(query: &[f32], key: &[f32], size: usize) -> Vec<f32> {
    let mut heatmap = vec![0.0f32; size * size];

    // Simplified attention: dot product of query and key
    for i in 0..size {
        for j in 0..size {
            if i < query.len() && j < key.len() {
                heatmap[i * size + j] = query[i] * key[j];
            }
        }
    }

    // Softmax normalize each row
    for i in 0..size {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..size {
            max_val = max_val.max(heatmap[i * size + j]);
        }

        let mut sum = 0.0f32;
        for j in 0..size {
            heatmap[i * size + j] = (heatmap[i * size + j] - max_val).exp();
            sum += heatmap[i * size + j];
        }

        if sum > 0.0 {
            for j in 0..size {
                heatmap[i * size + j] /= sum;
            }
        }
    }

    heatmap
}

/// Decode Sudoku output to readable format
#[wasm_bindgen]
pub fn decode_sudoku_output(output: &[u8]) -> String {
    let mut result = String::new();
    for (i, &v) in output.iter().enumerate() {
        if i > 0 && i % 9 == 0 {
            result.push('\n');
        }
        if v == 0 {
            result.push('.');
        } else {
            result.push((b'0' + v) as char);
        }
    }
    result
}

/// Check if Sudoku solution is valid
#[wasm_bindgen]
pub fn validate_sudoku(grid: &[u8]) -> bool {
    if grid.len() != 81 {
        return false;
    }

    // Check rows
    for i in 0..9 {
        let mut seen = [false; 10];
        for j in 0..9 {
            let v = grid[i * 9 + j] as usize;
            if v < 1 || v > 9 || seen[v] {
                return false;
            }
            seen[v] = true;
        }
    }

    // Check columns
    for j in 0..9 {
        let mut seen = [false; 10];
        for i in 0..9 {
            let v = grid[i * 9 + j] as usize;
            if v < 1 || v > 9 || seen[v] {
                return false;
            }
            seen[v] = true;
        }
    }

    // Check 3x3 boxes
    for box_i in 0..3 {
        for box_j in 0..3 {
            let mut seen = [false; 10];
            for i in 0..3 {
                for j in 0..3 {
                    let v = grid[(box_i * 3 + i) * 9 + (box_j * 3 + j)] as usize;
                    if v < 1 || v > 9 || seen[v] {
                        return false;
                    }
                    seen[v] = true;
                }
            }
        }
    }

    true
}
