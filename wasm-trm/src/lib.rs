//! TRM - Tiny Recursive Model (WASM Implementation)
//!
//! This is a real implementation of the TRM architecture from the paper
//! "Less is More: Recursive Reasoning with Tiny Networks"

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use rand::Rng;

mod tensor;
mod layers;
mod model;

pub use tensor::Tensor;
pub use layers::*;
pub use model::*;

// Console logging for debug
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// TRM Configuration
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct TRMConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub expansion: f32,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub l_layers: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub use_mlp_t: bool,
}

#[wasm_bindgen]
impl TRMConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            hidden_size: 256,
            num_heads: 8,
            expansion: 4.0,
            h_cycles: 3,
            l_cycles: 6,
            l_layers: 2,
            vocab_size: 12,  // 0-9 + empty + special
            seq_len: 81,     // 9x9 sudoku
            use_mlp_t: true, // MLP-T variant (faster)
        }
    }

    pub fn for_sudoku() -> Self {
        Self {
            hidden_size: 128,
            num_heads: 4,
            expansion: 2.0,
            h_cycles: 3,
            l_cycles: 6,
            l_layers: 2,
            vocab_size: 12,
            seq_len: 81,
            use_mlp_t: true,
        }
    }

    pub fn for_maze() -> Self {
        Self {
            hidden_size: 128,
            num_heads: 4,
            expansion: 2.0,
            h_cycles: 3,
            l_cycles: 4,
            l_layers: 2,
            vocab_size: 4, // wall, path, start, end
            seq_len: 225,  // 15x15
            use_mlp_t: true,
        }
    }
}

/// Step information for visualization
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct StepInfo {
    pub h_step: usize,
    pub l_step: usize,
    pub total_steps: usize,
    pub confidence: f32,
    #[wasm_bindgen(skip)]
    pub z_h_activations: Vec<f32>,
    #[wasm_bindgen(skip)]
    pub z_l_activations: Vec<f32>,
    #[wasm_bindgen(skip)]
    pub attention_weights: Vec<f32>,
}

#[wasm_bindgen]
impl StepInfo {
    pub fn get_z_h_activations(&self) -> Vec<f32> {
        self.z_h_activations.clone()
    }

    pub fn get_z_l_activations(&self) -> Vec<f32> {
        self.z_l_activations.clone()
    }

    pub fn get_attention_weights(&self) -> Vec<f32> {
        self.attention_weights.clone()
    }
}

/// Main TRM Model
#[wasm_bindgen]
pub struct TRM {
    config: TRMConfig,
    // Embeddings
    embed_tokens: Tensor,
    // L-level layers (reasoning)
    l_layers: Vec<ReasoningBlock>,
    // LM head for output
    lm_head: Tensor,
    // Initial states
    h_init: Tensor,
    l_init: Tensor,
    // Current state
    z_h: Tensor,
    z_l: Tensor,
    // Step tracking
    current_h: usize,
    current_l: usize,
    total_steps: usize,
}

#[wasm_bindgen]
impl TRM {
    #[wasm_bindgen(constructor)]
    pub fn new(config: TRMConfig) -> Self {
        console_log!("Initializing TRM with hidden_size={}, H={}, L={}",
            config.hidden_size, config.h_cycles, config.l_cycles);

        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let seq_len = config.seq_len;

        // Initialize with random weights (in production, load from file)
        let mut rng = rand::thread_rng();
        let init_std = 1.0 / (hidden_size as f32).sqrt();

        // Token embeddings [vocab_size x hidden_size]
        let embed_tokens = Tensor::randn(vocab_size, hidden_size, init_std);

        // L-level reasoning layers
        let mut l_layers = Vec::new();
        for _ in 0..config.l_layers {
            l_layers.push(ReasoningBlock::new(hidden_size, config.expansion, config.use_mlp_t));
        }

        // LM head [hidden_size x vocab_size]
        let lm_head = Tensor::randn(hidden_size, vocab_size, init_std);

        // Initial states
        let h_init = Tensor::randn(1, hidden_size, 1.0);
        let l_init = Tensor::randn(1, hidden_size, 1.0);

        // Current states [seq_len x hidden_size]
        let z_h = Tensor::zeros(seq_len, hidden_size);
        let z_l = Tensor::zeros(seq_len, hidden_size);

        Self {
            config,
            embed_tokens,
            l_layers,
            lm_head,
            h_init,
            l_init,
            z_h,
            z_l,
            current_h: 0,
            current_l: 0,
            total_steps: 0,
        }
    }

    /// Reset the model state
    pub fn reset(&mut self) {
        let seq_len = self.config.seq_len;
        let hidden_size = self.config.hidden_size;

        // Reset to initial states (broadcast h_init and l_init)
        self.z_h = Tensor::zeros(seq_len, hidden_size);
        self.z_l = Tensor::zeros(seq_len, hidden_size);

        // Initialize each position with the initial state
        for i in 0..seq_len {
            for j in 0..hidden_size {
                self.z_h.data[i * hidden_size + j] = self.h_init.data[j];
                self.z_l.data[i * hidden_size + j] = self.l_init.data[j];
            }
        }

        self.current_h = 0;
        self.current_l = 0;
        self.total_steps = 0;
    }

    /// Embed input tokens
    fn embed_input(&self, input: &[u8]) -> Tensor {
        let seq_len = input.len();
        let hidden_size = self.config.hidden_size;
        let mut embedded = Tensor::zeros(seq_len, hidden_size);

        for (i, &token) in input.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx < self.config.vocab_size {
                for j in 0..hidden_size {
                    embedded.data[i * hidden_size + j] =
                        self.embed_tokens.data[token_idx * hidden_size + j];
                }
            }
        }

        // Scale by sqrt(hidden_size)
        let scale = (hidden_size as f32).sqrt();
        for v in embedded.data.iter_mut() {
            *v *= scale;
        }

        embedded
    }

    /// Perform one L-cycle step
    fn l_cycle_step(&mut self, input_embed: &Tensor) {
        // z_L = L_level(z_L, z_H + input)
        let combined = self.z_h.add(input_embed);

        for layer in &self.l_layers {
            self.z_l = layer.forward(&self.z_l, &combined);
        }
    }

    /// Perform one H-cycle step
    fn h_cycle_step(&mut self) {
        // z_H = L_level(z_H, z_L)
        for layer in &self.l_layers {
            self.z_h = layer.forward(&self.z_h, &self.z_l);
        }
    }

    /// Run one complete reasoning step and return step info
    pub fn step(&mut self, input: &[u8]) -> StepInfo {
        let input_embed = self.embed_input(input);

        if self.current_l < self.config.l_cycles {
            // L-cycle
            self.l_cycle_step(&input_embed);
            self.current_l += 1;
            self.total_steps += 1;
        } else if self.current_h < self.config.h_cycles {
            // H-cycle (update answer)
            self.h_cycle_step();
            self.current_h += 1;
            self.current_l = 0;
        }

        // Compute confidence from z_H activation magnitudes
        let confidence = self.compute_confidence();

        // Get activations for visualization (sample first few dimensions)
        let sample_size = 64.min(self.config.hidden_size);
        let z_h_sample: Vec<f32> = self.z_h.data.iter().take(sample_size).cloned().collect();
        let z_l_sample: Vec<f32> = self.z_l.data.iter().take(sample_size).cloned().collect();

        StepInfo {
            h_step: self.current_h,
            l_step: self.current_l,
            total_steps: self.total_steps,
            confidence,
            z_h_activations: z_h_sample,
            z_l_activations: z_l_sample,
            attention_weights: vec![], // TODO: add if using attention
        }
    }

    /// Run full inference
    pub fn forward(&mut self, input: &[u8]) -> Vec<u8> {
        self.reset();
        let input_embed = self.embed_input(input);

        // H-cycles
        for _h in 0..self.config.h_cycles {
            // L-cycles (update latent)
            for _l in 0..self.config.l_cycles {
                self.l_cycle_step(&input_embed);
            }
            // Update answer
            self.h_cycle_step();
        }

        // Output via LM head
        self.predict_output()
    }

    /// Get predicted output from z_H
    fn predict_output(&self) -> Vec<u8> {
        let seq_len = self.config.seq_len;
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let mut output = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let mut logits = vec![0.0f32; vocab_size];

            // Compute logits: z_H[i] @ lm_head
            for v in 0..vocab_size {
                for h in 0..hidden_size {
                    logits[v] += self.z_h.data[i * hidden_size + h]
                        * self.lm_head.data[h * vocab_size + v];
                }
            }

            // Argmax
            let mut max_idx = 0;
            let mut max_val = logits[0];
            for (idx, &val) in logits.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }

            output.push(max_idx as u8);
        }

        output
    }

    /// Compute confidence score
    fn compute_confidence(&self) -> f32 {
        // Use activation magnitude as proxy for confidence
        let sum: f32 = self.z_h.data.iter().map(|x| x.abs()).sum();
        let mean = sum / self.z_h.data.len() as f32;

        // Sigmoid-like transform to 0-1 range
        1.0 / (1.0 + (-mean).exp())
    }

    /// Get current output prediction
    pub fn get_current_output(&self) -> Vec<u8> {
        self.predict_output()
    }

    /// Check if inference is complete
    pub fn is_complete(&self) -> bool {
        self.current_h >= self.config.h_cycles
    }

    /// Get z_H as flat array for visualization
    pub fn get_z_h(&self) -> Vec<f32> {
        self.z_h.data.clone()
    }

    /// Get z_L as flat array for visualization
    pub fn get_z_l(&self) -> Vec<f32> {
        self.z_l.data.clone()
    }
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("TRM WASM module initialized!");
}
