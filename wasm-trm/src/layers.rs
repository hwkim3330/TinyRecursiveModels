//! Neural network layers for TRM

use crate::tensor::Tensor;

/// SwiGLU feedforward layer
/// SwiGLU(x) = (W_gate(x) * SiLU(W_up(x))) @ W_down
pub struct SwiGLU {
    pub gate_up: Tensor,  // [hidden_size, inter_size * 2]
    pub down: Tensor,     // [inter_size, hidden_size]
    inter_size: usize,
}

impl SwiGLU {
    pub fn new(hidden_size: usize, expansion: f32) -> Self {
        // Compute intermediate size (similar to PyTorch implementation)
        let inter_size = ((expansion * hidden_size as f32 * 2.0 / 3.0) as usize + 255) / 256 * 256;
        let init_std = 1.0 / (hidden_size as f32).sqrt();

        Self {
            gate_up: Tensor::randn(hidden_size, inter_size * 2, init_std),
            down: Tensor::randn(inter_size, hidden_size, init_std),
            inter_size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [seq_len, hidden_size]
        // gate_up: [hidden_size, inter_size * 2]
        let proj = x.matmul(&self.gate_up);  // [seq_len, inter_size * 2]

        // Split into gate and up
        let mut gate = Tensor::zeros(x.rows, self.inter_size);
        let mut up = Tensor::zeros(x.rows, self.inter_size);

        for i in 0..x.rows {
            for j in 0..self.inter_size {
                gate.data[i * self.inter_size + j] = proj.data[i * self.inter_size * 2 + j];
                up.data[i * self.inter_size + j] = proj.data[i * self.inter_size * 2 + self.inter_size + j];
            }
        }

        // SwiGLU: SiLU(gate) * up
        let gate_activated = gate.silu();
        let gated = gate_activated.mul(&up);

        // Down projection
        gated.matmul(&self.down)
    }
}

/// Reasoning Block (single transformer-like layer)
/// Used in L-level reasoning
pub struct ReasoningBlock {
    pub mlp: SwiGLU,
    pub mlp_t: Option<SwiGLU>,  // Optional transpose MLP for MLP-T variant
    use_mlp_t: bool,
    rms_eps: f32,
}

impl ReasoningBlock {
    pub fn new(hidden_size: usize, expansion: f32, use_mlp_t: bool) -> Self {
        Self {
            mlp: SwiGLU::new(hidden_size, expansion),
            mlp_t: if use_mlp_t {
                // MLP-T operates on transposed dimensions
                Some(SwiGLU::new(hidden_size, expansion))
            } else {
                None
            },
            use_mlp_t,
            rms_eps: 1e-5,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor, input_injection: &Tensor) -> Tensor {
        // Add input injection (residual-like)
        let mut x = hidden_states.add(input_injection);

        // MLP-T variant: operate on transposed tensor
        if self.use_mlp_t {
            if let Some(mlp_t) = &self.mlp_t {
                // Transpose: [seq_len, hidden_size] -> [hidden_size, seq_len]
                let x_t = x.transpose();
                let out_t = mlp_t.forward(&x_t);
                // Transpose back and add residual
                let out = out_t.transpose();
                x = x.add(&out).rms_norm(self.rms_eps);
            }
        }

        // Regular MLP
        let out = self.mlp.forward(&x);
        x.add(&out).rms_norm(self.rms_eps)
    }
}

/// Multi-head Attention (optional, for non-MLP-T variant)
pub struct Attention {
    pub qkv_proj: Tensor,  // [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
    pub o_proj: Tensor,    // [num_heads * head_dim, hidden_size]
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        let init_std = 1.0 / (hidden_size as f32).sqrt();

        Self {
            qkv_proj: Tensor::randn(hidden_size, 3 * hidden_size, init_std),
            o_proj: Tensor::randn(hidden_size, hidden_size, init_std),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.rows;
        let hidden_size = x.cols;

        // Compute Q, K, V
        let qkv = x.matmul(&self.qkv_proj);

        // Split into Q, K, V (simplified, single-head for now)
        let mut q = Tensor::zeros(seq_len, hidden_size);
        let mut k = Tensor::zeros(seq_len, hidden_size);
        let mut v = Tensor::zeros(seq_len, hidden_size);

        for i in 0..seq_len {
            for j in 0..hidden_size {
                q.data[i * hidden_size + j] = qkv.data[i * 3 * hidden_size + j];
                k.data[i * hidden_size + j] = qkv.data[i * 3 * hidden_size + hidden_size + j];
                v.data[i * hidden_size + j] = qkv.data[i * 3 * hidden_size + 2 * hidden_size + j];
            }
        }

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let k_t = k.transpose();
        let scores = q.matmul(&k_t).scale(scale);
        let attn_weights = scores.softmax();
        let attn_output = attn_weights.matmul(&v);

        // Output projection
        attn_output.matmul(&self.o_proj)
    }
}
