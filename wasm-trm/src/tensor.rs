//! Simple tensor implementation for WASM

use rand::Rng;

/// 2D Tensor (row-major layout)
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    /// Create a zero tensor
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a tensor with random normal values
    pub fn randn(rows: usize, cols: usize, std: f32) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                z * std
            })
            .collect();

        Self { data, rows, cols }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows, "Matrix dimensions don't match for matmul");

        let mut result = Tensor::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0f32;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }

        result
    }

    /// RMS Normalization
    pub fn rms_norm(&self, eps: f32) -> Tensor {
        let mut result = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            // Compute RMS for this row
            let mut sum_sq = 0.0f32;
            for j in 0..self.cols {
                let val = self.data[i * self.cols + j];
                sum_sq += val * val;
            }
            let rms = (sum_sq / self.cols as f32 + eps).sqrt();

            // Normalize
            for j in 0..self.cols {
                result.data[i * self.cols + j] = self.data[i * self.cols + j] / rms;
            }
        }

        result
    }

    /// SiLU activation (Swish): x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();

        Tensor {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Transpose
    pub fn transpose(&self) -> Tensor {
        let mut result = Tensor::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        result
    }

    /// Get row as slice
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Set row from slice
    pub fn set_row(&mut self, i: usize, values: &[f32]) {
        let start = i * self.cols;
        self.data[start..start + self.cols].copy_from_slice(values);
    }

    /// Softmax along rows
    pub fn softmax(&self) -> Tensor {
        let mut result = Tensor::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..self.cols {
                max_val = max_val.max(self.data[i * self.cols + j]);
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for j in 0..self.cols {
                let val = (self.data[i * self.cols + j] - max_val).exp();
                result.data[i * self.cols + j] = val;
                sum += val;
            }

            // Normalize
            for j in 0..self.cols {
                result.data[i * self.cols + j] /= sum;
            }
        }

        result
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Standard deviation
    pub fn std(&self) -> f32 {
        let mean = self.mean();
        let variance: f32 = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.data.len() as f32;
        variance.sqrt()
    }
}
