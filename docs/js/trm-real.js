/**
 * TRM (Tiny Recursive Model) - Pure JavaScript Implementation
 * Real neural network running in browser
 */

class Tensor {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float32Array(rows * cols);
    }

    static zeros(rows, cols) {
        return new Tensor(rows, cols);
    }

    static randn(rows, cols, std = 1.0) {
        const t = new Tensor(rows, cols);
        for (let i = 0; i < t.data.length; i++) {
            // Box-Muller transform
            const u1 = Math.max(Math.random(), 1e-10);
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            t.data[i] = z * std;
        }
        return t;
    }

    clone() {
        const t = new Tensor(this.rows, this.cols);
        t.data.set(this.data);
        return t;
    }

    add(other) {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    mul(other) {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    scale(s) {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * s;
        }
        return result;
    }

    matmul(other) {
        const result = Tensor.zeros(this.rows, other.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i * this.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        return result;
    }

    transpose() {
        const result = new Tensor(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return result;
    }

    rmsNorm(eps = 1e-5) {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            let sumSq = 0;
            for (let j = 0; j < this.cols; j++) {
                const val = this.data[i * this.cols + j];
                sumSq += val * val;
            }
            const rms = Math.sqrt(sumSq / this.cols + eps);
            for (let j = 0; j < this.cols; j++) {
                result.data[i * this.cols + j] = this.data[i * this.cols + j] / rms;
            }
        }
        return result;
    }

    silu() {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            const x = this.data[i];
            result.data[i] = x / (1 + Math.exp(-x));
        }
        return result;
    }

    softmax() {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            let maxVal = -Infinity;
            for (let j = 0; j < this.cols; j++) {
                maxVal = Math.max(maxVal, this.data[i * this.cols + j]);
            }
            let sum = 0;
            for (let j = 0; j < this.cols; j++) {
                const val = Math.exp(this.data[i * this.cols + j] - maxVal);
                result.data[i * this.cols + j] = val;
                sum += val;
            }
            for (let j = 0; j < this.cols; j++) {
                result.data[i * this.cols + j] /= sum;
            }
        }
        return result;
    }

    getActivationStats() {
        let min = Infinity, max = -Infinity, sum = 0;
        for (let i = 0; i < this.data.length; i++) {
            min = Math.min(min, this.data[i]);
            max = Math.max(max, this.data[i]);
            sum += Math.abs(this.data[i]);
        }
        return { min, max, mean: sum / this.data.length };
    }
}

/**
 * SwiGLU Feedforward Layer
 */
class SwiGLU {
    constructor(hiddenSize, expansion = 2.0) {
        const interSize = Math.ceil((expansion * hiddenSize * 2 / 3) / 64) * 64;
        this.interSize = interSize;
        const initStd = 1.0 / Math.sqrt(hiddenSize);

        this.gateUp = Tensor.randn(hiddenSize, interSize * 2, initStd);
        this.down = Tensor.randn(interSize, hiddenSize, initStd);
    }

    forward(x) {
        // x: [seq_len, hidden_size]
        const proj = x.matmul(this.gateUp); // [seq_len, inter_size * 2]

        // Split into gate and up
        const gate = new Tensor(x.rows, this.interSize);
        const up = new Tensor(x.rows, this.interSize);

        for (let i = 0; i < x.rows; i++) {
            for (let j = 0; j < this.interSize; j++) {
                gate.data[i * this.interSize + j] = proj.data[i * this.interSize * 2 + j];
                up.data[i * this.interSize + j] = proj.data[i * this.interSize * 2 + this.interSize + j];
            }
        }

        // SwiGLU: SiLU(gate) * up
        const gateActivated = gate.silu();
        const gated = gateActivated.mul(up);

        return gated.matmul(this.down);
    }
}

/**
 * Reasoning Block (MLP-T variant)
 */
class ReasoningBlock {
    constructor(hiddenSize, expansion = 2.0, useMlpT = true) {
        this.mlp = new SwiGLU(hiddenSize, expansion);
        this.mlpT = useMlpT ? new SwiGLU(hiddenSize, expansion) : null;
        this.useMlpT = useMlpT;
        this.rmsEps = 1e-5;
    }

    forward(hiddenStates, inputInjection) {
        let x = hiddenStates.add(inputInjection);

        if (this.useMlpT && this.mlpT) {
            const xT = x.transpose();
            const outT = this.mlpT.forward(xT);
            const out = outT.transpose();
            x = x.add(out).rmsNorm(this.rmsEps);
        }

        const out = this.mlp.forward(x);
        return x.add(out).rmsNorm(this.rmsEps);
    }
}

/**
 * TRM Configuration
 */
class TRMConfig {
    constructor(options = {}) {
        this.hiddenSize = options.hiddenSize || 128;
        this.numHeads = options.numHeads || 4;
        this.expansion = options.expansion || 2.0;
        this.hCycles = options.hCycles || 3;
        this.lCycles = options.lCycles || 6;
        this.lLayers = options.lLayers || 2;
        this.vocabSize = options.vocabSize || 12;
        this.seqLen = options.seqLen || 81;
        this.useMlpT = options.useMlpT !== undefined ? options.useMlpT : true;
    }

    static forSudoku() {
        return new TRMConfig({
            hiddenSize: 128,
            numHeads: 4,
            expansion: 2.0,
            hCycles: 3,
            lCycles: 6,
            lLayers: 2,
            vocabSize: 12,
            seqLen: 81,
            useMlpT: true
        });
    }

    static forMaze() {
        return new TRMConfig({
            hiddenSize: 128,
            numHeads: 4,
            expansion: 2.0,
            hCycles: 3,
            lCycles: 4,
            lLayers: 2,
            vocabSize: 4,
            seqLen: 225,
            useMlpT: true
        });
    }
}

/**
 * Main TRM Model
 */
class TRM {
    constructor(config) {
        this.config = config;
        console.log(`Initializing TRM: hidden=${config.hiddenSize}, H=${config.hCycles}, L=${config.lCycles}`);

        const initStd = 1.0 / Math.sqrt(config.hiddenSize);

        // Token embeddings
        this.embedTokens = Tensor.randn(config.vocabSize, config.hiddenSize, initStd);

        // L-level reasoning layers
        this.lLayers = [];
        for (let i = 0; i < config.lLayers; i++) {
            this.lLayers.push(new ReasoningBlock(config.hiddenSize, config.expansion, config.useMlpT));
        }

        // LM head
        this.lmHead = Tensor.randn(config.hiddenSize, config.vocabSize, initStd);

        // Initial states
        this.hInit = Tensor.randn(1, config.hiddenSize, 1.0);
        this.lInit = Tensor.randn(1, config.hiddenSize, 1.0);

        // Current states
        this.zH = null;
        this.zL = null;
        this.currentH = 0;
        this.currentL = 0;
        this.totalSteps = 0;

        this.reset();
    }

    reset() {
        const { seqLen, hiddenSize } = this.config;

        this.zH = Tensor.zeros(seqLen, hiddenSize);
        this.zL = Tensor.zeros(seqLen, hiddenSize);

        // Initialize with init states
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                this.zH.data[i * hiddenSize + j] = this.hInit.data[j];
                this.zL.data[i * hiddenSize + j] = this.lInit.data[j];
            }
        }

        this.currentH = 0;
        this.currentL = 0;
        this.totalSteps = 0;
    }

    embedInput(input) {
        const seqLen = input.length;
        const hiddenSize = this.config.hiddenSize;
        const embedded = Tensor.zeros(seqLen, hiddenSize);

        for (let i = 0; i < seqLen; i++) {
            const tokenIdx = input[i];
            if (tokenIdx < this.config.vocabSize) {
                for (let j = 0; j < hiddenSize; j++) {
                    embedded.data[i * hiddenSize + j] =
                        this.embedTokens.data[tokenIdx * hiddenSize + j];
                }
            }
        }

        return embedded.scale(Math.sqrt(hiddenSize));
    }

    lCycleStep(inputEmbed) {
        const combined = this.zH.add(inputEmbed);
        for (const layer of this.lLayers) {
            this.zL = layer.forward(this.zL, combined);
        }
    }

    hCycleStep() {
        for (const layer of this.lLayers) {
            this.zH = layer.forward(this.zH, this.zL);
        }
    }

    step(input) {
        const inputEmbed = this.embedInput(input);

        if (this.currentL < this.config.lCycles) {
            this.lCycleStep(inputEmbed);
            this.currentL++;
            this.totalSteps++;
        } else if (this.currentH < this.config.hCycles) {
            this.hCycleStep();
            this.currentH++;
            this.currentL = 0;
        }

        const confidence = this.computeConfidence();
        const stats = this.zH.getActivationStats();

        return {
            hStep: this.currentH,
            lStep: this.currentL,
            totalSteps: this.totalSteps,
            confidence,
            zHActivations: Array.from(this.zH.data.slice(0, 64)),
            zLActivations: Array.from(this.zL.data.slice(0, 64)),
            stats
        };
    }

    forward(input) {
        this.reset();
        const inputEmbed = this.embedInput(input);

        for (let h = 0; h < this.config.hCycles; h++) {
            for (let l = 0; l < this.config.lCycles; l++) {
                this.lCycleStep(inputEmbed);
            }
            this.hCycleStep();
        }

        return this.predictOutput();
    }

    predictOutput() {
        const { seqLen, hiddenSize, vocabSize } = this.config;
        const output = new Uint8Array(seqLen);

        for (let i = 0; i < seqLen; i++) {
            const logits = new Float32Array(vocabSize);

            for (let v = 0; v < vocabSize; v++) {
                for (let h = 0; h < hiddenSize; h++) {
                    logits[v] += this.zH.data[i * hiddenSize + h] *
                                 this.lmHead.data[h * vocabSize + v];
                }
            }

            // Argmax
            let maxIdx = 0;
            let maxVal = logits[0];
            for (let v = 1; v < vocabSize; v++) {
                if (logits[v] > maxVal) {
                    maxVal = logits[v];
                    maxIdx = v;
                }
            }
            output[i] = maxIdx;
        }

        return output;
    }

    computeConfidence() {
        let sum = 0;
        for (let i = 0; i < this.zH.data.length; i++) {
            sum += Math.abs(this.zH.data[i]);
        }
        const mean = sum / this.zH.data.length;
        return 1.0 / (1.0 + Math.exp(-mean));
    }

    getCurrentOutput() {
        return this.predictOutput();
    }

    isComplete() {
        return this.currentH >= this.config.hCycles;
    }

    getZH() {
        return this.zH.data;
    }

    getZL() {
        return this.zL.data;
    }
}

/**
 * Utility functions
 */
function normalizeActivations(activations) {
    if (activations.length === 0) return new Uint8Array(0);

    let min = Infinity, max = -Infinity;
    for (let i = 0; i < activations.length; i++) {
        min = Math.min(min, activations[i]);
        max = Math.max(max, activations[i]);
    }

    const range = max - min;
    if (range < 1e-6) {
        return new Uint8Array(activations.length).fill(128);
    }

    const result = new Uint8Array(activations.length);
    for (let i = 0; i < activations.length; i++) {
        result[i] = Math.floor((activations[i] - min) / range * 255);
    }
    return result;
}

function validateSudoku(grid) {
    if (grid.length !== 81) return false;

    // Check rows
    for (let i = 0; i < 9; i++) {
        const seen = new Set();
        for (let j = 0; j < 9; j++) {
            const v = grid[i * 9 + j];
            if (v < 1 || v > 9 || seen.has(v)) return false;
            seen.add(v);
        }
    }

    // Check columns
    for (let j = 0; j < 9; j++) {
        const seen = new Set();
        for (let i = 0; i < 9; i++) {
            const v = grid[i * 9 + j];
            if (v < 1 || v > 9 || seen.has(v)) return false;
            seen.add(v);
        }
    }

    // Check 3x3 boxes
    for (let bi = 0; bi < 3; bi++) {
        for (let bj = 0; bj < 3; bj++) {
            const seen = new Set();
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const v = grid[(bi * 3 + i) * 9 + (bj * 3 + j)];
                    if (v < 1 || v > 9 || seen.has(v)) return false;
                    seen.add(v);
                }
            }
        }
    }

    return true;
}

// Export for use
window.TRM = TRM;
window.TRMConfig = TRMConfig;
window.Tensor = Tensor;
window.normalizeActivations = normalizeActivations;
window.validateSudoku = validateSudoku;
