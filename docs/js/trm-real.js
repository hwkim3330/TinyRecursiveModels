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
 * Main TRM Model with Continuous Learning
 */
class TRM {
    constructor(config) {
        this.config = config;
        this.learningRate = 0.001;
        this.learningEnabled = false;
        this.momentum = 0.9;
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

    // Enable/disable learning
    setLearning(enabled, lr = 0.001) {
        this.learningEnabled = enabled;
        this.learningRate = lr;
        console.log(`Learning ${enabled ? 'enabled' : 'disabled'}, lr=${lr}`);
    }

    // Compute loss for a single position (cross-entropy like)
    computeLoss(predicted, target) {
        // Simplified loss: negative log probability
        const vocabSize = this.config.vocabSize;
        const hiddenSize = this.config.hiddenSize;
        let totalLoss = 0;
        let count = 0;

        for (let i = 0; i < target.length; i++) {
            if (target[i] > 0) { // Only for known targets
                // Get logits for this position
                const logits = new Float32Array(vocabSize);
                for (let v = 0; v < vocabSize; v++) {
                    for (let h = 0; h < hiddenSize; h++) {
                        logits[v] += this.zH.data[i * hiddenSize + h] *
                                     this.lmHead.data[h * vocabSize + v];
                    }
                }

                // Softmax
                let maxLogit = Math.max(...logits);
                let sumExp = 0;
                for (let v = 0; v < vocabSize; v++) {
                    sumExp += Math.exp(logits[v] - maxLogit);
                }

                // Cross-entropy loss
                const targetIdx = target[i];
                const logProb = logits[targetIdx] - maxLogit - Math.log(sumExp);
                totalLoss -= logProb;
                count++;
            }
        }

        return count > 0 ? totalLoss / count : 0;
    }

    // Simple gradient descent update based on Sudoku constraints
    learnFromConstraints(input, output) {
        if (!this.learningEnabled) return 0;

        const lr = this.learningRate;
        const hiddenSize = this.config.hiddenSize;
        const vocabSize = this.config.vocabSize;
        let totalUpdate = 0;

        // For each position, push embeddings toward correct values
        for (let i = 0; i < 81; i++) {
            const predicted = output[i];
            const given = input[i];

            // Skip if given (already correct)
            if (given > 0) continue;

            // Check Sudoku constraints for this position
            const row = Math.floor(i / 9);
            const col = i % 9;
            const box = Math.floor(row / 3) * 3 + Math.floor(col / 3);

            // Collect used numbers in row, col, box
            const used = new Set();
            for (let j = 0; j < 9; j++) {
                // Row
                const rowVal = output[row * 9 + j];
                if (rowVal > 0 && rowVal <= 9) used.add(rowVal);
                // Col
                const colVal = output[j * 9 + col];
                if (colVal > 0 && colVal <= 9) used.add(colVal);
            }
            // Box
            const boxRow = Math.floor(row / 3) * 3;
            const boxCol = Math.floor(col / 3) * 3;
            for (let r = 0; r < 3; r++) {
                for (let c = 0; c < 3; c++) {
                    const boxVal = output[(boxRow + r) * 9 + (boxCol + c)];
                    if (boxVal > 0 && boxVal <= 9) used.add(boxVal);
                }
            }

            // Find valid candidates
            const candidates = [];
            for (let v = 1; v <= 9; v++) {
                if (!used.has(v)) candidates.push(v);
            }

            // If current prediction is invalid, adjust embeddings
            if (predicted < 1 || predicted > 9 || used.has(predicted)) {
                if (candidates.length > 0) {
                    // Push toward a valid candidate
                    const target = candidates[Math.floor(Math.random() * candidates.length)];

                    // Update token embedding for target
                    for (let h = 0; h < hiddenSize; h++) {
                        const delta = (this.embedTokens.data[target * hiddenSize + h] -
                                      this.zH.data[i * hiddenSize + h]) * lr;
                        this.zH.data[i * hiddenSize + h] += delta;
                        totalUpdate += Math.abs(delta);
                    }
                }
            }
        }

        return totalUpdate;
    }

    // Train on current puzzle with self-supervision
    trainStep(input) {
        if (!this.learningEnabled) return { loss: 0, updates: 0 };

        // Forward pass
        const output = this.forward(input);

        // Compute constraint-based loss
        const loss = this.computeConstraintLoss(input, output);

        // Learn from constraints
        const updates = this.learnFromConstraints(input, output);

        // Also slightly adjust LM head based on confident predictions
        this.adjustLMHead(input, output);

        return { loss, updates, output };
    }

    // Compute constraint violation loss
    computeConstraintLoss(input, output) {
        let violations = 0;

        // Check rows
        for (let r = 0; r < 9; r++) {
            const seen = new Set();
            for (let c = 0; c < 9; c++) {
                const v = output[r * 9 + c];
                if (v > 0 && v <= 9) {
                    if (seen.has(v)) violations++;
                    seen.add(v);
                }
            }
        }

        // Check columns
        for (let c = 0; c < 9; c++) {
            const seen = new Set();
            for (let r = 0; r < 9; r++) {
                const v = output[r * 9 + c];
                if (v > 0 && v <= 9) {
                    if (seen.has(v)) violations++;
                    seen.add(v);
                }
            }
        }

        // Check boxes
        for (let br = 0; br < 3; br++) {
            for (let bc = 0; bc < 3; bc++) {
                const seen = new Set();
                for (let r = 0; r < 3; r++) {
                    for (let c = 0; c < 3; c++) {
                        const v = output[(br * 3 + r) * 9 + (bc * 3 + c)];
                        if (v > 0 && v <= 9) {
                            if (seen.has(v)) violations++;
                            seen.add(v);
                        }
                    }
                }
            }
        }

        // Count empty cells
        const empty = output.filter(v => v < 1 || v > 9).length;

        return violations + empty * 0.5;
    }

    // Adjust LM head weights slightly
    adjustLMHead(input, output) {
        const lr = this.learningRate * 0.1;
        const hiddenSize = this.config.hiddenSize;
        const vocabSize = this.config.vocabSize;

        for (let i = 0; i < 81; i++) {
            const given = input[i];
            if (given > 0) {
                // Strengthen connection for given values
                for (let h = 0; h < hiddenSize; h++) {
                    const idx = h * vocabSize + given;
                    this.lmHead.data[idx] += this.zH.data[i * hiddenSize + h] * lr * 0.01;
                }
            }
        }
    }

    // Run multiple training iterations
    train(input, iterations = 10, callback = null) {
        const results = [];
        for (let i = 0; i < iterations; i++) {
            const result = this.trainStep(input);
            results.push(result);
            if (callback) callback(i, result);
        }
        return results;
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
