/**
 * TRM Chess - Recursive Neural Network for Chess Position Analysis
 * Based on TRM (Tiny Recursive Model) architecture
 */

// Reuse Tensor class from trm-real.js or define if not loaded
if (typeof Tensor === 'undefined') {
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
    }
    window.Tensor = Tensor;
}

/**
 * SwiGLU Feedforward Layer for Chess
 */
class ChessSwiGLU {
    constructor(hiddenSize, expansion = 2.0) {
        const interSize = Math.ceil((expansion * hiddenSize * 2 / 3) / 64) * 64;
        this.interSize = interSize;
        const initStd = 1.0 / Math.sqrt(hiddenSize);

        this.gateUp = Tensor.randn(hiddenSize, interSize * 2, initStd);
        this.down = Tensor.randn(interSize, hiddenSize, initStd);
    }

    forward(x) {
        const proj = x.matmul(this.gateUp);

        const gate = new Tensor(x.rows, this.interSize);
        const up = new Tensor(x.rows, this.interSize);

        for (let i = 0; i < x.rows; i++) {
            for (let j = 0; j < this.interSize; j++) {
                gate.data[i * this.interSize + j] = proj.data[i * this.interSize * 2 + j];
                up.data[i * this.interSize + j] = proj.data[i * this.interSize * 2 + this.interSize + j];
            }
        }

        const gateActivated = gate.silu();
        const gated = gateActivated.mul(up);

        return gated.matmul(this.down);
    }
}

/**
 * Chess Reasoning Block (MLP-T variant)
 */
class ChessReasoningBlock {
    constructor(hiddenSize, expansion = 2.0, useMlpT = true) {
        this.mlp = new ChessSwiGLU(hiddenSize, expansion);
        this.mlpT = useMlpT ? new ChessSwiGLU(hiddenSize, expansion) : null;
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
 * Chess TRM Configuration
 */
class ChessTRMConfig {
    constructor(options = {}) {
        this.hiddenSize = options.hiddenSize || 128;
        this.numHeads = options.numHeads || 4;
        this.expansion = options.expansion || 2.0;
        this.hCycles = options.hCycles || 6;
        this.lCycles = options.lCycles || 12;
        this.lLayers = options.lLayers || 2;
        this.vocabSize = options.vocabSize || 13;  // 12 piece types + empty
        this.seqLen = options.seqLen || 64;        // 8x8 board
        this.useMlpT = options.useMlpT !== undefined ? options.useMlpT : true;
    }
}

/**
 * Main Chess TRM Model
 */
class ChessTRM {
    constructor(config) {
        this.config = config;
        console.log(`Initializing ChessTRM: hidden=${config.hiddenSize}, H=${config.hCycles}, L=${config.lCycles}`);

        const initStd = 1.0 / Math.sqrt(config.hiddenSize);

        // Piece embeddings (13 types: empty + 6 white + 6 black)
        this.embedTokens = Tensor.randn(config.vocabSize, config.hiddenSize, initStd);

        // Position embeddings (64 squares with chess-aware features)
        this.positionEmbed = this.createPositionEmbeddings(config.hiddenSize);

        // L-level reasoning layers
        this.lLayers = [];
        for (let i = 0; i < config.lLayers; i++) {
            this.lLayers.push(new ChessReasoningBlock(config.hiddenSize, config.expansion, config.useMlpT));
        }

        // Move prediction heads
        // From-square head (64 outputs)
        this.fromHead = Tensor.randn(config.hiddenSize, 64, initStd);
        // To-square head (64 outputs)
        this.toHead = Tensor.randn(config.hiddenSize, 64, initStd);

        // Initial states
        this.hInit = Tensor.randn(1, config.hiddenSize, 1.0);
        this.lInit = Tensor.randn(1, config.hiddenSize, 1.0);

        // Current states
        this.zH = null;
        this.zL = null;

        this.reset();
    }

    createPositionEmbeddings(hiddenSize) {
        const embed = Tensor.zeros(64, hiddenSize);

        for (let sq = 0; sq < 64; sq++) {
            const row = Math.floor(sq / 8);
            const col = sq % 8;

            for (let i = 0; i < hiddenSize; i++) {
                // Sinusoidal position encoding
                const freq = Math.pow(10000, -2 * Math.floor(i / 2) / hiddenSize);

                if (i % 4 === 0) {
                    embed.data[sq * hiddenSize + i] = Math.sin(row * freq);
                } else if (i % 4 === 1) {
                    embed.data[sq * hiddenSize + i] = Math.cos(row * freq);
                } else if (i % 4 === 2) {
                    embed.data[sq * hiddenSize + i] = Math.sin(col * freq);
                } else {
                    embed.data[sq * hiddenSize + i] = Math.cos(col * freq);
                }

                // Add chess-specific features
                // Center control bonus
                const centerDist = Math.abs(3.5 - row) + Math.abs(3.5 - col);
                embed.data[sq * hiddenSize + i] += (7 - centerDist) * 0.05 * (i % 8 === 0 ? 1 : 0);

                // King safety zones
                if ((row === 0 || row === 7) && (col < 3 || col > 4)) {
                    embed.data[sq * hiddenSize + i] += 0.1 * (i % 8 === 1 ? 1 : 0);
                }
            }
        }

        return embed;
    }

    reset() {
        const { seqLen, hiddenSize } = this.config;

        this.zH = Tensor.zeros(seqLen, hiddenSize);
        this.zL = Tensor.zeros(seqLen, hiddenSize);

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                this.zH.data[i * hiddenSize + j] = this.hInit.data[j];
                this.zL.data[i * hiddenSize + j] = this.lInit.data[j];
            }
        }
    }

    embedInput(input) {
        const seqLen = input.length;
        const hiddenSize = this.config.hiddenSize;
        const embedded = Tensor.zeros(seqLen, hiddenSize);

        for (let i = 0; i < seqLen; i++) {
            const tokenIdx = input[i];
            if (tokenIdx < this.config.vocabSize) {
                for (let j = 0; j < hiddenSize; j++) {
                    // Combine piece embedding + position embedding
                    embedded.data[i * hiddenSize + j] =
                        this.embedTokens.data[tokenIdx * hiddenSize + j] +
                        this.positionEmbed.data[i * hiddenSize + j];
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

    forward(input) {
        this.reset();
        const inputEmbed = this.embedInput(input);

        for (let h = 0; h < this.config.hCycles; h++) {
            for (let l = 0; l < this.config.lCycles; l++) {
                this.lCycleStep(inputEmbed);
            }
            this.hCycleStep();
        }

        return this.zH;
    }

    // Predict best moves from board state
    predictMoves(board) {
        const { hiddenSize } = this.config;

        // Aggregate hidden states for move prediction
        const globalState = Tensor.zeros(1, hiddenSize);
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                globalState.data[j] += this.zH.data[i * hiddenSize + j];
            }
        }
        for (let j = 0; j < hiddenSize; j++) {
            globalState.data[j] /= 64;
        }

        // Get from-square logits
        const fromLogits = new Float32Array(64);
        for (let sq = 0; sq < 64; sq++) {
            for (let h = 0; h < hiddenSize; h++) {
                fromLogits[sq] += this.zH.data[sq * hiddenSize + h] * this.fromHead.data[h * 64 + sq];
            }
        }

        // Get to-square logits for each from square
        const moves = [];

        // Only consider squares with pieces
        for (let from = 0; from < 64; from++) {
            const piece = board[from];
            if (piece === '.') continue;

            // Check if it's a valid piece to move (white pieces for now)
            const isWhite = piece === piece.toUpperCase();
            // For simplicity, assume white to move

            const toLogits = new Float32Array(64);
            for (let to = 0; to < 64; to++) {
                for (let h = 0; h < hiddenSize; h++) {
                    toLogits[to] += this.zH.data[from * hiddenSize + h] * this.toHead.data[h * 64 + to];
                }
            }

            // Combine from and to scores
            for (let to = 0; to < 64; to++) {
                if (from === to) continue;  // Can't move to same square

                const score = fromLogits[from] + toLogits[to];
                moves.push({
                    from: from,
                    to: to,
                    piece: piece,
                    score: score
                });
            }
        }

        // Sort by score (descending)
        moves.sort((a, b) => b.score - a.score);

        // Apply softmax to top moves for confidence
        const topMoves = moves.slice(0, 10);
        const maxScore = topMoves.length > 0 ? topMoves[0].score : 0;
        let sumExp = 0;
        for (const m of topMoves) {
            m.expScore = Math.exp(m.score - maxScore);
            sumExp += m.expScore;
        }
        for (const m of topMoves) {
            m.score = m.expScore / sumExp;
            delete m.expScore;
        }

        return topMoves;
    }

    getZH() {
        return this.zH.data;
    }

    getZL() {
        return this.zL.data;
    }
}

// Export
window.ChessTRM = ChessTRM;
window.ChessTRMConfig = ChessTRMConfig;
