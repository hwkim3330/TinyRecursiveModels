/**
 * TRM - Tiny Recursive Model (JavaScript Implementation)
 *
 * This is a simplified simulation of the TRM architecture for demonstration.
 * The actual model uses PyTorch with trained weights.
 *
 * Key concepts:
 * - Recursive improvement of answers
 * - H_cycles: High-level improvement steps
 * - L_cycles: Low-level reasoning cycles within each H step
 */

class TinyRecursiveModel {
    constructor(config = {}) {
        this.config = {
            H_cycles: config.H_cycles || 3,
            L_cycles: config.L_cycles || 6,
            hidden_size: config.hidden_size || 256,
            ...config
        };

        this.listeners = [];
    }

    /**
     * Subscribe to recursion step updates
     */
    onStep(callback) {
        this.listeners.push(callback);
    }

    /**
     * Emit step update to all listeners
     */
    emitStep(step) {
        this.listeners.forEach(cb => cb(step));
    }

    /**
     * Simulate the recursive reasoning process
     * @param {Object} input - Problem input (e.g., sudoku grid, maze)
     * @param {Function} updateFn - Function to update answer at each step
     * @returns {Object} - Final answer and confidence
     */
    async solve(input, updateFn) {
        let z_H = this.initializeLatent(input);  // Answer representation
        let z_L = this.initializeLatent(input);  // Latent reasoning state
        let confidence = 0;
        let totalSteps = 0;

        const startTime = performance.now();

        for (let h = 0; h < this.config.H_cycles; h++) {
            // L-level reasoning cycles
            for (let l = 0; l < this.config.L_cycles; l++) {
                totalSteps++;

                // Simulate reasoning step
                z_L = this.reasoningStep(z_L, z_H, input);

                this.emitStep({
                    type: 'L_cycle',
                    h_step: h + 1,
                    l_step: l + 1,
                    total: totalSteps,
                    confidence: this.computeConfidence(z_L)
                });

                // Small delay for visualization
                await this.sleep(50);
            }

            // Update answer representation
            z_H = this.updateAnswer(z_H, z_L);
            confidence = this.computeConfidence(z_H);

            // Apply update function to get current answer
            const currentAnswer = updateFn(z_H, h);

            this.emitStep({
                type: 'H_cycle',
                h_step: h + 1,
                total: totalSteps,
                confidence: confidence,
                answer: currentAnswer
            });

            await this.sleep(100);
        }

        const endTime = performance.now();

        return {
            answer: updateFn(z_H, this.config.H_cycles - 1),
            confidence: confidence,
            totalSteps: totalSteps,
            time: Math.round(endTime - startTime)
        };
    }

    /**
     * Initialize latent representation
     */
    initializeLatent(input) {
        // In real TRM, this would be learned initial states
        // Here we use the input as a starting point
        if (Array.isArray(input)) {
            return input.map(row =>
                Array.isArray(row) ? row.map(v => v || Math.random() * 0.1) : (row || Math.random() * 0.1)
            );
        }
        return input;
    }

    /**
     * Simulate L-level reasoning step
     * z_L = L_level(z_L, z_H + input)
     */
    reasoningStep(z_L, z_H, input) {
        // This simulates the MLP/Attention layers
        // In real TRM: SwiGLU + RMS norm + residual
        if (Array.isArray(z_L)) {
            return z_L.map((row, i) => {
                if (Array.isArray(row)) {
                    return row.map((val, j) => {
                        const h_val = Array.isArray(z_H[i]) ? z_H[i][j] : z_H[i];
                        const in_val = Array.isArray(input[i]) ? input[i][j] : (input[i] || 0);
                        // Simulated SiLU activation and mixing
                        const combined = val + 0.1 * (h_val + in_val);
                        return combined * (1 / (1 + Math.exp(-combined))); // SiLU-like
                    });
                }
                return val;
            });
        }
        return z_L;
    }

    /**
     * Update answer representation
     * z_H = L_level(z_H, z_L)
     */
    updateAnswer(z_H, z_L) {
        if (Array.isArray(z_H)) {
            return z_H.map((row, i) => {
                if (Array.isArray(row)) {
                    return row.map((val, j) => {
                        const l_val = Array.isArray(z_L[i]) ? z_L[i][j] : z_L[i];
                        // Gradual refinement
                        return val * 0.8 + l_val * 0.2;
                    });
                }
                return row;
            });
        }
        return z_H;
    }

    /**
     * Compute confidence score
     */
    computeConfidence(latent) {
        // Simulated confidence based on "stability" of representation
        if (Array.isArray(latent)) {
            let sum = 0;
            let count = 0;
            const flatten = (arr) => {
                arr.forEach(item => {
                    if (Array.isArray(item)) flatten(item);
                    else if (typeof item === 'number') {
                        sum += Math.abs(item);
                        count++;
                    }
                });
            };
            flatten(latent);
            // Higher values = more confident (saturated)
            return Math.min(1, (sum / count) * 0.3);
        }
        return 0.5;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Global TRM instance
window.TRM = new TinyRecursiveModel();

// Helper function to display recursion steps
function displayStep(containerId, step) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const stepEl = document.createElement('div');
    stepEl.className = 'recursion-step' + (step.type === 'H_cycle' ? ' active' : '');

    if (step.type === 'L_cycle') {
        stepEl.innerHTML = `
            <div class="step-number" style="background: rgba(6, 182, 212, 0.3);">${step.l_step}</div>
            <div>
                <div style="font-size: 0.875rem;">L-cycle ${step.l_step}/${window.TRM.config.L_cycles}</div>
                <div style="font-size: 0.75rem; color: #6b7280;">H-step ${step.h_step}, reasoning...</div>
            </div>
        `;
    } else {
        stepEl.innerHTML = `
            <div class="step-number">${step.h_step}</div>
            <div style="flex: 1;">
                <div style="font-size: 0.875rem; font-weight: 600;">H-cycle ${step.h_step}/${window.TRM.config.H_cycles}</div>
                <div style="font-size: 0.75rem; color: #10b981;">Confidence: ${(step.confidence * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    // Keep only last 10 steps visible
    while (container.children.length >= 10) {
        container.removeChild(container.firstChild);
    }

    container.appendChild(stepEl);
    container.scrollTop = container.scrollHeight;
}

// Update confidence display
function updateConfidence(prefix, confidence) {
    const confEl = document.getElementById(`${prefix}-confidence`);
    const barEl = document.getElementById(`${prefix}-confidence-bar`);

    if (confEl) confEl.textContent = `${(confidence * 100).toFixed(1)}%`;
    if (barEl) barEl.style.width = `${confidence * 100}%`;
}
