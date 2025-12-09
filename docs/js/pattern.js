/**
 * Pattern Recognition Demo for TRM
 * Demonstrates ARC-AGI style pattern recognition
 */

// ARC-AGI style patterns (input -> output transformations)
const PATTERNS = [
    {
        name: 'Rotate 90',
        input: [
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ],
        output: [
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 0]
        ],
        transform: (grid) => {
            const n = grid.length;
            return grid[0].map((_, i) => grid.map(row => row[i]).reverse());
        }
    },
    {
        name: 'Flip Horizontal',
        input: [
            [1, 2, 0],
            [0, 2, 0],
            [0, 2, 3]
        ],
        output: [
            [0, 2, 1],
            [0, 2, 0],
            [3, 2, 0]
        ],
        transform: (grid) => grid.map(row => [...row].reverse())
    },
    {
        name: 'Fill Pattern',
        input: [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ],
        output: [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ],
        transform: (grid) => {
            // Fill enclosed area
            const n = grid.length;
            const result = grid.map(row => [...row]);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (result[i][j] === 0) {
                        // Check if surrounded by 1s in some pattern
                        let hasTop = false, hasBottom = false, hasLeft = false, hasRight = false;
                        for (let k = 0; k < i; k++) if (grid[k][j] === 1) hasTop = true;
                        for (let k = i + 1; k < n; k++) if (grid[k][j] === 1) hasBottom = true;
                        for (let k = 0; k < j; k++) if (grid[i][k] === 1) hasLeft = true;
                        for (let k = j + 1; k < n; k++) if (grid[i][k] === 1) hasRight = true;
                        if (hasTop && hasBottom && hasLeft && hasRight) {
                            result[i][j] = 1;
                        }
                    }
                }
            }
            return result;
        }
    },
    {
        name: 'Scale 2x',
        input: [
            [1, 2],
            [3, 4]
        ],
        output: [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ],
        transform: (grid) => {
            const result = [];
            for (const row of grid) {
                const newRow = [];
                for (const cell of row) {
                    newRow.push(cell, cell);
                }
                result.push([...newRow], [...newRow]);
            }
            return result;
        }
    },
    {
        name: 'Color Swap',
        input: [
            [1, 0, 2],
            [0, 1, 0],
            [2, 0, 1]
        ],
        output: [
            [2, 0, 1],
            [0, 2, 0],
            [1, 0, 2]
        ],
        transform: (grid) => grid.map(row => row.map(c => c === 1 ? 2 : c === 2 ? 1 : c))
    }
];

// Colors for pattern display
const PATTERN_COLORS = [
    '#1e293b', // 0 - dark
    '#ef4444', // 1 - red
    '#3b82f6', // 2 - blue
    '#10b981', // 3 - green
    '#f59e0b', // 4 - yellow
    '#8b5cf6', // 5 - purple
    '#ec4899', // 6 - pink
    '#06b6d4', // 7 - cyan
    '#f97316', // 8 - orange
    '#6366f1'  // 9 - indigo
];

let currentPattern = null;
let predictedOutput = null;
let solvingPattern = false;

/**
 * Initialize Pattern demo
 */
function initPattern() {
    newPattern();
}

/**
 * Select a new random pattern
 */
function newPattern() {
    if (solvingPattern) return;

    const idx = Math.floor(Math.random() * PATTERNS.length);
    currentPattern = PATTERNS[idx];
    predictedOutput = null;

    renderPatternGrid('pattern-input', currentPattern.input);
    renderPatternGrid('pattern-output', null, currentPattern.output.length, currentPattern.output[0].length);
    clearSteps('pattern-steps');
}

/**
 * Render a pattern grid
 */
function renderPatternGrid(containerId, grid, rows = null, cols = null) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (!grid) {
        // Empty grid placeholder
        const r = rows || 3;
        const c = cols || 3;
        container.style.gridTemplateColumns = `repeat(${c}, 28px)`;
        for (let i = 0; i < r * c; i++) {
            const cell = document.createElement('div');
            cell.style.cssText = `width: 28px; height: 28px; background: #1e293b; border-radius: 4px;`;
            container.appendChild(cell);
        }
        return;
    }

    const numRows = grid.length;
    const numCols = grid[0].length;
    container.style.gridTemplateColumns = `repeat(${numCols}, 28px)`;

    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const cell = document.createElement('div');
            const value = grid[i][j];
            cell.style.cssText = `
                width: 28px;
                height: 28px;
                background: ${PATTERN_COLORS[value] || PATTERN_COLORS[0]};
                border-radius: 4px;
                transition: all 0.3s;
            `;
            container.appendChild(cell);
        }
    }
}

/**
 * Solve pattern using TRM simulation
 */
async function solvePattern() {
    if (solvingPattern) return;
    solvingPattern = true;

    clearSteps('pattern-steps');

    // Configure TRM
    window.TRM.config.H_cycles = 4;
    window.TRM.config.L_cycles = 3;

    // Track prediction progress
    let partialOutput = currentPattern.output.map(row =>
        row.map(() => 0)
    );

    // Set up step listener
    window.TRM.listeners = [];
    window.TRM.onStep(step => {
        displayStep('pattern-steps', step);

        if (step.type === 'H_cycle') {
            // Progressively reveal output
            const progress = step.h_step / window.TRM.config.H_cycles;
            revealOutput(progress);
        }
    });

    // Solve
    await window.TRM.solve(currentPattern.input, (latent, hStep) => {
        return currentPattern.transform(currentPattern.input);
    });

    // Show final output
    renderPatternGrid('pattern-output', currentPattern.output);

    // Show result
    const stepsContainer = document.getElementById('pattern-steps');
    const resultEl = document.createElement('div');
    resultEl.style.cssText = 'margin-top: 1rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border-left: 3px solid #10b981;';
    resultEl.innerHTML = `
        <div style="font-weight: 600; color: #10b981;">Pattern Recognized!</div>
        <div style="font-size: 0.875rem; color: #9ca3af; margin-top: 0.25rem;">Transform: ${currentPattern.name}</div>
    `;
    stepsContainer.appendChild(resultEl);

    solvingPattern = false;
}

/**
 * Progressively reveal output during solving
 */
function revealOutput(progress) {
    const output = currentPattern.output;
    const numRows = output.length;
    const numCols = output[0].length;
    const totalCells = numRows * numCols;
    const cellsToReveal = Math.floor(progress * totalCells);

    const partialOutput = output.map(row => row.map(() => 0));

    // Reveal cells in order
    let revealed = 0;
    for (let i = 0; i < numRows && revealed < cellsToReveal; i++) {
        for (let j = 0; j < numCols && revealed < cellsToReveal; j++) {
            partialOutput[i][j] = output[i][j];
            revealed++;
        }
    }

    renderPatternGrid('pattern-output', partialOutput);
}

/**
 * Clear steps container
 */
function clearSteps(containerId) {
    const container = document.getElementById(containerId);
    if (container) container.innerHTML = '';
}
