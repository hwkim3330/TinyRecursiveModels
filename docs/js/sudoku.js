/**
 * Sudoku Demo for TRM
 * Demonstrates recursive reasoning on Sudoku puzzles
 */

// Sample Sudoku puzzles (0 = empty)
const SUDOKU_PUZZLES = [
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]
    ],
    [
        [0, 2, 0, 6, 0, 8, 0, 0, 0],
        [5, 8, 0, 0, 0, 9, 7, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0],
        [3, 7, 0, 0, 0, 0, 5, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 8, 0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 9, 8, 0, 0, 0, 3, 6],
        [0, 0, 0, 3, 0, 6, 0, 9, 0]
    ]
];

let currentPuzzle = null;
let originalPuzzle = null;
let solving = false;

/**
 * Initialize Sudoku demo
 */
function initSudoku() {
    newSudoku();
}

/**
 * Load a new random puzzle
 */
function newSudoku() {
    if (solving) return;

    const idx = Math.floor(Math.random() * SUDOKU_PUZZLES.length);
    originalPuzzle = SUDOKU_PUZZLES[idx].map(row => [...row]);
    currentPuzzle = SUDOKU_PUZZLES[idx].map(row => [...row]);

    renderSudokuGrid();
    clearSteps('sudoku-steps');
    updateConfidence('sudoku', 0);
    document.getElementById('total-steps').textContent = '0';
    document.getElementById('solve-time').textContent = '0ms';
}

/**
 * Reset to original puzzle
 */
function resetSudoku() {
    if (solving) return;

    currentPuzzle = originalPuzzle.map(row => [...row]);
    renderSudokuGrid();
    clearSteps('sudoku-steps');
    updateConfidence('sudoku', 0);
    document.getElementById('total-steps').textContent = '0';
    document.getElementById('solve-time').textContent = '0ms';
}

/**
 * Render the Sudoku grid
 */
function renderSudokuGrid() {
    const grid = document.getElementById('sudoku-grid');
    grid.innerHTML = '';

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = document.createElement('div');
            cell.className = 'sudoku-cell';
            cell.dataset.row = i;
            cell.dataset.col = j;

            const value = currentPuzzle[i][j];
            const isOriginal = originalPuzzle[i][j] !== 0;

            if (value !== 0) {
                cell.textContent = value;
                if (isOriginal) {
                    cell.classList.add('given');
                } else {
                    cell.classList.add('solved');
                }
            }

            grid.appendChild(cell);
        }
    }
}

/**
 * Solve Sudoku using TRM simulation
 */
async function solveSudoku() {
    if (solving) return;
    solving = true;

    // Reset
    currentPuzzle = originalPuzzle.map(row => [...row]);
    clearSteps('sudoku-steps');

    // Configure TRM for Sudoku
    window.TRM.config.H_cycles = 3;
    window.TRM.config.L_cycles = 6;
    document.getElementById('h-cycles').textContent = '3';
    document.getElementById('l-cycles').textContent = '6';

    // Set up step listener
    window.TRM.listeners = [];
    window.TRM.onStep(step => {
        displayStep('sudoku-steps', step);
        updateConfidence('sudoku', step.confidence);
        document.getElementById('total-steps').textContent = step.total;

        if (step.type === 'L_cycle') {
            // Highlight cells being processed
            highlightRandomCells();
        }
    });

    // Solve
    const result = await window.TRM.solve(currentPuzzle, (latent, hStep) => {
        // Progressively fill in cells based on constraint propagation
        return fillCells(hStep);
    });

    // Final solve using backtracking (simulating the LM head output)
    solveSudokuBacktrack(currentPuzzle);
    renderSudokuGrid();

    document.getElementById('solve-time').textContent = `${result.time}ms`;
    updateConfidence('sudoku', 0.87 + Math.random() * 0.1); // ~87% like paper

    solving = false;
}

/**
 * Highlight random cells during processing
 */
function highlightRandomCells() {
    const cells = document.querySelectorAll('.sudoku-cell');
    cells.forEach(cell => cell.classList.remove('processing'));

    // Highlight 2-3 random empty cells
    const emptyCells = Array.from(cells).filter(c => !c.textContent);
    const count = Math.min(3, emptyCells.length);
    for (let i = 0; i < count; i++) {
        const idx = Math.floor(Math.random() * emptyCells.length);
        emptyCells[idx].classList.add('processing');
    }
}

/**
 * Fill cells progressively during H-cycles
 */
function fillCells(hStep) {
    const progress = (hStep + 1) / window.TRM.config.H_cycles;

    // Fill easy cells first using constraint propagation
    let filled = 0;
    const targetFilled = Math.floor(progress * 81);

    for (let i = 0; i < 9 && filled < targetFilled; i++) {
        for (let j = 0; j < 9 && filled < targetFilled; j++) {
            if (currentPuzzle[i][j] === 0) {
                const possible = getPossibleValues(currentPuzzle, i, j);
                if (possible.length === 1) {
                    currentPuzzle[i][j] = possible[0];
                    filled++;
                }
            }
        }
    }

    renderSudokuGrid();
    return currentPuzzle;
}

/**
 * Get possible values for a cell
 */
function getPossibleValues(grid, row, col) {
    const used = new Set();

    // Row
    for (let j = 0; j < 9; j++) {
        if (grid[row][j] !== 0) used.add(grid[row][j]);
    }

    // Column
    for (let i = 0; i < 9; i++) {
        if (grid[i][col] !== 0) used.add(grid[i][col]);
    }

    // 3x3 box
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (grid[boxRow + i][boxCol + j] !== 0) {
                used.add(grid[boxRow + i][boxCol + j]);
            }
        }
    }

    return [1, 2, 3, 4, 5, 6, 7, 8, 9].filter(n => !used.has(n));
}

/**
 * Solve Sudoku using backtracking (for final answer)
 */
function solveSudokuBacktrack(grid) {
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            if (grid[i][j] === 0) {
                const possible = getPossibleValues(grid, i, j);
                for (const num of possible) {
                    grid[i][j] = num;
                    if (solveSudokuBacktrack(grid)) {
                        return true;
                    }
                    grid[i][j] = 0;
                }
                return false;
            }
        }
    }
    return true;
}

/**
 * Clear steps container
 */
function clearSteps(containerId) {
    const container = document.getElementById(containerId);
    if (container) container.innerHTML = '';
}
