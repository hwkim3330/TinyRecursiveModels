/**
 * Maze Demo for TRM
 * Demonstrates recursive reasoning on maze pathfinding
 */

let currentMaze = null;
let mazeSize = 15;
let startPos = null;
let endPos = null;
let solvingMaze = false;

/**
 * Initialize Maze demo
 */
function initMaze() {
    newMaze();
}

/**
 * Generate a new random maze using recursive backtracking
 */
function newMaze() {
    if (solvingMaze) return;

    const size = mazeSize;
    currentMaze = Array(size).fill(null).map(() => Array(size).fill(1)); // 1 = wall

    // Generate maze using recursive backtracking
    generateMaze(1, 1);

    // Set start and end
    startPos = { x: 1, y: 1 };
    endPos = { x: size - 2, y: size - 2 };
    currentMaze[startPos.y][startPos.x] = 2; // 2 = start
    currentMaze[endPos.y][endPos.x] = 3; // 3 = end

    renderMazeGrid();
    clearSteps('maze-steps');
    updateConfidence('maze', 0);
    document.getElementById('maze-confidence').textContent = '-';
}

/**
 * Generate maze using recursive backtracking
 */
function generateMaze(x, y) {
    currentMaze[y][x] = 0; // 0 = path

    const directions = [
        [0, -2], [0, 2], [-2, 0], [2, 0]
    ].sort(() => Math.random() - 0.5);

    for (const [dx, dy] of directions) {
        const nx = x + dx;
        const ny = y + dy;

        if (nx > 0 && nx < mazeSize - 1 && ny > 0 && ny < mazeSize - 1 && currentMaze[ny][nx] === 1) {
            currentMaze[y + dy / 2][x + dx / 2] = 0; // Remove wall between
            generateMaze(nx, ny);
        }
    }
}

/**
 * Reset maze (clear solution)
 */
function resetMaze() {
    if (solvingMaze) return;

    // Clear solution markers
    for (let y = 0; y < mazeSize; y++) {
        for (let x = 0; x < mazeSize; x++) {
            if (currentMaze[y][x] === 4) { // 4 = solution path
                currentMaze[y][x] = 0;
            }
        }
    }

    renderMazeGrid();
    clearSteps('maze-steps');
    updateConfidence('maze', 0);
    document.getElementById('maze-confidence').textContent = '-';
}

/**
 * Render the maze grid
 */
function renderMazeGrid() {
    const grid = document.getElementById('maze-grid');
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${mazeSize}, 16px)`;

    for (let y = 0; y < mazeSize; y++) {
        for (let x = 0; x < mazeSize; x++) {
            const cell = document.createElement('div');
            cell.className = 'maze-cell';

            switch (currentMaze[y][x]) {
                case 0: cell.classList.add('path'); break;
                case 1: cell.classList.add('wall'); break;
                case 2: cell.classList.add('start'); break;
                case 3: cell.classList.add('end'); break;
                case 4: cell.classList.add('solution'); break;
            }

            grid.appendChild(cell);
        }
    }
}

/**
 * Solve maze using TRM simulation
 */
async function solveMaze() {
    if (solvingMaze) return;
    solvingMaze = true;

    resetMaze();

    // Configure TRM for Maze
    window.TRM.config.H_cycles = 3;
    window.TRM.config.L_cycles = 4;

    // Track explored cells for visualization
    let explored = new Set();
    let path = [];

    // Set up step listener
    window.TRM.listeners = [];
    window.TRM.onStep(async step => {
        displayStep('maze-steps', step);
        updateConfidence('maze', step.confidence);

        if (step.type === 'L_cycle') {
            // Explore new cells
            await exploreStep(explored);
        }
    });

    // Solve
    const startTime = performance.now();

    await window.TRM.solve(currentMaze, (latent, hStep) => {
        // Each H-cycle, try to extend the path
        return extendPath(hStep);
    });

    // Final BFS solve
    path = solveMazeBFS();

    // Mark solution path
    for (const pos of path) {
        if (currentMaze[pos.y][pos.x] === 0) {
            currentMaze[pos.y][pos.x] = 4;
        }
    }

    renderMazeGrid();

    const endTime = performance.now();
    const found = path.length > 0;

    document.getElementById('maze-confidence').textContent = found ? 'Yes' : 'No';
    updateConfidence('maze', found ? 0.95 : 0);

    solvingMaze = false;
}

/**
 * Visualize exploration step
 */
async function exploreStep(explored) {
    const cells = document.querySelectorAll('.maze-cell');

    // Find unvisited path cells adjacent to explored cells
    const candidates = [];
    for (let y = 1; y < mazeSize - 1; y++) {
        for (let x = 1; x < mazeSize - 1; x++) {
            if (currentMaze[y][x] === 0 && !explored.has(`${x},${y}`)) {
                // Check if adjacent to explored
                const neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]];
                for (const [dx, dy] of neighbors) {
                    if (explored.has(`${x + dx},${y + dy}`) || (x + dx === startPos.x && y + dy === startPos.y)) {
                        candidates.push({ x, y });
                        break;
                    }
                }
            }
        }
    }

    // Explore a random candidate
    if (candidates.length > 0) {
        const cell = candidates[Math.floor(Math.random() * candidates.length)];
        explored.add(`${cell.x},${cell.y}`);

        // Highlight
        const idx = cell.y * mazeSize + cell.x;
        if (cells[idx]) {
            cells[idx].classList.add('exploring');
            setTimeout(() => cells[idx].classList.remove('exploring'), 200);
        }
    }
}

/**
 * Extend path during H-cycles
 */
function extendPath(hStep) {
    // Visualize progressive exploration
    return currentMaze;
}

/**
 * Solve maze using BFS (for final answer)
 */
function solveMazeBFS() {
    const queue = [{ ...startPos, path: [startPos] }];
    const visited = new Set([`${startPos.x},${startPos.y}`]);

    while (queue.length > 0) {
        const { x, y, path } = queue.shift();

        if (x === endPos.x && y === endPos.y) {
            return path;
        }

        const neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        for (const [dx, dy] of neighbors) {
            const nx = x + dx;
            const ny = y + dy;
            const key = `${nx},${ny}`;

            if (nx >= 0 && nx < mazeSize && ny >= 0 && ny < mazeSize &&
                !visited.has(key) && currentMaze[ny][nx] !== 1) {
                visited.add(key);
                queue.push({ x: nx, y: ny, path: [...path, { x: nx, y: ny }] });
            }
        }
    }

    return []; // No path found
}
