/* tslint:disable */
/* eslint-disable */

export class StepInfo {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  get_z_h_activations(): Float32Array;
  get_z_l_activations(): Float32Array;
  get_attention_weights(): Float32Array;
  h_step: number;
  l_step: number;
  total_steps: number;
  confidence: number;
}

export class TRM {
  free(): void;
  [Symbol.dispose](): void;
  constructor(config: TRMConfig);
  /**
   * Reset the model state
   */
  reset(): void;
  /**
   * Run one complete reasoning step and return step info
   */
  step(input: Uint8Array): StepInfo;
  /**
   * Run full inference
   */
  forward(input: Uint8Array): Uint8Array;
  /**
   * Get current output prediction
   */
  get_current_output(): Uint8Array;
  /**
   * Check if inference is complete
   */
  is_complete(): boolean;
  /**
   * Get z_H as flat array for visualization
   */
  get_z_h(): Float32Array;
  /**
   * Get z_L as flat array for visualization
   */
  get_z_l(): Float32Array;
}

export class TRMConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  static for_sudoku(): TRMConfig;
  static for_maze(): TRMConfig;
  hidden_size: number;
  num_heads: number;
  expansion: number;
  h_cycles: number;
  l_cycles: number;
  l_layers: number;
  vocab_size: number;
  seq_len: number;
  use_mlp_t: boolean;
}

/**
 * Compute attention-style heatmap from two activation vectors
 */
export function compute_attention_heatmap(query: Float32Array, key: Float32Array, size: number): Float32Array;

/**
 * Utility to create a Maze input tensor
 */
export function create_maze_input(grid: Uint8Array, width: number, height: number): Uint8Array;

/**
 * Utility to create a Sudoku input tensor
 */
export function create_sudoku_input(grid: Uint8Array): Uint8Array;

/**
 * Decode Sudoku output to readable format
 */
export function decode_sudoku_output(output: Uint8Array): string;

/**
 * Initialize WASM module
 */
export function init(): void;

/**
 * Visualization helper: normalize activations to 0-255 range
 */
export function normalize_activations(activations: Float32Array): Uint8Array;

/**
 * Check if Sudoku solution is valid
 */
export function validate_sudoku(grid: Uint8Array): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_trmconfig_free: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_hidden_size: (a: number) => number;
  readonly __wbg_set_trmconfig_hidden_size: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_num_heads: (a: number) => number;
  readonly __wbg_set_trmconfig_num_heads: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_expansion: (a: number) => number;
  readonly __wbg_set_trmconfig_expansion: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_h_cycles: (a: number) => number;
  readonly __wbg_set_trmconfig_h_cycles: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_l_cycles: (a: number) => number;
  readonly __wbg_set_trmconfig_l_cycles: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_l_layers: (a: number) => number;
  readonly __wbg_set_trmconfig_l_layers: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_vocab_size: (a: number) => number;
  readonly __wbg_set_trmconfig_vocab_size: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_seq_len: (a: number) => number;
  readonly __wbg_set_trmconfig_seq_len: (a: number, b: number) => void;
  readonly __wbg_get_trmconfig_use_mlp_t: (a: number) => number;
  readonly __wbg_set_trmconfig_use_mlp_t: (a: number, b: number) => void;
  readonly trmconfig_new: () => number;
  readonly trmconfig_for_sudoku: () => number;
  readonly trmconfig_for_maze: () => number;
  readonly __wbg_stepinfo_free: (a: number, b: number) => void;
  readonly __wbg_get_stepinfo_h_step: (a: number) => number;
  readonly __wbg_set_stepinfo_h_step: (a: number, b: number) => void;
  readonly __wbg_get_stepinfo_l_step: (a: number) => number;
  readonly __wbg_set_stepinfo_l_step: (a: number, b: number) => void;
  readonly __wbg_get_stepinfo_total_steps: (a: number) => number;
  readonly __wbg_set_stepinfo_total_steps: (a: number, b: number) => void;
  readonly __wbg_get_stepinfo_confidence: (a: number) => number;
  readonly __wbg_set_stepinfo_confidence: (a: number, b: number) => void;
  readonly stepinfo_get_z_h_activations: (a: number) => [number, number];
  readonly stepinfo_get_z_l_activations: (a: number) => [number, number];
  readonly stepinfo_get_attention_weights: (a: number) => [number, number];
  readonly __wbg_trm_free: (a: number, b: number) => void;
  readonly trm_new: (a: number) => number;
  readonly trm_reset: (a: number) => void;
  readonly trm_step: (a: number, b: number, c: number) => number;
  readonly trm_forward: (a: number, b: number, c: number) => [number, number];
  readonly trm_get_current_output: (a: number) => [number, number];
  readonly trm_is_complete: (a: number) => number;
  readonly trm_get_z_h: (a: number) => [number, number];
  readonly trm_get_z_l: (a: number) => [number, number];
  readonly init: () => void;
  readonly create_sudoku_input: (a: number, b: number) => [number, number];
  readonly create_maze_input: (a: number, b: number, c: number, d: number) => [number, number];
  readonly normalize_activations: (a: number, b: number) => [number, number];
  readonly compute_attention_heatmap: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly decode_sudoku_output: (a: number, b: number) => [number, number];
  readonly validate_sudoku: (a: number, b: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
