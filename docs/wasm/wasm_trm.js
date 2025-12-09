let wasm;

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

const StepInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_stepinfo_free(ptr >>> 0, 1));

const TRMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trm_free(ptr >>> 0, 1));

const TRMConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trmconfig_free(ptr >>> 0, 1));

/**
 * Step information for visualization
 */
export class StepInfo {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(StepInfo.prototype);
        obj.__wbg_ptr = ptr;
        StepInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StepInfoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_stepinfo_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get h_step() {
        const ret = wasm.__wbg_get_stepinfo_h_step(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set h_step(arg0) {
        wasm.__wbg_set_stepinfo_h_step(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get l_step() {
        const ret = wasm.__wbg_get_stepinfo_l_step(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set l_step(arg0) {
        wasm.__wbg_set_stepinfo_l_step(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get total_steps() {
        const ret = wasm.__wbg_get_stepinfo_total_steps(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set total_steps(arg0) {
        wasm.__wbg_set_stepinfo_total_steps(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get confidence() {
        const ret = wasm.__wbg_get_stepinfo_confidence(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set confidence(arg0) {
        wasm.__wbg_set_stepinfo_confidence(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {Float32Array}
     */
    get_z_h_activations() {
        const ret = wasm.stepinfo_get_z_h_activations(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    get_z_l_activations() {
        const ret = wasm.stepinfo_get_z_l_activations(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    get_attention_weights() {
        const ret = wasm.stepinfo_get_attention_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) StepInfo.prototype[Symbol.dispose] = StepInfo.prototype.free;

/**
 * Main TRM Model
 */
export class TRM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TRMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trm_free(ptr, 0);
    }
    /**
     * @param {TRMConfig} config
     */
    constructor(config) {
        _assertClass(config, TRMConfig);
        var ptr0 = config.__destroy_into_raw();
        const ret = wasm.trm_new(ptr0);
        this.__wbg_ptr = ret >>> 0;
        TRMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reset the model state
     */
    reset() {
        wasm.trm_reset(this.__wbg_ptr);
    }
    /**
     * Run one complete reasoning step and return step info
     * @param {Uint8Array} input
     * @returns {StepInfo}
     */
    step(input) {
        const ptr0 = passArray8ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trm_step(this.__wbg_ptr, ptr0, len0);
        return StepInfo.__wrap(ret);
    }
    /**
     * Run full inference
     * @param {Uint8Array} input
     * @returns {Uint8Array}
     */
    forward(input) {
        const ptr0 = passArray8ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trm_forward(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v2;
    }
    /**
     * Get current output prediction
     * @returns {Uint8Array}
     */
    get_current_output() {
        const ret = wasm.trm_get_current_output(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Check if inference is complete
     * @returns {boolean}
     */
    is_complete() {
        const ret = wasm.trm_is_complete(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get z_H as flat array for visualization
     * @returns {Float32Array}
     */
    get_z_h() {
        const ret = wasm.trm_get_z_h(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get z_L as flat array for visualization
     * @returns {Float32Array}
     */
    get_z_l() {
        const ret = wasm.trm_get_z_l(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) TRM.prototype[Symbol.dispose] = TRM.prototype.free;

/**
 * TRM Configuration
 */
export class TRMConfig {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(TRMConfig.prototype);
        obj.__wbg_ptr = ptr;
        TRMConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TRMConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trmconfig_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get hidden_size() {
        const ret = wasm.__wbg_get_trmconfig_hidden_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set hidden_size(arg0) {
        wasm.__wbg_set_trmconfig_hidden_size(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get num_heads() {
        const ret = wasm.__wbg_get_trmconfig_num_heads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set num_heads(arg0) {
        wasm.__wbg_set_trmconfig_num_heads(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get expansion() {
        const ret = wasm.__wbg_get_trmconfig_expansion(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set expansion(arg0) {
        wasm.__wbg_set_trmconfig_expansion(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get h_cycles() {
        const ret = wasm.__wbg_get_trmconfig_h_cycles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set h_cycles(arg0) {
        wasm.__wbg_set_trmconfig_h_cycles(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get l_cycles() {
        const ret = wasm.__wbg_get_trmconfig_l_cycles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set l_cycles(arg0) {
        wasm.__wbg_set_trmconfig_l_cycles(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get l_layers() {
        const ret = wasm.__wbg_get_trmconfig_l_layers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set l_layers(arg0) {
        wasm.__wbg_set_trmconfig_l_layers(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get vocab_size() {
        const ret = wasm.__wbg_get_trmconfig_vocab_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set vocab_size(arg0) {
        wasm.__wbg_set_trmconfig_vocab_size(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get seq_len() {
        const ret = wasm.__wbg_get_trmconfig_seq_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set seq_len(arg0) {
        wasm.__wbg_set_trmconfig_seq_len(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get use_mlp_t() {
        const ret = wasm.__wbg_get_trmconfig_use_mlp_t(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set use_mlp_t(arg0) {
        wasm.__wbg_set_trmconfig_use_mlp_t(this.__wbg_ptr, arg0);
    }
    constructor() {
        const ret = wasm.trmconfig_new();
        this.__wbg_ptr = ret >>> 0;
        TRMConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {TRMConfig}
     */
    static for_sudoku() {
        const ret = wasm.trmconfig_for_sudoku();
        return TRMConfig.__wrap(ret);
    }
    /**
     * @returns {TRMConfig}
     */
    static for_maze() {
        const ret = wasm.trmconfig_for_maze();
        return TRMConfig.__wrap(ret);
    }
}
if (Symbol.dispose) TRMConfig.prototype[Symbol.dispose] = TRMConfig.prototype.free;

/**
 * Compute attention-style heatmap from two activation vectors
 * @param {Float32Array} query
 * @param {Float32Array} key
 * @param {number} size
 * @returns {Float32Array}
 */
export function compute_attention_heatmap(query, key, size) {
    const ptr0 = passArrayF32ToWasm0(query, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(key, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.compute_attention_heatmap(ptr0, len0, ptr1, len1, size);
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * Utility to create a Maze input tensor
 * @param {Uint8Array} grid
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array}
 */
export function create_maze_input(grid, width, height) {
    const ptr0 = passArray8ToWasm0(grid, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.create_maze_input(ptr0, len0, width, height);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Utility to create a Sudoku input tensor
 * @param {Uint8Array} grid
 * @returns {Uint8Array}
 */
export function create_sudoku_input(grid) {
    const ptr0 = passArray8ToWasm0(grid, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.create_sudoku_input(ptr0, len0);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Decode Sudoku output to readable format
 * @param {Uint8Array} output
 * @returns {string}
 */
export function decode_sudoku_output(output) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passArray8ToWasm0(output, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decode_sudoku_output(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Initialize WASM module
 */
export function init() {
    wasm.init();
}

/**
 * Visualization helper: normalize activations to 0-255 range
 * @param {Float32Array} activations
 * @returns {Uint8Array}
 */
export function normalize_activations(activations) {
    const ptr0 = passArrayF32ToWasm0(activations, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.normalize_activations(ptr0, len0);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Check if Sudoku solution is valid
 * @param {Uint8Array} grid
 * @returns {boolean}
 */
export function validate_sudoku(grid) {
    const ptr0 = passArray8ToWasm0(grid, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.validate_sudoku(ptr0, len0);
    return ret !== 0;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_is_function_8d400b8b1af978cd = function(arg0) {
        const ret = typeof(arg0) === 'function';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_object_ce774f3490692386 = function(arg0) {
        const val = arg0;
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_string_704ef9c8fc131030 = function(arg0) {
        const ret = typeof(arg0) === 'string';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_undefined_f6b95eab589e0269 = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_call_3020136f7a2d6e44 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.call(arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_call_abb4ff46ce38be40 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_crypto_574e78ad8b13b65f = function(arg0) {
        const ret = arg0.crypto;
        return ret;
    };
    imports.wbg.__wbg_getRandomValues_b8f5dbd5f3995a9e = function() { return handleError(function (arg0, arg1) {
        arg0.getRandomValues(arg1);
    }, arguments) };
    imports.wbg.__wbg_length_22ac23eaec9d8053 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_log_40b15fe39549f271 = function(arg0, arg1) {
        console.log(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_msCrypto_a61aeb35a24c1329 = function(arg0) {
        const ret = arg0.msCrypto;
        return ret;
    };
    imports.wbg.__wbg_new_no_args_cb138f77cf6151ee = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_new_with_length_aa5eaf41d35235e5 = function(arg0) {
        const ret = new Uint8Array(arg0 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_node_905d3e251edff8a2 = function(arg0) {
        const ret = arg0.node;
        return ret;
    };
    imports.wbg.__wbg_process_dc0fbacc7c1c06f7 = function(arg0) {
        const ret = arg0.process;
        return ret;
    };
    imports.wbg.__wbg_prototypesetcall_dfe9b766cdc1f1fd = function(arg0, arg1, arg2) {
        Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
    };
    imports.wbg.__wbg_randomFillSync_ac0988aba3254290 = function() { return handleError(function (arg0, arg1) {
        arg0.randomFillSync(arg1);
    }, arguments) };
    imports.wbg.__wbg_require_60cc747a6bc5215a = function() { return handleError(function () {
        const ret = module.require;
        return ret;
    }, arguments) };
    imports.wbg.__wbg_static_accessor_GLOBAL_769e6b65d6557335 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_60cf02db4de8e1c1 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_08f5a74c69739274 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_a8924b26aa92d024 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_subarray_845f2f5bce7d061a = function(arg0, arg1, arg2) {
        const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_versions_c01dfd4722a88165 = function(arg0) {
        const ret = arg0.versions;
        return ret;
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_cast_cb9088102bce6b30 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
        const ret = getArrayU8FromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('wasm_trm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
