# gpu_lz4_compressor.py
# Implementação LZ4 Extendida (3-byte offsets) Acelerada por GPU (OpenCL)

import pyopencl as cl
import numpy as np
import lz4.frame
import time
from typing import Tuple, Optional
import contextlib

# Carregar configurações do config.txt centralizado
import config_loader

# Constantes LZ4 (carregadas de config.txt)
HASH_LOG = config_loader.get_hash_log()
HASH_CANDIDATES = config_loader.get_hash_candidates()
GOOD_ENOUGH_MATCH = config_loader.get_good_enough_match()
HASH_ENTRIES = (1 << HASH_LOG)

HASH_TABLE_SIZE = HASH_ENTRIES * HASH_CANDIDATES

def gen_opencl_vector_load(k, ptr, base_offset, dest_arr):
    code = [f"// Vectorized Load for K={k} (Generated Python)"]
    offset = 0
    remaining = k
    sizes = [16, 8, 4, 2, 1]
    
    for s in sizes:
        while remaining >= s:
            vec_offset_elem = offset // s
            if s == 1:
                code.append(f"{dest_arr}[{offset}] = {ptr}[{base_offset} + {offset}];")
            else:
                var_name = f"v_load_{offset}"
                code.append(f"uint{s} {var_name} = vload{s}({vec_offset_elem}, {ptr} + {base_offset});")
                # Unpack
                opts = "0123456789ABCDEF"
                for i in range(s):
                    code.append(f"{dest_arr}[{offset + i}] = {var_name}.s{opts[i]};")
            
            remaining -= s
            offset += s
    return "\n    ".join(code)

def gen_opencl_vector_store(k, ptr, base_offset, values):
    code = [f"// Vectorized Store for K={k} (Generated Python)"]
    offset = 0
    remaining = k
    sizes = [16, 8, 4, 2, 1]
    val_idx = 0
    
    for s in sizes:
        while remaining >= s:
            vec_offset_elem = offset // s
            if s == 1:
                code.append(f"{ptr}[{base_offset} + {offset}] = {values[val_idx]};")
                val_idx += 1
            else:
                var_name = f"v_store_{offset}"
                code.append(f"uint{s} {var_name};")
                for i in range(s):
                    code.append(f"{var_name}.s{'0123456789ABCDEF'[i]} = {values[val_idx]};")
                    val_idx += 1
                code.append(f"vstore{s}({var_name}, {vec_offset_elem}, {ptr} + {base_offset});")
            
            remaining -= s
            offset += s
    return "\n    ".join(code)

# Template do kernel LZ4 OpenCL - usa .format() para substituir valores de config
_LZ4_KERNEL_TEMPLATE = """
// lz4_compress_ext3.cl
// Implementação LZ4 Extendida com offsets de 3 bytes para OpenCL
// Baseado no princípio LZ77, adaptado para um único work-item (gid=0).
// MAX_DISTANCE: 16MB (vs 64KB do LZ4 padrão)

// Constantes LZ4
#define MIN_MATCH 4

// Tabela de hash (valores configuráveis via config.txt):
// - HASH_LOG define o número de entradas base: 2^HASH_LOG
// - HASH_CANDIDATES define quantas posições cada entrada guarda
#define HASH_LOG        {hash_log}u
#define HASH_ENTRIES    (1u << HASH_LOG)
#define HASH_CANDIDATES {hash_candidates}u
#define HASH_TABLE_SIZE (HASH_ENTRIES * HASH_CANDIDATES)

// Hash LZ4 padrão
#define PRIME 2654435761U // 0x9E3779B1

// Janela Extendida (offset codificado em 3 bytes)
#define MAX_DISTANCE (1u << 24) // 16777216 bytes (~16 MB)

// Lazy parsing: só trocamos para match em ip+1 se ele for bem melhor
#define LAZY_DELTA 2u  // exige que next_match_len >= current_match_len + LAZY_DELTA

// Debug Flags
#define DBG_START          1u
#define DBG_LOOP_START     2u
#define DBG_MATCH_FOUND    3u
#define DBG_NO_MATCH       4u
#define DBG_LOOP_END       5u
#define DBG_FINALIZATION   6u
#define DBG_SUCCESS        7u
#define DBG_ERROR_OP_LIMIT 8u

// Função de hash macro (4 bytes)
#define HASH_FUNC(pos) ((input_data[pos] | (input_data[pos+1] << 8) | \\
                         (input_data[pos+2] << 16) | (input_data[pos+3] << 24)) \\
                        * PRIME) >> (32 - HASH_LOG)

// Função de hash de 4 bytes do LZ4
inline uint LZ4_hash4(const __global uchar* ptr) {{
    uint value = (uint)ptr[0] |
                 ((uint)ptr[1] << 8) |
                 ((uint)ptr[2] << 16) |
                 ((uint)ptr[3] << 24);
    return (value * PRIME) >> (32 - HASH_LOG);
}}

// Função para escrever o comprimento estendido
inline __global uchar* write_length(__global uchar* op, uint length, __global uchar* op_limit) {{
    // Fast path: maioria dos casos (len < 255)
    if (length < 255) {{
        if (op < op_limit) *op++ = (uchar)length;
        return op;
    }}

    while (length >= 255) {{
        if (op >= op_limit) return op;
        *op++ = 255;
        length -= 255;
    }}
    if (op < op_limit) *op++ = (uchar)length;
    return op;
}}

// ---------------------------------------------------------------------------
// FIND_BEST_MATCH:
//  - Procura o melhor match para 'ip' / current_pos
//  - Usa HASH_CANDIDATES posições por hash
//  - Atualiza a tabela (insere current_pos na "cabeça" da lista)
//  - Retorna best_match_pos / best_match_len via ponteiros
// ---------------------------------------------------------------------------
// Helper para carregar 4 bytes de forma segura (evita unaligned access crash)
inline uint load4_safe(__global const uchar* p) {{
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}}

// Helper para carregar 8 bytes de forma segura
inline ulong load8_safe(__global const uchar* p) {{
    return (ulong)p[0] | ((ulong)p[1] << 8) | ((ulong)p[2] << 16) | ((ulong)p[3] << 24) |
           ((ulong)p[4] << 32) | ((ulong)p[5] << 40) | ((ulong)p[6] << 48) | ((ulong)p[7] << 56);
}}

// ============================================================
// OTIMIZAÇÃO: CACHE DE HASH EM MEMÓRIA PRIVADA
// ============================================================
// Como cada work-item processa um frame inteiro, usamos memória
// privada (registradores) para cachear entradas de hash recentes.
// Isso reduz acessos à memória global para dados com alta localidade.
// ============================================================
#define HASH_CACHE_SIZE 32u
#define HASH_CACHE_MASK (HASH_CACHE_SIZE - 1)

// Estrutura do cache: armazena (hash, posição do primeiro candidato)
// Isso evita re-leitura da memória global para hashes repetidos

inline void find_best_match(
    __global const uchar* input_data,
    uint input_size,
    __global const uchar* ip,
    uint current_pos,
    uint last_literal_pos,
    __global uint* hash_table,
    uint* best_match_pos,
    uint* best_match_len,
    uint hash_value
) {{
    uint base_index = hash_value * HASH_CANDIDATES;

    uint local_best_pos = 0;
    uint local_best_len = 0;
    
    // Early exit threshold: {good_enough_match} bytes = "good enough" match (configurável via config.txt)
    #define GOOD_ENOUGH_MATCH {good_enough_match}u

    // ============================================================
    // OTIMIZAÇÃO: Pré-carregar candidatos em memória privada
    // ============================================================
    // Carregar todos os candidatos de uma vez reduz latência de memória
    // USO EFICIENTE DE LOAD VETORIZADO PARA HASH_CANDIDATES={hash_candidates}
    uint candidates[HASH_CANDIDATES];
    {load_candidates_code}

    // Top-K Search: Iterate through all candidates (from most recent to oldest)
    for (uint i = 0; i < HASH_CANDIDATES; i++) {{
        uint candidate_pos = candidates[i];
        
        // Early termination: empty slot means no more candidates
        if (candidate_pos == 0) break;
        
        uint distance = current_pos - candidate_pos;
        
        // Skip if invalid distance or out of bounds
        if (distance == 0 || distance >= MAX_DISTANCE || candidate_pos + MIN_MATCH > input_size) {{
            continue;
        }}
        
        // Quick 4-byte prefix check (using private copy avoids re-fetch)
        if (ip[0] != input_data[candidate_pos] ||
            ip[1] != input_data[candidate_pos + 1] ||
            ip[2] != input_data[candidate_pos + 2] ||
            ip[3] != input_data[candidate_pos + 3]) {{
            continue;  // Prefix mismatch, try next candidate
        }}

        // Extend match (optimized: 8-byte and 4-byte chunks)
        uint match_len = MIN_MATCH;
        __global const uchar* match_ip  = ip + MIN_MATCH;
        __global const uchar* match_ref = input_data + candidate_pos + MIN_MATCH;

        uint max_match_len = last_literal_pos - current_pos;

        // Fast path: compare 8 bytes at a time (SAFE UNALIGNED LOAD)
        while (match_len + 8 <= max_match_len) {{
            ulong val1 = load8_safe(match_ip);
            ulong val2 = load8_safe(match_ref);
            if (val1 != val2) break;
            
            match_len += 8;
            match_ip += 8;
            match_ref += 8;
        }}

        // Fast path: compare 4 bytes at a time (SAFE UNALIGNED LOAD)
        while (match_len + 4 <= max_match_len) {{
            uint val1 = load4_safe(match_ip);
            uint val2 = load4_safe(match_ref);
            if (val1 != val2) break;
            
            match_len += 4;
            match_ip += 4;
            match_ref += 4;
        }}

        // Slow path: byte by byte
        while (match_len < max_match_len && *match_ip == *match_ref) {{
            match_len++;
            match_ip++;
            match_ref++;
        }}

        // Update best if this match is better
        if (match_len > local_best_len) {{
            local_best_len = match_len;
            local_best_pos = candidate_pos;
            
            // Early exit: found "good enough" match
            if (match_len >= GOOD_ENOUGH_MATCH) {{
                break;  // Stop searching, this match is good enough
            }}
        }}
    }}

    // ============================================================
    // Update hash table: shift right and insert current_pos at position 0
    // OTIMIZAÇÃO: Usar a cópia privada para determinar valores a escrever com STORE VETORIZADO
    // ============================================================
    // A lista a escrever é [current_pos, candidates[0], ... candidates[K-2]]
    {update_hash_table_code}

    *best_match_pos = local_best_pos;
    *best_match_len = local_best_len;
}}

__kernel void lz4_compress_block(
    __global const uchar* input_buffer,      // Buffer único contendo todos os frames concatenados
    __global uchar* output_buffer,           // Buffer único para saídas
    __global const uint* input_offsets,      // Offsets de início de cada frame no input_buffer
    __global const uint* input_sizes,        // Tamanho de cada frame
    __global const uint* output_offsets,     // Offsets de início de cada frame no output_buffer
    __global const uint* output_max_sizes,   // Tamanho máximo de saída para cada frame
    __global uint* hash_table_buffer,        // Buffer gigante contendo todas as tabelas hash (stride = HASH_TABLE_SIZE)
    __global uint* debug_out_buffer,         // Buffer de debug (stride = 1)
    __global uint* compressed_size_out_buffer // Buffer de tamanhos finais (stride = 1)
) {{
    // Cada thread processa UM frame inteiro
    uint gid = get_global_id(0);
    
    // Setup dos ponteiros e tamanhos para ESTE frame
    uint input_start = input_offsets[gid];
    uint input_size = input_sizes[gid];
    uint output_start = output_offsets[gid];
    uint max_compressed_size = output_max_sizes[gid];
    
    // Ponteiros locais
    __global const uchar* input_data = input_buffer + input_start;
    __global uchar* output_data = output_buffer + output_start;
    
    // Tabela Hash dedicada para esta thread/frame
    // Offset = gid * HASH_TABLE_SIZE (em elementos uint, não bytes)
    __global uint* hash_table = hash_table_buffer + (gid * HASH_TABLE_SIZE);
    
    // Debug e Output Size dedicados
    __global uint* debug_out = debug_out_buffer ? (debug_out_buffer + gid) : 0;
    __global uint* compressed_size_out = compressed_size_out_buffer + gid;

    if (debug_out) *debug_out = DBG_START;

    __global const uchar* ip = input_data;
    __global uchar* op = output_data;
    __global uchar* op_limit = output_data + max_compressed_size;

    // Margem de segurança para evitar problemas nos últimos bytes (LASTLITERALS)
    const uint last_literal_pos = input_size - 12;

    uint anchor = 0;
    uint consecutive_misses = 0;

    uint loop_safety_count = 0;
    
    // Hash Caching Optimization
    uint cached_hash = 0;
    uint cached_pos = 0;

    if (debug_out) *debug_out = DBG_LOOP_START;

    while ((ip - input_data) < last_literal_pos && op < op_limit) {{
        // loop_safety_count removed - was causing premature exit on incompressible data
        
        uint current_pos = (uint)(ip - input_data);
        uint hash;

        // Hash Caching Check
        if (current_pos == cached_pos) {{
            hash = cached_hash;
        }} else {{
            hash = LZ4_hash4(ip);
        }}

        // --------------------------------------------------------------------
        // 1) MATCH EM ip (current_pos)
        // --------------------------------------------------------------------
        uint best_pos1 = 0;
        uint best_len1 = 0;
        find_best_match(
            input_data,
            input_size,
            ip,
            current_pos,
            last_literal_pos,
            hash_table,
            &best_pos1,
            &best_len1,
            hash
        );

        // Se não encontrou match em ip: trata como literal simples
        if (best_len1 < MIN_MATCH) {{
            if (debug_out) *debug_out = DBG_NO_MATCH;
            consecutive_misses++;

            // Adaptive Skip com 3 Níveis: AGRESSIVO (dobrado dos valores originais)
            uint skip_step = 1;
            
            // Nível 1: Misses moderados (16-64) - threshold reduzido pela metade
            // Skip progressivo: 1→2→4→8→16
            if (consecutive_misses > 16 && consecutive_misses <= 64) {{
                skip_step = min(16u, 1u << ((consecutive_misses - 16) / 16));
            }}
            // Nível 2: Muitos misses (64-256) - dados muito incompressíveis
            // Skip mais agressivo: 16→32→64
            else if (consecutive_misses > 64 && consecutive_misses <= 256) {{
                skip_step = min(64u, 16u + ((consecutive_misses - 64) / 32));
            }}
            // Nível 3: Incompressível total (>256)
            // Skip máximo: 64 bytes de uma vez (dobrado de 32)
            else if (consecutive_misses > 256) {{
                skip_step = 64u;
            }}
            
            ip += skip_step;
            continue;
        }}

        // Temos um match em ip, então podemos considerar lazy parsing (lookahead).
        consecutive_misses = 0;

        // --------------------------------------------------------------------
        // 2) LAZY PARSING: tenta match em ip+1 (next_pos)
        // --------------------------------------------------------------------
        uint sel_match_pos = best_pos1;
        uint sel_match_len = best_len1;
        uint sel_start_pos = current_pos;          // posição onde o match começa
        __global const uchar* sel_ip = ip;         // ponteiro para início do match

        uint next_pos = current_pos + 1;
        
        // Otimização: Se o match atual já for bom o suficiente (>32), pula lazy parsing
        if (best_len1 < 32 && next_pos < last_literal_pos) {{
            uint best_pos2 = 0;
            uint best_len2 = 0;
            
            // Calculate hash for next position and cache it
            uint next_hash = LZ4_hash4(ip + 1);
            cached_hash = next_hash;
            cached_pos = next_pos;

            find_best_match(
                input_data,
                input_size,
                ip + 1,
                next_pos,
                last_literal_pos,
                hash_table,
                &best_pos2,
                &best_len2,
                next_hash
            );

            // Só troca para o match em ip+1 se ele for "bem melhor"
            if (best_len2 >= MIN_MATCH &&
                best_len2 >= best_len1 + LAZY_DELTA) {{

                sel_match_pos = best_pos2;
                sel_match_len = best_len2;
                sel_start_pos = next_pos;
                sel_ip = ip + 1;
            }}
        }}

        if (debug_out) *debug_out = DBG_MATCH_FOUND;

        // --------------------------------------------------------------------
        // 3) EMISSÃO DO MATCH SELECIONADO (LZ4 compatível)
        // --------------------------------------------------------------------

        // Literais: tudo desde anchor até o byte anterior ao início do match
        uint literal_len = sel_start_pos - anchor;

        // Output buffer pre-check: verifica espaço ANTES de emitir
        // Worst case: 1 token + 5 lit_len + literal_len + 3 offset + 5 match_len
        uint total_output_needed = 1 + 5 + literal_len + 3 + 5;
        if (op + total_output_needed > op_limit) {{
            if (debug_out) *debug_out = DBG_ERROR_OP_LIMIT;
            break;
        }}

        __global uchar* token_ptr = op++;

        if (literal_len >= 15) {{
            op = write_length(op, literal_len - 15, op_limit);
            if (op >= op_limit) break;
        }}

        // Copia literais (Otimizado: 8 bytes por vez)
        __global const uchar* lit_src = input_data + anchor;
        uint lit_rem = literal_len;
        
        while (lit_rem >= 8) {{
            // OTIMIZAÇÃO: Vectorized copy usando vload8/vstore8 (safe for unaligned)
            uchar8 vec = vload8(0, lit_src);
            vstore8(vec, 0, op);
            
            op += 8;
            lit_src += 8;
            lit_rem -= 8;
        }}
        while (lit_rem > 0) {{
            *op++ = *lit_src++;
            lit_rem--;
        }}

        // Offset (3 bytes, formato extendido lz_ext3_gpu)
        uint offset = sel_start_pos - sel_match_pos;
        op[0] = (uchar)(offset & 0xFF);
        op[1] = (uchar)((offset >> 8) & 0xFF);
        op[2] = (uchar)((offset >> 16) & 0xFF);
        op += 3;

        // Match length
        uint match_len = sel_match_len;
        uint match_len_enc = match_len - MIN_MATCH;

        uchar token = (uchar)((literal_len < 15 ? literal_len : 15) << 4);
        token |= (uchar)(match_len_enc < 15 ? match_len_enc : 15);
        *token_ptr = token;

        if (match_len_enc >= 15) {{
            op = write_length(op, match_len_enc - 15, op_limit);
            if (op >= op_limit) break;
        }}

        // Avança IP para o fim do match
        ip = input_data + sel_start_pos + match_len;
        anchor = (uint)(ip - input_data);

        // Atualiza hash para ip-2 (como em LZ4) se ainda houver espaço
        if (match_len > MIN_MATCH && (ip - input_data) < last_literal_pos) {{
            uint pos_for_hash = (uint)(ip - input_data) - 2;
            __global const uchar* ptr_for_hash = input_data + pos_for_hash;

            uint h2 = LZ4_hash4(ptr_for_hash);
            uint base2 = h2 * HASH_CANDIDATES;

            for (int c = (int)HASH_CANDIDATES - 1; c > 0; --c) {{
                hash_table[base2 + (uint)c] = hash_table[base2 + (uint)(c - 1)];
            }}
            hash_table[base2 + 0] = pos_for_hash;
        }}
    }}

    if (debug_out) *debug_out = DBG_LOOP_END;

    // ------------------------------------------------------------------------
    // FINALIZAÇÃO: emite literais restantes (se houver)
    // ------------------------------------------------------------------------
    uint literal_len = input_size - anchor;

    uint length_bytes = (literal_len / 255) + 1;
    if (op + 1 + length_bytes + literal_len < op_limit) {{
        if (debug_out) *debug_out = DBG_FINALIZATION;

        __global uchar* token_ptr = op++;
        uchar token = (uchar)((literal_len < 15 ? literal_len : 15) << 4);
        *token_ptr = token;

        if (literal_len >= 15) {{
            op = write_length(op, literal_len - 15, op_limit);
        }}

        // Copia literais finais (Otimizado: 8 bytes por vez usando vload8/vstore8)
        __global const uchar* lit_src = input_data + anchor;
        uint lit_rem = literal_len;
        
        // Vectorized copy - 8 bytes at a time (coalesced access)
        while (lit_rem >= 8) {{
            uchar8 vec = vload8(0, lit_src);
            vstore8(vec, 0, op);
            op += 8;
            lit_src += 8;
            lit_rem -= 8;
        }}
        
        // Handle remaining bytes (< 8)
        while (lit_rem > 0) {{
            *op++ = *lit_src++;
            lit_rem--;
        }}

        if (debug_out) *debug_out = DBG_SUCCESS;
    }} else {{
        if (debug_out) *debug_out = DBG_ERROR_OP_LIMIT;
    }}

    *compressed_size_out = (uint)(op - output_data);
}}
"""

# Prepare values to write for the shift (current_pos, then candidates 0..K-2)
_values_to_write = ["current_pos"] + [f"candidates[{i}]" for i in range(HASH_CANDIDATES - 1)]

# Gerar código C customizado para acesso vetorial
_load_code = gen_opencl_vector_load(HASH_CANDIDATES, "hash_table", "base_index", "candidates")
_store_code = gen_opencl_vector_store(HASH_CANDIDATES, "hash_table", "base_index", _values_to_write)

# Gerar kernel final substituindo os valores de configuração
LZ4_KERNEL_SOURCE = _LZ4_KERNEL_TEMPLATE.format(
    hash_log=HASH_LOG,
    hash_candidates=HASH_CANDIDATES,
    good_enough_match=GOOD_ENOUGH_MATCH,
    load_candidates_code=_load_code,
    update_hash_table_code=_store_code
)



class OpenCLBuffer:
    """Gerenciador de contexto para buffers OpenCL."""
    def __init__(self, ctx, flags, size_or_hostbuf=None, hostbuf=None):
        self.ctx = ctx
        self.flags = flags
        self.size_or_hostbuf = size_or_hostbuf
        self.hostbuf = hostbuf
        self.buffer = None

    def __enter__(self):
        if self.hostbuf is not None:
            # Buffer com dados iniciais do host
            self.buffer = cl.Buffer(self.ctx, self.flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.hostbuf)
        elif self.size_or_hostbuf is not None:
            # Buffer alocado na GPU (sem dados iniciais)
            self.buffer = cl.Buffer(self.ctx, self.flags, self.size_or_hostbuf)
        else:
            raise ValueError("Deve fornecer size_or_hostbuf ou hostbuf")
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        # O pyopencl gerencia a liberação de recursos quando o objeto é deletado.
        # Forçar a exclusão garante que o recurso seja liberado mais rapidamente.
        if self.buffer:
            del self.buffer

class GPU_LZ4_Compressor:
    """
    Wrapper para o kernel LZ4 OpenCL.
    Implementa uma compressão LZ4 simplificada (PoC) na GPU.
    """
    def __init__(self, device_index: int = 0, max_input_size: int = 16 * 1024 * 1024, batch_size: Optional[int] = None):
        self.enabled = False
        self.device_index = device_index
        self.max_input_size = 0 # Será definido por allocate_buffers
        
        # Calcular batch size automaticamente se não fornecido
        if batch_size is None:
            try:
                from gpu_capabilities import get_recommended_batch_size
                frame_size_mb = max_input_size // (1024 * 1024)
                self.batch_size = get_recommended_batch_size(frame_size_mb=frame_size_mb)
                print(f"[GPU_LZ4_Compressor] Batch Size calculado automaticamente: {self.batch_size} frames")
            except Exception as e:
                print(f"[GPU_LZ4_Compressor] Erro ao calcular batch size: {e}. Usando padrão: 24")
                self.batch_size = 24
        else:
            # Usuário forneceu explicitamente - respeitar o valor
            self.batch_size = batch_size
            print(f"[GPU_LZ4_Compressor] Usando Batch Size fornecido pelo usuário: {self.batch_size} frames")
        
        self.ctx: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.program: Optional[cl.Program] = None
        self.kernel = None
        
        # Buffers Persistentes (GPU)
        self.buf_in = None
        self.buf_out = None
        self.buf_input_offsets = None
        self.buf_input_sizes = None
        self.buf_output_offsets = None
        self.buf_output_max_sizes = None
        self.buf_compressed_sizes = None
        self.buf_debug = None 
        self.buf_hash_table = None
        
        # Pinned Memory Buffers (Host - para DMA transfer)
        self.pinned_input = None
        self.pinned_output = None
        self.pinned_input_array = None  # numpy view do buffer
        self.pinned_output_array = None
        
        # Event tracking para async transfers
        self.last_upload_event = None
        self.last_kernel_event = None
        
        self._initialize_opencl()
        if self.enabled:
            self.allocate_buffers(max_input_size)

    def _initialize_opencl(self):
        try:
            # 1. Configuração do Contexto
            platforms = cl.get_platforms()
            if not platforms: raise Exception("Nenhuma plataforma OpenCL encontrada")
            
            devices = []
            for p in platforms:
                try:
                    devs = p.get_devices(device_type=cl.device_type.GPU)
                    devices.extend(devs)
                except: pass # Ignorar plataformas sem GPUs ou erros
            
            if not devices: raise Exception("Nenhuma GPU OpenCL encontrada")
            if self.device_index >= len(devices): raise Exception(f"GPU index {self.device_index} inválido")
                
            target_device = devices[self.device_index]

            # Usar a GPU especificada
            self.ctx = cl.Context(devices=[target_device])
            self.queue = cl.CommandQueue(self.ctx)
            
            # 2. Carregar e Construir o Programa
            self.program = cl.Program(self.ctx, LZ4_KERNEL_SOURCE).build()
            # cria UMA instância de kernel e guarda
            self.kernel = cl.Kernel(self.program, "lz4_compress_block")

            
            self.enabled = True
            print(f"[GPU_LZ4] Compressor LZ4 OpenCL ativado em: {target_device.name} (Index: {self.device_index})")

        except Exception as e:
            print(f"[GPU_LZ4] Falha na inicialização OpenCL (Dev {self.device_index}): {e}")
            self.enabled = False

    def allocate_buffers(self, frame_size: int):
        if not self.enabled: return
        
        try:
            # Alocar buffers apenas se o tamanho mudar ou não existirem
            if self.max_input_size != frame_size:
                # Liberar buffers antigos se existirem
                self.release_buffers()

                self.max_input_size = frame_size
                
                # Tamanho máximo de saída LZ4 com 3-byte offsets:
                # Overhead aumentado devido aos offsets de 3 bytes
                # Usando margem conservadora para dados incompressíveis
                self.max_compressed_size = (
                self.max_input_size
                + self.max_input_size // 255
                + 128   # margem de segurança aumentada
)

                
                mf = cl.mem_flags
                
                # Calcular tamanhos totais ANTES de alocar
                total_input_size = self.max_input_size * self.batch_size
                total_output_size = self.max_compressed_size * self.batch_size
                hash_table_bytes = HASH_TABLE_SIZE * 4 * self.batch_size
                total_vram_needed = total_input_size + total_output_size + hash_table_bytes
                
                # Verificar VRAM disponível
                try:
                    # Pegar device da GPU
                    devices = []
                    platforms = cl.get_platforms()
                    for p in platforms:
                        try:
                            devices.extend(p.get_devices(device_type=cl.device_type.GPU))
                        except:
                            pass
                    
                    if self.device_index < len(devices):
                        device = devices[self.device_index]
                        vram_total = device.global_mem_size
                        vram_total_mb = vram_total / 1024 / 1024
                        vram_needed_mb = total_vram_needed / 1024 / 1024
                        vram_usage_pct = (total_vram_needed / vram_total) * 100
                        
                        print(f"[GPU_LZ4] VRAM Total: {vram_total_mb:.0f}MB | Necessário: {vram_needed_mb:.0f}MB ({vram_usage_pct:.1f}%)")
                        
                        if total_vram_needed > vram_total * 0.83:  # > 83% de VRAM
                            print(f"[GPU_LZ4] AVISO: Uso de VRAM muito alto ({vram_usage_pct:.1f}%)!")
                            print(f"[GPU_LZ4] Considere reduzir batch_size ou HASH_CANDIDATES")
                        
                        if total_vram_needed > vram_total:
                            print(f"[GPU_LZ4] ERRO: Batch size muito grande! Necessário {vram_needed_mb:.0f}MB mas apenas {vram_total_mb:.0f}MB disponível")
                            self.enabled = False
                            return
                except Exception as e:
                    print(f"[GPU_LZ4] Aviso: Não foi possível verificar VRAM: {e}")
                
                # Buffers de Dados (Tamanho Total = Frame Size * Batch Size)
                total_input_size = self.max_input_size * self.batch_size
                total_output_size = self.max_compressed_size * self.batch_size
                
                # ============================================================
                # OTIMIZAÇÃO: PINNED MEMORY + ASYNC TRANSFERS
                # ============================================================
                # Usar ALLOC_HOST_PTR para buffers de entrada/saída
                # Isso habilita DMA transfers (Direct Memory Access) entre CPU e GPU,
                # ignorando a CPU e acelerando a transferência significativamente.
                # 
                # Fluxo otimizado:
                # 1. CPU escreve no pinned buffer (mapeado na RAM)
                # 2. GPU lê via DMA (não-bloqueante)
                # 3. Kernel executa
                # 4. GPU escreve resultado via DMA
                # 5. CPU lê do pinned buffer
                # ============================================================
                
                # Buffer de Input com Pinned Memory
                self.buf_in = cl.Buffer(
                    self.ctx, 
                    mf.READ_ONLY | mf.ALLOC_HOST_PTR, 
                    total_input_size
                )
                
                # Buffer de Output com Pinned Memory
                self.buf_out = cl.Buffer(
                    self.ctx, 
                    mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, 
                    total_output_size
                )
                
                # Mapear buffers pinned para acesso pelo host
                # Isso cria uma view numpy que aponta diretamente para a memória pinned
                self.pinned_input_array = np.zeros(total_input_size, dtype=np.uint8)
                self.pinned_output_array = np.zeros(total_output_size, dtype=np.uint8)
                
                # Buffers de Controle (Arrays de tamanho batch_size)
                uint_size = np.dtype(np.uint32).itemsize
                self.buf_input_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                self.buf_input_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                self.buf_output_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                self.buf_output_max_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                self.buf_compressed_sizes = cl.Buffer(self.ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                self.buf_debug = cl.Buffer(self.ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, self.batch_size * uint_size)
                
                # Hash Table Global (HASH_TABLE_SIZE * Batch Size)
                # Não usa ALLOC_HOST_PTR pois não precisa de transferência frequente
                hash_table_bytes = HASH_TABLE_SIZE * 4 * self.batch_size
                self.buf_hash_table = cl.Buffer(
                    self.ctx, 
                    mf.READ_WRITE, 
                    hash_table_bytes
                )
                
                # Calcular uso total de VRAM
                total_vram_mb = (total_input_size + total_output_size + hash_table_bytes) / 1024 / 1024
                
                print(f"[GPU_LZ4] Buffers alocados com PINNED MEMORY (batch={self.batch_size}):")
                print(f"  - Input (Pinned): {total_input_size/1024/1024:.1f}MB ({self.max_input_size/1024/1024:.1f}MB x {self.batch_size})")
                print(f"  - Output (Pinned): {total_output_size/1024/1024:.1f}MB ({self.max_compressed_size/1024/1024:.1f}MB x {self.batch_size})")
                print(f"  - Hash: {hash_table_bytes/1024/1024:.1f}MB ({HASH_TABLE_SIZE*4/1024:.0f}KB x {self.batch_size})")
                print(f"  - TOTAL VRAM: {total_vram_mb:.1f}MB")
                print(f"  - Async Transfers: ENABLED")
                
        except Exception as e:
            print(f"[GPU_LZ4] Erro ao alocar buffers: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False

    def release_buffers(self):
        """Libera explicitamente os buffers OpenCL e Pinned Memory."""
        # GPU Buffers
        if self.buf_in: del self.buf_in; self.buf_in = None
        if self.buf_out: del self.buf_out; self.buf_out = None
        if self.buf_input_offsets: del self.buf_input_offsets; self.buf_input_offsets = None
        if self.buf_input_sizes: del self.buf_input_sizes; self.buf_input_sizes = None
        if self.buf_output_offsets: del self.buf_output_offsets; self.buf_output_offsets = None
        if self.buf_output_max_sizes: del self.buf_output_max_sizes; self.buf_output_max_sizes = None
        if self.buf_compressed_sizes: del self.buf_compressed_sizes; self.buf_compressed_sizes = None
        if self.buf_debug: del self.buf_debug; self.buf_debug = None
        if self.buf_hash_table: del self.buf_hash_table; self.buf_hash_table = None
        
        # Pinned Memory Arrays
        self.pinned_input_array = None
        self.pinned_output_array = None
        
        # Event tracking
        self.last_upload_event = None
        self.last_kernel_event = None
        
        self.max_input_size = 0
        self.max_compressed_size = 0

    def release(self):
        """Libera explicitamente todos os recursos OpenCL."""
        self.enabled = False
        self.release_buffers()
        if self.queue:
            self.queue.finish()
            self.queue = None
        if self.program:
            self.program = None
        if self.kernel:
            self.kernel = None
        if self.ctx:
            self.ctx = None
        
        import gc
        gc.collect()

    def compress_batch(self, frames: List[bytes], frame_ids: List[int] = None) -> List[Tuple[bytes, int, float]]:
        """
        Comprime um lote de frames em paralelo na GPU.
        """
        if not self.enabled:
            # Fallback CPU sequencial
            results = []
            for i, data in enumerate(frames):
                # Retorna RAW para manter compatibilidade com lógica antiga se falhar
                results.append((data, len(data), 0.0)) 
            return results

        num_frames = len(frames)
        if num_frames > self.batch_size:
            raise ValueError(f"Batch size {num_frames} excede limite {self.batch_size}")

        start_gpu = time.time()
        
        # 1. Preparar metadados e buffers de host
        input_offsets = np.zeros(self.batch_size, dtype=np.uint32)
        input_sizes = np.zeros(self.batch_size, dtype=np.uint32)
        output_offsets = np.zeros(self.batch_size, dtype=np.uint32)
        output_max_sizes = np.zeros(self.batch_size, dtype=np.uint32)
        
        current_input_offset = 0
        
        # Concatenar inputs
        total_input_bytes = sum(len(f) for f in frames)
        
        # Verificar se cabe nos buffers
        if total_input_bytes > self.max_input_size * self.batch_size:
             print(f"[GPU_LZ4] Aviso: Batch total size {total_input_bytes} excede alocação padrão.")
             return [(f, len(f), 0.0) for f in frames]

        # ============================================================
        # OTIMIZAÇÃO: Usar Pinned Memory Array diretamente
        # ============================================================
        # Escrever diretamente no array pinned evita cópia extra
        pinned_view = self.pinned_input_array[:total_input_bytes]
        
        for i, frame in enumerate(frames):
            size = len(frame)
            input_offsets[i] = current_input_offset
            input_sizes[i] = size
            
            # Copiar dados diretamente para o buffer pinned
            pinned_view[current_input_offset:current_input_offset+size] = np.frombuffer(frame, dtype=np.uint8)
            
            current_input_offset += size
            
        # Recalcular output offsets para usar stride fixo e seguro
        for i in range(self.batch_size):
            output_offsets[i] = i * self.max_compressed_size
            output_max_sizes[i] = self.max_compressed_size

        # ============================================================
        # 2. ASYNC TRANSFERS: Upload não-bloqueante com eventos
        # ============================================================
        # Todas as transferências são não-bloqueantes (is_blocking=False)
        # O pipeline permite overlap: CPU prepara próximo batch enquanto GPU processa
        
        # Upload do buffer de input (DMA transfer via pinned memory)
        evt_input = cl.enqueue_copy(
            self.queue, 
            self.buf_in, 
            pinned_view,
            is_blocking=False
        )
        
        # Upload dos metadados (pequenos, também async)
        evt_offsets = cl.enqueue_copy(self.queue, self.buf_input_offsets, input_offsets, is_blocking=False)
        evt_sizes = cl.enqueue_copy(self.queue, self.buf_input_sizes, input_sizes, is_blocking=False)
        evt_out_off = cl.enqueue_copy(self.queue, self.buf_output_offsets, output_offsets, is_blocking=False)
        evt_max_sizes = cl.enqueue_copy(self.queue, self.buf_output_max_sizes, output_max_sizes, is_blocking=False)
        
        # Guardar último evento de upload para potencial pipeline futuro
        self.last_upload_event = evt_input
        
        # ============================================================
        # 3. KERNEL EXECUTION: Aguardar uploads antes de executar
        # ============================================================
        global_size = (num_frames,)
        local_size = None

        # O kernel depende dos uploads - usar wait_for para sincronização
        kernel_event = self.kernel(
            self.queue, global_size, local_size,
            self.buf_in,
            self.buf_out,
            self.buf_input_offsets,
            self.buf_input_sizes,
            self.buf_output_offsets,
            self.buf_output_max_sizes,
            self.buf_hash_table,
            self.buf_debug,
            self.buf_compressed_sizes,
            wait_for=[evt_input, evt_offsets, evt_sizes, evt_out_off, evt_max_sizes]
        )
        
        self.last_kernel_event = kernel_event
      
        # ============================================================
        # 4. ASYNC READ: Ler resultados (depende do kernel)
        # ============================================================
        compressed_sizes = np.zeros(self.batch_size, dtype=np.uint32)
        debug_codes = np.zeros(self.batch_size, dtype=np.uint32)
        
        # Ler metadados de resultado (pequenos, pode ser blocking)
        evt_read_sizes = cl.enqueue_copy(
            self.queue, 
            compressed_sizes, 
            self.buf_compressed_sizes, 
            wait_for=[kernel_event],
            is_blocking=False
        )
        evt_read_debug = cl.enqueue_copy(
            self.queue, 
            debug_codes, 
            self.buf_debug, 
            wait_for=[kernel_event],
            is_blocking=False
        )
        
        # Aguardar leitura dos metadados para processar resultados
        evt_read_sizes.wait()
        evt_read_debug.wait()
        
        results = []
        total_time = time.time() - start_gpu
        avg_time = total_time / num_frames if num_frames > 0 else 0
        
        # Ler output comprimido do buffer pinned (mais eficiente que múltiplas leituras)
        # Primeiro, copiar todo o output para o array pinned
        total_output_needed = sum(compressed_sizes[i] for i in range(num_frames) 
                                   if debug_codes[i] != 8 and compressed_sizes[i] < input_sizes[i] and compressed_sizes[i] > 0)
        
        if total_output_needed > 0:
            # Ler todo o buffer de output de uma vez (mais eficiente)
            cl.enqueue_copy(
                self.queue, 
                self.pinned_output_array, 
                self.buf_out, 
                wait_for=[kernel_event],
                is_blocking=True
            )
        
        for i in range(num_frames):
            comp_size = int(compressed_sizes[i])
            debug_code = int(debug_codes[i])
            orig_size = int(input_sizes[i])
            
            # Verificar erro/limite
            if debug_code == 8 or comp_size >= orig_size or comp_size == 0:
                results.append((frames[i], len(frames[i]), avg_time))
            else:
                # Ler dados comprimidos do buffer pinned (já carregado)
                out_offset = int(output_offsets[i])
                compressed_data = bytes(self.pinned_output_array[out_offset:out_offset+comp_size])
                results.append((compressed_data, comp_size, avg_time))
                
        return results

    def compress(self, data: bytes, frame_id: int = -1) -> Tuple[bytes, int, float]:
        """Wrapper de compatibilidade para comprimir um único frame."""
        results = self.compress_batch([data], [frame_id])
        return results[0]
