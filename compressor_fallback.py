"""
compressor_fallback.py
CPU Fallback LZ4 Compressor with Extended Format (3-byte offsets)

Compressor LZ4 ext3 CPU-only para uso quando OpenCL/GPU não está disponível.
Usa processamento paralelo baseado no número de CPUs lógicos.

Uso:
    from compressor_fallback import CPU_LZ4_Compressor
    
    compressor = CPU_LZ4_Compressor()
    compressed, size, elapsed = compressor.compress(data, frame_id=0)
    
    # Batch processing
    results = compressor.compress_batch(frames, frame_ids)
"""

from __future__ import annotations
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

# Constantes LZ4 (compatíveis com gpu_lz4_compressor.py)
HASH_LOG = 20
HASH_CANDIDATES = 7
HASH_ENTRIES = (1 << HASH_LOG)  # 1048576 (1M)
MIN_MATCH = 4
MAX_DISTANCE = (1 << 24)  # 16MB window (3-byte offset)
PRIME = 0x9E3779B1  # Hash prime

# Lazy parsing: só troca para match em ip+1 se for bem melhor
LAZY_DELTA = 2

# Early exit threshold
GOOD_ENOUGH_MATCH = 128


def _hash4(data: bytes, pos: int) -> int:
    """Hash LZ4 de 4 bytes."""
    if pos + 4 > len(data):
        return 0
    value = (data[pos] | 
             (data[pos + 1] << 8) | 
             (data[pos + 2] << 16) | 
             (data[pos + 3] << 24))
    return ((value * PRIME) & 0xFFFFFFFF) >> (32 - HASH_LOG)


def _find_best_match(
    data: bytes,
    data_len: int,
    current_pos: int,
    last_literal_pos: int,
    hash_table: List[List[int]],
    hash_value: int
) -> Tuple[int, int]:
    """
    Encontra o melhor match para a posição atual.
    Retorna (best_match_pos, best_match_len).
    """
    best_pos = 0
    best_len = 0
    
    candidates = hash_table[hash_value]
    
    for candidate_pos in candidates:
        if candidate_pos == 0:
            break
            
        distance = current_pos - candidate_pos
        
        if distance == 0 or distance >= MAX_DISTANCE or candidate_pos + MIN_MATCH > data_len:
            continue
        
        # Quick 4-byte prefix check
        if (data[current_pos] != data[candidate_pos] or
            data[current_pos + 1] != data[candidate_pos + 1] or
            data[current_pos + 2] != data[candidate_pos + 2] or
            data[current_pos + 3] != data[candidate_pos + 3]):
            continue
        
        # Extend match
        match_len = MIN_MATCH
        max_match_len = last_literal_pos - current_pos
        
        while match_len < max_match_len:
            if data[current_pos + match_len] != data[candidate_pos + match_len]:
                break
            match_len += 1
        
        if match_len > best_len:
            best_len = match_len
            best_pos = candidate_pos
            
            if match_len >= GOOD_ENOUGH_MATCH:
                break
    
    # Update hash table: shift right and insert current_pos at position 0
    new_candidates = [current_pos] + candidates[:-1]
    hash_table[hash_value] = new_candidates
    
    return best_pos, best_len


def compress_lz4_ext3(data: bytes) -> bytes:
    """
    Comprime dados usando LZ4 ext3 (3-byte offsets, 16MB window).
    Compatível com decompress_lz4_ext3().
    
    Args:
        data: Dados a comprimir
        
    Returns:
        Dados comprimidos ou dados originais se incompressíveis
    """
    input_size = len(data)
    if input_size < 13:  # Muito pequeno para comprimir
        return data
    
    # Output buffer (worst case: ligeiramente maior que input)
    max_output_size = input_size + (input_size // 255) + 128
    output = bytearray(max_output_size)
    
    # Hash table: lista de listas com HASH_CANDIDATES posições por hash
    hash_table: List[List[int]] = [[0] * HASH_CANDIDATES for _ in range(HASH_ENTRIES)]
    
    ip = 0  # Input position
    op = 0  # Output position
    anchor = 0  # Início dos literais pendentes
    
    last_literal_pos = input_size - 12  # Margem de segurança
    consecutive_misses = 0
    
    while ip < last_literal_pos and op < max_output_size - 20:
        hash_val = _hash4(data, ip)
        
        # Buscar match
        best_pos, best_len = _find_best_match(
            data, input_size, ip, last_literal_pos, hash_table, hash_val
        )
        
        if best_len < MIN_MATCH:
            # Sem match - adaptive skip
            consecutive_misses += 1
            
            skip_step = 1
            if consecutive_misses > 16 and consecutive_misses <= 64:
                skip_step = min(16, 1 << ((consecutive_misses - 16) // 16))
            elif consecutive_misses > 64 and consecutive_misses <= 256:
                skip_step = min(64, 16 + ((consecutive_misses - 64) // 32))
            elif consecutive_misses > 256:
                skip_step = 64
            
            ip += skip_step
            continue
        
        consecutive_misses = 0
        
        # Lazy parsing: tentar match em ip+1
        sel_match_pos = best_pos
        sel_match_len = best_len
        sel_start_pos = ip
        
        if best_len < 32 and ip + 1 < last_literal_pos:
            next_hash = _hash4(data, ip + 1)
            best_pos2, best_len2 = _find_best_match(
                data, input_size, ip + 1, last_literal_pos, hash_table, next_hash
            )
            
            if best_len2 >= MIN_MATCH and best_len2 >= best_len + LAZY_DELTA:
                sel_match_pos = best_pos2
                sel_match_len = best_len2
                sel_start_pos = ip + 1
        
        # Emitir match
        literal_len = sel_start_pos - anchor
        
        # Verificar espaço no output
        total_needed = 1 + (literal_len // 255) + 1 + literal_len + 3 + ((sel_match_len - MIN_MATCH) // 255) + 1
        if op + total_needed > max_output_size:
            break
        
        # Token
        token_pos = op
        op += 1
        
        # Literal length encoding
        if literal_len >= 15:
            output[token_pos] = 15 << 4
            remaining = literal_len - 15
            while remaining >= 255:
                output[op] = 255
                op += 1
                remaining -= 255
            output[op] = remaining
            op += 1
        else:
            output[token_pos] = literal_len << 4
        
        # Copy literals
        output[op:op + literal_len] = data[anchor:anchor + literal_len]
        op += literal_len
        
        # Offset (3 bytes, little-endian)
        offset = sel_start_pos - sel_match_pos
        output[op] = offset & 0xFF
        output[op + 1] = (offset >> 8) & 0xFF
        output[op + 2] = (offset >> 16) & 0xFF
        op += 3
        
        # Match length encoding
        match_len_enc = sel_match_len - MIN_MATCH
        if match_len_enc >= 15:
            output[token_pos] |= 15
            remaining = match_len_enc - 15
            while remaining >= 255:
                output[op] = 255
                op += 1
                remaining -= 255
            output[op] = remaining
            op += 1
        else:
            output[token_pos] |= match_len_enc
        
        # Avançar IP
        ip = sel_start_pos + sel_match_len
        anchor = ip
        
        # Update hash para ip-2
        if sel_match_len > MIN_MATCH and ip < last_literal_pos:
            pos_hash = ip - 2
            h2 = _hash4(data, pos_hash)
            new_candidates = [pos_hash] + hash_table[h2][:-1]
            hash_table[h2] = new_candidates
    
    # Finalização: emitir literais restantes
    literal_len = input_size - anchor
    
    length_bytes = (literal_len // 255) + 2
    if op + 1 + length_bytes + literal_len < max_output_size:
        token_pos = op
        op += 1
        
        if literal_len >= 15:
            output[token_pos] = 15 << 4
            remaining = literal_len - 15
            while remaining >= 255:
                output[op] = 255
                op += 1
                remaining -= 255
            output[op] = remaining
            op += 1
        else:
            output[token_pos] = literal_len << 4
        
        output[op:op + literal_len] = data[anchor:anchor + literal_len]
        op += literal_len
    else:
        # Sem espaço - retornar original
        return data
    
    # Verificar se compressão foi efetiva
    if op >= input_size:
        return data  # Sem ganho, retornar original
    
    return bytes(output[:op])


def _compress_single_frame(args: Tuple[bytes, int]) -> Tuple[int, bytes, int, float]:
    """
    Worker function para compressão paralela.
    
    Args:
        args: Tupla (frame_data, frame_id)
        
    Returns:
        Tupla (frame_id, compressed_data, compressed_size, elapsed_time)
    """
    data, frame_id = args
    start = time.time()
    
    compressed = compress_lz4_ext3(data)
    
    elapsed = time.time() - start
    return (frame_id, compressed, len(compressed), elapsed)


class CPU_LZ4_Compressor:
    """
    Compressor LZ4 ext3 CPU-only com processamento paralelo.
    Interface compatível com GPU_LZ4_Compressor.
    """
    
    def __init__(self):
        self.enabled = True
        self.cpu_count = multiprocessing.cpu_count()
        self.device_index = -1  # -1 = CPU
        
        print(f"[CPU_LZ4] Compressor CPU inicializado com {self.cpu_count} workers")
    
    def compress_batch(
        self, 
        frames: List[bytes], 
        frame_ids: Optional[List[int]] = None
    ) -> List[Tuple[bytes, int, float]]:
        """
        Comprime um lote de frames em paralelo usando CPU.
        
        Args:
            frames: Lista de frames a comprimir
            frame_ids: IDs dos frames (opcional)
            
        Returns:
            Lista de tuplas (compressed_data, compressed_size, elapsed_time)
        """
        if frame_ids is None:
            frame_ids = list(range(len(frames)))
        
        num_frames = len(frames)
        if num_frames == 0:
            return []
        
        start_total = time.time()
        
        # Preparar argumentos para workers
        args_list = [(frames[i], frame_ids[i]) for i in range(num_frames)]
        
        # Processar em paralelo
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = {
                executor.submit(_compress_single_frame, args): args[1] 
                for args in args_list
            }
            
            for future in as_completed(futures):
                frame_id = futures[future]
                try:
                    fid, compressed, comp_size, elapsed = future.result()
                    results_dict[fid] = (compressed, comp_size, elapsed)
                except Exception as e:
                    # Fallback RAW em caso de erro
                    idx = frame_ids.index(frame_id)
                    orig_data = frames[idx]
                    results_dict[frame_id] = (orig_data, len(orig_data), 0.0)
                    print(f"[CPU_LZ4] Erro no frame {frame_id}: {e}")
        
        # Ordenar resultados por frame_id
        results = [results_dict[fid] for fid in frame_ids]
        
        total_elapsed = time.time() - start_total
        
        # Stats
        total_orig = sum(len(f) for f in frames)
        total_comp = sum(r[1] for r in results)
        if total_orig > 0:
            ratio = (1 - total_comp / total_orig) * 100
            speed = (total_orig / 1024 / 1024) / total_elapsed if total_elapsed > 0 else 0
            print(f"[CPU_LZ4] Batch {num_frames} frames: {total_orig/1024/1024:.1f}MB -> {total_comp/1024/1024:.1f}MB ({ratio:.1f}%) @ {speed:.1f} MB/s")
        
        return results
    
    def compress(self, data: bytes, frame_id: int = -1) -> Tuple[bytes, int, float]:
        """
        Comprime um único frame.
        
        Args:
            data: Dados a comprimir
            frame_id: ID do frame
            
        Returns:
            Tupla (compressed_data, compressed_size, elapsed_time)
        """
        results = self.compress_batch([data], [frame_id])
        return results[0]
    
    def release(self):
        """Libera recursos (compatibilidade com GPU_LZ4_Compressor)."""
        pass
