"""
Descompressor para o formato LZ4 GPU (LZ4 + Dedup).

Suporta:
- Deduplicação (copia de arquivos originais)
- LZ4 (Descompressão via CPU - compatível com o output da GPU)
- RAW

Uso:
    python decompressor_lz4.py F:\\saida_final.001 -o F:\\restaurado
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import struct
import zlib
import lz4.block # Mudança aqui: frame -> block
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from decompressor_lz4_ext3 import decompress_lz4_ext3
try:
    from gpu_lz4_decompressor import GPU_LZ4_Decompressor
except ImportError:
    GPU_LZ4_Decompressor = None

# Import GPU selector for interactive GPU selection
try:
    from gpu_selector import prompt_gpu_selection, get_excluded_indices
    GPU_SELECTOR_AVAILABLE = True
except ImportError:
    GPU_SELECTOR_AVAILABLE = False

# ============================================================
# CONFIGURAÇÃO DE BATCH SIZE
# ============================================================
# Defina como None para cálculo automático baseado em GPU capabilities,
# ou um valor inteiro para usar batch size fixo.
# Exemplo: BATCH_SIZE_OVERRIDE = 24  # Usar batch fixo de 24 frames
#          BATCH_SIZE_OVERRIDE = None # Calcular automaticamente (padrão)
#2-4 GB	8-16 frames
#4-8 GB	16-32 frames
#8-12 GB	32-64 frames
#12+ GB	64-128 frames

BATCH_SIZE_OVERRIDE = None
# ============================================================

# Determinar GPU_BATCH_SIZE: override fixo ou cálculo automático
if BATCH_SIZE_OVERRIDE is not None:
    GPU_BATCH_SIZE = BATCH_SIZE_OVERRIDE
    print(f"[GPU_LZ4] Batch Size FIXO (definido pelo usuário): {GPU_BATCH_SIZE} frames")
else:
    try:
        from gpu_capabilities import get_recommended_batch_size
        # Calcular batch size baseado nas capacidades da GPU (frame padrão: 16MB)
        # Usa 2/3 da recomendação conservadora
        GPU_BATCH_SIZE = get_recommended_batch_size(frame_size_mb=16)
        print(f"[GPU_LZ4] Batch Size AUTOMÁTICO: {GPU_BATCH_SIZE} frames (baseado em GPU capabilities)")
    except Exception as e:
        print(f"[GPU_LZ4] Erro ao calcular batch size: {e}. Usando padrão: 24")
        GPU_BATCH_SIZE = 24

MAX_WORKER_THREADS = 2       # Fewer workers to reduce GPU contention (OpenCL serialization)
GPU_FALLBACK_ENABLED = True  # Enable CPU fallback on GPU errors

def process_gpu_batch(frames: List[Dict], vol_handles: Dict, vol_locks: Dict, parent_dir: Path, gpu_decompressors: List[GPU_LZ4_Decompressor]) -> List[Tuple[int, bytes]]:
    """
    Process frames using GPU decompressors with automatic CPU fallback.
    Distributes frames round-robin across multiple GPUs if available.
    
    Args:
        frames: List of frame metadata dicts
        vol_handles: Dict of open file handles for volumes
        vol_locks: Dict of threading locks for volume access
        parent_dir: Parent directory containing volume files
        gpu_decompressors: List of GPU_LZ4_Decompressor instances
        
    Returns:
        List of (frame_id, decompressed_data) tuples
    """
    if not gpu_decompressors or not any(d.enabled for d in gpu_decompressors):
        # No GPUs available, return empty (will be handled by CPU)
        return []
    
    # Filter enabled GPUs
    enabled_gpus = [d for d in gpu_decompressors if d.enabled]
    
    # Distribute frames round-robin across GPUs
    gpu_assignments = [[] for _ in enabled_gpus]
    for i, fr in enumerate(frames):
        gpu_idx = i % len(enabled_gpus)
        gpu_assignments[gpu_idx].append((fr, i))  # Store frame with original index
    
    # Process each GPU's batch in parallel
    from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
    
    final_results = []
    
    with TPE(max_workers=len(enabled_gpus)) as executor:
        futures = {}
        
        # Submit GPU tasks
        for gpu_idx, gpu_frames in enumerate(gpu_assignments):
            if not gpu_frames:
                continue
                
            # Extract just the frames (without index)
            frames_only = [fr for fr, _ in gpu_frames]
            
            future = executor.submit(
                _process_single_gpu_batch,
                frames_only,
                enabled_gpus[gpu_idx],
                vol_handles,
                vol_locks,
                parent_dir
            )
            futures[future] = (gpu_idx, gpu_frames)
        
        # Collect results
        for future in as_completed(futures):
            gpu_idx, gpu_frames = futures[future]
            try:
                gpu_results = future.result()
                
                # gpu_results is List[Tuple[frame_id, bytes]]
                final_results.extend(gpu_results)
                
            except Exception as e:
                print(f"\n[ERRO] GPU {gpu_idx} batch failed: {e}")
                # Fallback: process these frames on CPU
                for fr, _ in gpu_frames:
                    try:
                        # Read and decompress on CPU
                        vol_name = fr["volume_name"]
                        if vol_name not in vol_handles:
                            vol_handles[vol_name] = open(parent_dir / vol_name, "rb")
                            vol_locks[vol_name] = threading.Lock()
                        
                        with vol_locks[vol_name]:
                            f = vol_handles[vol_name]
                            f.seek(fr["offset"])
                            c_data = f.read(fr["compressed_size"])
                        
                        u_data = decompress_lz4_ext3(c_data, fr["uncompressed_size"])
                        final_results.append((fr["frame_id"], u_data))
                    except Exception as cpu_e:
                        print(f"\n[ERRO] CPU fallback failed for frame {fr['frame_id']}: {cpu_e}")
    
    return final_results

def _process_single_gpu_batch(frames: List[Dict], gpu_decompressor: GPU_LZ4_Decompressor, 
                              vol_handles: Dict, vol_locks: Dict, parent_dir: Path) -> List[Tuple[int, bytes]]:
    """
    Process a batch of frames on a single GPU (worker function).
    """
    # Read compressed data for all frames
    compressed_data = []
    uncompressed_sizes = []
    
    for fr in frames:
        vol_name = fr["volume_name"]
        offset = fr["offset"]
        c_size = fr["compressed_size"]
        u_size = fr["uncompressed_size"]
        
        # Lazy load volume handle
        if vol_name not in vol_handles:
            vol_handles[vol_name] = open(parent_dir / vol_name, "rb")
            vol_locks[vol_name] = threading.Lock()
        
        # Thread-safe file read
        with vol_locks[vol_name]:
            f = vol_handles[vol_name]
            f.seek(offset)
            data = f.read(c_size)
        
        compressed_data.append(data)
        uncompressed_sizes.append(u_size)
    
    # Try GPU decompression
    gpu_results = gpu_decompressor.decompress_batch(compressed_data, uncompressed_sizes)
    
    # Process results with CPU fallback
    final_results = []
    for i, result in enumerate(gpu_results):
        fr = frames[i]
        if result is not None:
            # GPU success
            final_results.append((fr["frame_id"], result))
        else:
            # GPU failed - fallback to CPU
            try:
                u_data = decompress_lz4_ext3(compressed_data[i], uncompressed_sizes[i])
                final_results.append((fr["frame_id"], u_data))
            except Exception as e:
                print(f"\n[ERRO] CPU fallback failed for frame {fr['frame_id']}: {e}")
    
    return final_results

    """
    Process frames using GPU decompressor with automatic CPU fallback.
    
    Args:
        frames: List of frame metadata dicts
        vol_handles: Dict of open file handles for volumes
        vol_locks: Dict of threading locks for volume access
        parent_dir: Parent directory containing volume files
        gpu_decompressor: GPU_LZ4_Decompressor instance
        
    Returns:
        List of (frame_id, decompressed_data) tuples
    """
    # Read compressed data for all frames
    compressed_data = []
    uncompressed_sizes = []
    
    for fr in frames:
        vol_name = fr["volume_name"]
        offset = fr["offset"]
        c_size = fr["compressed_size"]
        u_size = fr["uncompressed_size"]
        
        # Lazy load volume handle
        if vol_name not in vol_handles:
            vol_handles[vol_name] = open(parent_dir / vol_name, "rb")
            vol_locks[vol_name] = threading.Lock()
        
        # Thread-safe file read
        with vol_locks[vol_name]:
            f = vol_handles[vol_name]
            f.seek(offset)
            data = f.read(c_size)
        
        compressed_data.append(data)
        uncompressed_sizes.append(u_size)
    
    # Try GPU decompression
    gpu_results = gpu_decompressor.decompress_batch(compressed_data, uncompressed_sizes)
    
    # Process results with CPU fallback
    final_results = []
    for i, result in enumerate(gpu_results):
        fr = frames[i]
        if result is not None:
            # GPU success
            final_results.append((fr["frame_id"], result))
        else:
            # GPU failed - fallback to CPU
            try:
                u_data = decompress_lz4_ext3(compressed_data[i], uncompressed_sizes[i])
                final_results.append((fr["frame_id"], u_data))
            except Exception as e:
                print(f"\n[ERRO] CPU fallback failed for frame {fr['frame_id']}: {e}")
                # Return zeros as last resort
                final_results.append((fr["frame_id"], b'\x00' * uncompressed_sizes[i]))
    
    return final_results

def process_cpu_frames(frames: List[Dict], vol_handles: Dict, vol_locks: Dict, parent_dir: Path, frame_modes: Dict[int, str]) -> List[Tuple[int, bytes]]:
    """
    Process frames using CPU decompression (for non-GPU modes or fallback).
    Uses parallel processing with 1 frame per logical CPU core.
    
    Args:
        frames: List of frame metadata dicts
        vol_handles: Dict of open file handles for volumes
        vol_locks: Dict of threading locks for volume access
        parent_dir: Parent directory containing volume files
        frame_modes: Dict mapping frame_id to compression mode
        
    Returns:
        List of (frame_id, decompressed_data) tuples
    """
    if not frames:
        return []
    
    import multiprocessing
    
    # Get CPU count for parallel processing
    cpu_count = multiprocessing.cpu_count()
    
    # Process frames in parallel
    from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
    
    results = []
    
    with TPE(max_workers=cpu_count) as executor:
        # Submit all frames for parallel processing
        futures = {}
        
        for fr in frames:
            future = executor.submit(
                _decompress_single_cpu_frame,
                fr,
                vol_handles,
                vol_locks,
                parent_dir,
                frame_modes
            )
            futures[future] = fr["frame_id"]
        
        # Collect results as they complete
        for future in as_completed(futures):
            frame_id = futures[future]
            try:
                u_data = future.result()
                results.append((frame_id, u_data))
            except Exception as e:
                print(f"\n[ERRO] CPU frame {frame_id} failed: {e}")
                # Return zeros as fallback
                fr = next(f for f in frames if f["frame_id"] == frame_id)
                results.append((frame_id, b'\x00' * fr["uncompressed_size"]))
    
    return results

def _decompress_single_cpu_frame(fr: Dict, vol_handles: Dict, vol_locks: Dict, 
                                  parent_dir: Path, frame_modes: Dict[int, str]) -> bytes:
    """
    Decompress a single frame on CPU (worker function for parallel processing).
    Thread-safe file access with locks.
    
    Args:
        fr: Frame metadata dict
        vol_handles: Dict of open file handles for volumes
        vol_locks: Dict of threading locks for volume access
        parent_dir: Parent directory containing volume files
        frame_modes: Dict mapping frame_id to compression mode
        
    Returns:
        Decompressed data as bytes
    """
    fid = fr["frame_id"]
    vol_name = fr["volume_name"]
    offset = fr["offset"]
    c_size = fr["compressed_size"]
    u_size = fr["uncompressed_size"]
    
    # Lazy load volume handle
    if vol_name not in vol_handles:
        vol_handles[vol_name] = open(parent_dir / vol_name, "rb")
        vol_locks[vol_name] = threading.Lock()
    
    # Thread-safe file read
    with vol_locks[vol_name]:
        f = vol_handles[vol_name]
        f.seek(offset)
        c_data = f.read(c_size)
    
    mode = frame_modes.get(fid, "lz_ext3_gpu")
    
    # Decompress based on mode
    if mode == "lz_ext3_gpu":
        u_data = decompress_lz4_ext3(c_data, u_size)
    elif mode in ["lz4_gpu", "lz4"]:
        u_data = lz4.block.decompress(c_data, uncompressed_size=u_size)
    elif mode == "raw":
        u_data = c_data
    else:
        print(f"\n[AVISO] Modo desconhecido {mode} frame {fid}, usando RAW")
        u_data = c_data
    
    return u_data

def process_batch(batch_frames: List[Dict], vol_handles: Dict, vol_locks: Dict, parent_dir: Path, frame_modes: Dict[int, str], 
                  gpu_decompressors: List[GPU_LZ4_Decompressor]) -> List[Tuple[int, bytes]]:
    """
    Process a batch of frames with GPU acceleration + CPU fallback.
    
    Args:
        batch_frames: List of frame metadata dicts
        vol_handles: Dict of open file handles for volumes
        vol_locks: Dict of threading locks for volume access
        parent_dir: Parent directory containing volume files
        frame_modes: Dict mapping frame_id to compression mode
        gpu_decompressors: List of GPU_LZ4_Decompressor instances
        
    Returns:
        List of (frame_id, decompressed_data) tuples sorted by frame_id
    """
    # Separate GPU-compatible frames
    gpu_frames = []
    cpu_frames = []
    
    for fr in batch_frames:
        mode = frame_modes.get(fr["frame_id"], "lz_ext3_gpu")
        has_enabled_gpu = any(d.enabled for d in gpu_decompressors)
        if mode == "lz_ext3_gpu" and has_enabled_gpu:
            gpu_frames.append(fr)
        else:
            cpu_frames.append(fr)
    
    results = []
    
    # GPU batch processing
    if gpu_frames:
        gpu_results = process_gpu_batch(gpu_frames, vol_handles, vol_locks, parent_dir, gpu_decompressors)
        results.extend(gpu_results)
    
    # CPU processing (includes GPU fallbacks)
    if cpu_frames:
        cpu_results = process_cpu_frames(cpu_frames, vol_handles, vol_locks, parent_dir, frame_modes)
        results.extend(cpu_results)
    
    # Sort by frame_id to maintain order
    results.sort(key=lambda x: x[0])
    return results

def main() -> int:
    parser = argparse.ArgumentParser(description="Descompressor LZ4 GPU (Embedded Index)")
    parser.add_argument("archive_base", help="Caminho de qualquer volume (ex: F:\\saida_final.001)")
    parser.add_argument("-o", "--output", required=True, help="Pasta de destino")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = Path(args.archive_base).resolve()
    parent_dir = base_path.parent
    name_stem = base_path.name
    
    # Descobrir prefixo (ex: saida_final)
    import re
    match = re.match(r"(.*)\.(\d{3})$", name_stem)
    if match:
        prefix = match.group(1)
    else:
        prefix = name_stem
        
    # Listar volumes
    volumes_files = sorted(parent_dir.glob(f"{prefix}.*"))
    volumes_files = [v for v in volumes_files if re.match(r".*\.\d{3}$", v.name)]
    
    if not volumes_files:
        print(f"Erro: Nenhum volume encontrado com prefixo '{prefix}' em {parent_dir}")
        return 1
        
    last_vol = volumes_files[-1]
    print(f"Volumes encontrados: {len(volumes_files)}. Último: {last_vol.name}")
    
    # Ler Footer do último volume
    # [Offset (8)] [Size (8)] [Magic (8)] = 24 bytes
    with open(last_vol, "rb") as f:
        f.seek(0, 2) # Fim
        file_size = f.tell()
        if file_size < 24:
            print("Erro: Arquivo muito pequeno para conter footer.")
            return 1
            
        f.seek(-24, 2)
        footer = f.read(24)
        
        offset, size, magic = struct.unpack('<QQ8s', footer)
        
        if magic != b'GPU_IDX1':
            print(f"Erro: Assinatura inválida no footer: {magic}")
            return 1
            
        print(f"Índice encontrado: Offset={offset}, Size={size}")
        
        f.seek(offset)
        compressed_index = f.read(size)
        
    try:
        index_bytes = zlib.decompress(compressed_index)
        index = json.loads(index_bytes.decode('utf-8'))
        print("Índice carregado e descomprimido com sucesso.")
    except Exception as e:
        print(f"Erro ao descomprimir/ler índice: {e}")
        return 1
        
    # Carregar metadados
    files = index["files"]
    frames = index["frames"]
    params = index["params"]
    
    # Mapa de frames por ID
    frames_map = {f["frame_id"]: f for f in frames}
    
    # Modos de compressão por frame
    frame_modes = params.get("frame_modes", {})
    # Converter chaves para int
    frame_modes = {int(k): v for k, v in frame_modes.items()}
    
    print(f"Restaurando {len(files)} arquivos para {output_dir}...")
    
    current_file_idx = 0
    current_file_pos = 0
    curr_fp = None
    
    # Cache de handles de volumes para evitar abrir/fechar repetidamente
    vol_handles = {}
    vol_locks = {}  # Locks for thread-safe file access
    
    def get_vol_handle(vol_name):
        if vol_name not in vol_handles:
            vol_handles[vol_name] = open(parent_dir / vol_name, "rb")
            vol_locks[vol_name] = threading.Lock()  # Create lock for this volume
        return vol_handles[vol_name]

    # Lista para processar duplicatas no final (garante que originais existam)
    duplicates = []

    # Função para escrever dados
    def write_data(data: bytes):
        nonlocal current_file_idx, current_file_pos, curr_fp
        
        data_pos = 0
        data_len = len(data)
        
        while data_pos < data_len:
            # Avançar arquivos já completados ou duplicatas
            while current_file_idx < len(files):
                curr_file_entry = files[current_file_idx]
                
                if curr_file_entry.get("is_duplicate"):
                    # Adiar duplicata para o final
                    duplicates.append(curr_file_entry)
                        
                    current_file_idx += 1
                    current_file_pos = 0
                    continue
                
                remaining = curr_file_entry["size"] - current_file_pos
                if remaining > 0:
                    break
                
                if curr_fp:
                    curr_fp.close()
                    curr_fp = None
                
                current_file_idx += 1
                current_file_pos = 0
            
            if current_file_idx >= len(files):
                break
                
            curr_file_entry = files[current_file_idx]
            
            if curr_fp is None:
                p = output_dir / curr_file_entry["path_rel"]
                
                # Verificar se algum componente do caminho existe como arquivo (conflito)
                try:
                    parent_path = p.parent
                    # Verificar todos os ancestrais até o output_dir
                    for ancestor in list(parent_path.parents) + [parent_path]:
                        # Parar ao chegar no output_dir
                        if ancestor == output_dir or ancestor in output_dir.parents:
                            break
                        # Verificar se existe como arquivo (conflito)
                        if ancestor.exists() and ancestor.is_file():
                            print(f"[ERRO] Conflito de caminho: '{ancestor}' existe como arquivo mas é necessário como diretório")
                            print(f"       Arquivo destino: {p}")
                            # Tentar renomear arquivo conflitante
                            conflicting_file = ancestor
                            backup_name = conflicting_file.with_suffix(conflicting_file.suffix + ".backup")
                            print(f"       Renomeando '{conflicting_file}' para '{backup_name}'")
                            conflicting_file.rename(backup_name)
                    
                    parent_path.mkdir(parents=True, exist_ok=True)
                except FileExistsError as e:
                    print(f"[ERRO] Não foi possível criar diretório '{parent_path}': {e}")
                    print(f"       Arquivo destino: {p}")
                    raise
                
                curr_fp = open(p, "wb")
            
            remaining = curr_file_entry["size"] - current_file_pos
            chunk_size = min(data_len - data_pos, remaining)
            
            curr_fp.write(data[data_pos : data_pos + chunk_size])
            
            current_file_pos += chunk_size
            data_pos += chunk_size
            
            if current_file_pos >= curr_file_entry["size"]:
                curr_fp.close()
                curr_fp = None
                current_file_idx += 1
                current_file_pos = 0


    # Initialize GPU decompressors (one per GPU)
    gpu_decompressors = []
    
    if GPU_LZ4_Decompressor is not None:
        try:
            import pyopencl as cl
            
            # Detect all available GPUs
            platforms = cl.get_platforms()
            if platforms:
                all_gpus = []
                gpu_indices_map = []  # Maps filtered index to global index
                global_idx = 0
                
                for p in platforms:
                    try:
                        platform_gpus = p.get_devices(device_type=cl.device_type.GPU)
                        for gpu in platform_gpus:
                            all_gpus.append(gpu)
                            gpu_indices_map.append(global_idx)
                            global_idx += 1
                    except:
                        pass
                
                if all_gpus:
                    # Prompt for GPU selection (only if selector available and multiple GPUs)
                    excluded_indices = []
                    if GPU_SELECTOR_AVAILABLE:
                        excluded_indices = prompt_gpu_selection()
                    
                    # Initialize one decompressor per enabled GPU
                    for idx, gpu_global_idx in enumerate(gpu_indices_map):
                        if gpu_global_idx in excluded_indices:
                            continue  # Skip excluded GPUs
                        
                        try:
                            decompressor = GPU_LZ4_Decompressor(device_index=gpu_global_idx)
                            if decompressor.enabled:
                                gpu_decompressors.append(decompressor)
                        except Exception as e:
                            print(f"[GPU_LZ4] Failed to initialize GPU {gpu_global_idx}: {e}")
            
            if not gpu_decompressors:
                print("[GPU_LZ4] No GPUs available, using CPU only")
                # No dummy decompressor needed, just empty list triggers CPU fallback
        except ImportError:
             print("[GPU_LZ4] PyOpenCL not found. Using CPU only.")
        except Exception as e:
            print(f"[GPU_LZ4] GPU detection failed: {e}")
    else:
        print("[GPU_LZ4] GPU Decompressor module not available. Using CPU only.")
    
    # Prepare frames and batches
    sorted_frames = sorted(frames, key=lambda x: x["frame_id"])
    batches = [sorted_frames[i:i+GPU_BATCH_SIZE] for i in range(0, len(sorted_frames), GPU_BATCH_SIZE)]
    
    total_frames = len(sorted_frames)
    start_time = time.time()
    bytes_written = 0
    frames_processed = 0
    
    # Statistics tracking
    gpu_frames_count = 0
    cpu_frames_count = 0
    gpu_fallback_count = 0
    
    print(f"\n[GPU_LZ4] GPU Decompressors initialized: {len([d for d in gpu_decompressors if d.enabled])}")
    print(f"[GPU_LZ4] Processing {total_frames} frames in {len(batches)} batches (size={GPU_BATCH_SIZE})")
    print(f"[GPU_LZ4] Worker threads: {MAX_WORKER_THREADS}\n")
    
    # Process batches with threading - CONTINUOUS PIPELINE
    # Instead of submitting all at once, maintain a constant queue of active batches
    MAX_PENDING_BATCHES = MAX_WORKER_THREADS * 2  # Keep 2x workers in queue for smooth flow
    
    with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
        batch_iter = iter(enumerate(batches))
        pending_futures = {}
        
        # Buffer de resultados para garantir ordem de escrita
        pending_results = {} # frame_id -> u_data
        next_frame_to_write = 0
        
        # Initial submission - fill the queue
        for _ in range(min(MAX_PENDING_BATCHES, len(batches))):
            try:
                batch_idx, batch = next(batch_iter)
                future = executor.submit(process_batch, batch, vol_handles, vol_locks, parent_dir, frame_modes, gpu_decompressors)
                pending_futures[future] = (batch_idx, batch)  # Store batch for fallback
            except StopIteration:
                break
        
        # Process results and submit new batches as old ones complete
        while pending_futures or pending_results:
            # Se não há futures pendentes, mas há resultados (caso final), não espera
            if pending_futures:
                # Wait for at least one to complete
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    batch_idx, batch = pending_futures.pop(future)
                    
                    try:
                        batch_results = future.result()
                        
                        # Armazenar resultados no buffer
                        for frame_id, u_data in batch_results:
                            pending_results[frame_id] = u_data
                            
                    except Exception as e:
                        print(f"\n[ERRO] Batch {batch_idx} failed: {e}. Reprocessing with CPU fallback...")
                        
                        # Fallback: Process this batch entirely on CPU
                        try:
                            cpu_results = process_cpu_frames(batch, vol_handles, vol_locks, parent_dir, frame_modes)
                            
                            # Armazenar resultados do fallback no buffer
                            for frame_id, u_data in cpu_results:
                                pending_results[frame_id] = u_data
                                
                        except Exception as cpu_e:
                            print(f"\n[ERRO CRÍTICO] CPU fallback também falhou para batch {batch_idx}: {cpu_e}")
                    
                    # Submit new batch to maintain queue depth
                    try:
                        new_batch_idx, new_batch = next(batch_iter)
                        new_future = executor.submit(process_batch, new_batch, vol_handles, vol_locks, parent_dir, frame_modes, gpu_decompressors)
                        pending_futures[new_future] = (new_batch_idx, new_batch)  # Store batch for fallback
                    except StopIteration:
                        # No more batches to submit
                        pass
            
            # Escrever frames na ordem correta (1, 2, 3...)
            while next_frame_to_write in pending_results:
                u_data = pending_results.pop(next_frame_to_write)
                
                write_data(u_data)
                bytes_written += len(u_data)
                frames_processed += 1
                
                # Update statistics (approximate, since we don't have mode here easily without lookup)
                # Recalculate mode for stats
                mode = frame_modes.get(next_frame_to_write, "lz_ext3_gpu")
                has_enabled_gpu = any(d.enabled for d in gpu_decompressors)
                if mode == "lz_ext3_gpu" and has_enabled_gpu:
                    gpu_frames_count += 1
                else:
                    cpu_frames_count += 1
                
                # Progress reporting
                if frames_processed % 10 == 0 or frames_processed == total_frames:
                    elapsed = time.time() - start_time
                    speed = (bytes_written / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    pending_count = len(pending_futures)
                    print(f"Progresso: {frames_processed}/{total_frames} frames | {speed:.1f} MB/s | GPU: {gpu_frames_count} | CPU: {cpu_frames_count} | Fila: {pending_count}", end="\r")
                
                next_frame_to_write += 1
            
            # Se não tem mais futures e o buffer esvaziou, acabou
            if not pending_futures and not pending_results:
                break
            
       # Fechar handles de volumes
    for f in vol_handles.values():
        f.close()

    # ------------------------------------------------------------------
    # Recriar arquivos duplicados (reverso da deduplicação)
    # ------------------------------------------------------------------
    if duplicates:
        print(f"\nCriando {len(duplicates)} arquivos duplicados (dedup reverso)...")
        created = 0

        for dup in duplicates:
            dup_rel = dup.get("path_rel")
            orig_rel = dup.get("original")

            if not dup_rel or not orig_rel:
                # Entrada estranha/incompleta, pula
                print(f"[AVISO] Entrada duplicada sem 'path_rel' ou 'original': {dup}")
                continue

            src_path = output_dir / orig_rel
            dst_path = output_dir / dup_rel

            try:
                # Garante que a pasta do destino exista
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                # Tenta criar hardlink (mais leve em disco).
                # Se não funcionar (SO, permissões, volumes diferentes), cai pra cópia.
                try:
                    import os
                    if dst_path.exists():
                        dst_path.unlink()
                    os.link(src_path, dst_path)
                except Exception:
                    shutil.copy2(src_path, dst_path)

                created += 1
            except Exception as e:
                print(f"[ERRO] Falha ao criar duplicata '{dup_rel}' a partir de '{orig_rel}': {e}")

        print(f"[Duplicatas] {created}/{len(duplicates)} arquivos duplicados recriados.")
    else:
        print("\nNenhuma duplicata registrada no índice (duplicates vazio).")

    # Final statistics
    total_time = time.time() - start_time
    avg_speed = (bytes_written / 1024 / 1024) / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Descompressão concluída!")
    print(f"{'='*60}")
    print(f"Tempo total:      {total_time:.1f}s")
    print(f"Frames totais:    {total_frames}")
    print(f"  - GPU:          {gpu_frames_count} ({100*gpu_frames_count/total_frames:.1f}%)")
    print(f"  - CPU:          {cpu_frames_count} ({100*cpu_frames_count/total_frames:.1f}%)")
    print(f"Dados escritos:   {bytes_written / 1024 / 1024:.1f} MB")
    print(f"Velocidade média: {avg_speed:.1f} MB/s")
    print(f"{'='*60}\n")
    
    return 0

    
    

if __name__ == "__main__":
    raise SystemExit(main())
