"""
Compressor LZ4 GPU para Streams (stdin/stdout).

Pipeline Simplificado para Streams:
1. Leitura de stdin em chunks (frames)
2. Compressão LZ4 GPU: Alta performance
3. Fallback RAW: Para dados incompressíveis
4. Escrita de volumes em disco

Uso:
    cat largefile.bin | python compressor_lz4_stream.py -o output.gpu
    tar cf - /data | python compressor_lz4_stream.py -o backup.gpu --frame-size-mb 16
    
Futuro:
    cat file | python compressor_lz4_stream.py --stdout > output.gpu.001
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator
import time

from iotools import VolumeWriter, FrameMeta
from gpu_lz4_compressor import GPU_LZ4_Compressor
from gpu_capabilities import get_recommended_batch_size

# ============================================================
# CONFIGURAÇÃO
# ============================================================
BATCH_SIZE_OVERRIDE = None  # None = automático
STDIN_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB para leitura

def read_stdin_frames(frame_size: int) -> Iterator[tuple[int, bytes]]:
    """
    Lê dados de stdin e gera frames numerados.
    
    Args:
        frame_size: Tamanho de cada frame em bytes
        
    Yields:
        tuple[int, bytes]: (frame_id, frame_data)
    """
    # Abrir stdin em modo binário
    if sys.stdin.isatty():
        print("[Stream] ERRO: Nenhum dado em stdin. Use pipes ou redirecionamento.", file=sys.stderr)
        print("Exemplo: cat file | python compressor_lz4_stream.py -o output.gpu", file=sys.stderr)
        return
    
    # Reabrir stdin em modo binário no Windows
    import os
    if os.name == 'nt':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    
    stdin_binary = sys.stdin.buffer
    
    frame_id = 0
    total_read = 0
    
    print("[Stream] Lendo dados de stdin...", file=sys.stderr)
    
    while True:
        # Ler um frame completo
        frame_data = stdin_binary.read(frame_size)
        
        if not frame_data:
            break
        
        total_read += len(frame_data)
        
        # Log periódico
        if frame_id % 50 == 0:
            mb_read = total_read / (1024 * 1024)
            print(f"[Stream] Frame {frame_id}: {mb_read:.1f} MB lidos", file=sys.stderr)
        
        yield frame_id, frame_data
        frame_id += 1
    
    print(f"[Stream] Leitura concluída: {frame_id} frames, {total_read / (1024*1024):.2f} MB", file=sys.stderr)

def compress_stream_lz4(
    frame_size: int,
    max_volume_size: int,
    output_base: Path,
) -> List[FrameMeta]:
    """
    Compressor de stream focado em LZ4 GPU.
    Lê de stdin, comprime e escreve volumes.
    """
    writer = VolumeWriter(output_base, max_volume_size)
    all_frames: List[FrameMeta] = []
    
    # 1. Detectar GPUs e Inicializar Compressores
    import pyopencl as cl
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        devices.extend(p.get_devices(device_type=cl.device_type.GPU))
    
    compressors = []
    if not devices:
        print("[Compressor] Nenhuma GPU detectada para compressão!", file=sys.stderr)
    else:
        print(f"[Compressor] Detectadas {len(devices)} GPUs. Inicializando compressores...", file=sys.stderr)
        for i in range(len(devices)):
            comp = GPU_LZ4_Compressor(device_index=i, max_input_size=frame_size)
            if comp.enabled:
                compressors.append(comp)
            else:
                print(f"[Compressor] Falha ao inicializar compressor na GPU {i}", file=sys.stderr)

    if not compressors:
        print("[Compressor] AVISO: Nenhum compressor GPU disponível. Fallback para RAW.", file=sys.stderr)
    
    # Fila de compressores
    import queue
    from concurrent.futures import ThreadPoolExecutor
    
    compressor_queue = queue.Queue()
    for c in compressors:
        compressor_queue.put(c)
    
    print(f"[Compressor] Iniciando compressão com {len(compressors)} workers GPU...", file=sys.stderr)
    
    frame_modes: Dict[int, str] = {}
    
    # Stats
    global_stats = {
        "lz_ext3_gpu": 0,
        "raw": 0,
        "total_orig_bytes": 0,
        "total_compressed_bytes": 0
    }
    gpu_frame_counts = [0] * len(compressors) if compressors else [0]
    gpu_last_frames = [-1] * len(compressors) if compressors else [-1]
    current_vol_name = None
    
    def print_stats():
        """Imprime estatísticas acumuladas"""
        if global_stats["total_orig_bytes"] == 0:
            return
        
        total_compressed = global_stats["lz_ext3_gpu"] + global_stats["raw"]
        if total_compressed == 0:
            return
        
        lz4_pct = (global_stats["lz_ext3_gpu"] / total_compressed) * 100
        raw_pct = (global_stats["raw"] / total_compressed) * 100
        
        total_orig_mb = global_stats["total_orig_bytes"] / 1024 / 1024
        total_comp_mb = global_stats["total_compressed_bytes"] / 1024 / 1024
        
        reduction_pct = 0
        if global_stats["total_orig_bytes"] > 0:
            reduction_pct = (1 - (global_stats["total_compressed_bytes"] / global_stats["total_orig_bytes"])) * 100
        
        print(
            f"[Compressor] | "
            f"LZ_EXT3_GPU={global_stats['lz_ext3_gpu']} ({lz4_pct:.2f}%) | "
            f"RAW={global_stats['raw']} ({raw_pct:.2f}%) | "
            f"Redução = {reduction_pct:.1f}%",
            file=sys.stderr
        )
        
        if compressors and len(compressors) >= 1:
            gpu_work_str = " | ".join([f"GPU{i+1} = {gpu_frame_counts[i]}" for i in range(len(compressors))])
            print(f"HIT: {gpu_work_str} | Original = {total_orig_mb:.0f}MB | Comprimido = {total_comp_mb:.0f}MB", file=sys.stderr)
    
    # Worker de compressão em batch
    def compress_batch_task(batch_data):
        frames = [b[1] for b in batch_data]
        frame_ids = [b[0] for b in batch_data]
        
        results = []
        
        if not compressors:
            # Fallback RAW
            for fid, data in batch_data:
                results.append({
                    "mode": "raw",
                    "bytes": data,
                    "size": len(data),
                    "frame_id": fid,
                    "orig_size": len(data),
                    "gpu_index": -1
                })
            return results
        
        comp = compressor_queue.get()
        gpu_idx = comp.device_index
        
        try:
            batch_results = comp.compress_batch(frames, frame_ids)
            
            for i, (res_bytes, res_size, _) in enumerate(batch_results):
                orig_data = frames[i]
                orig_size = len(orig_data)
                fid = frame_ids[i]
                
                res = {
                    "mode": "raw",
                    "bytes": orig_data,
                    "size": orig_size,
                    "frame_id": fid,
                    "orig_size": orig_size,
                    "gpu_index": gpu_idx
                }
                
                if res_size < orig_size and res_size > 0:
                    res["mode"] = "lz_ext3_gpu"
                    res["bytes"] = res_bytes
                    res["size"] = res_size
                
                results.append(res)
                
        except Exception as e:
            print(f"[Compressor] Erro Batch GPU {gpu_idx}: {e}", file=sys.stderr)
            # Fallback RAW
            for fid, data in batch_data:
                results.append({
                    "mode": "raw",
                    "bytes": data,
                    "size": len(data),
                    "frame_id": fid,
                    "orig_size": len(data),
                    "gpu_index": gpu_idx
                })
        finally:
            compressor_queue.put(comp)
        
        return results
    
    # Execução paralela com batching
    num_workers = len(compressors) if compressors else 1
    
    # Determinar batch size
    if BATCH_SIZE_OVERRIDE is not None:
        BATCH_SIZE = BATCH_SIZE_OVERRIDE
        print(f"[Compressor] Batch Size FIXO: {BATCH_SIZE} frames", file=sys.stderr)
    else:
        frame_size_mb = frame_size // (1024 * 1024)
        BATCH_SIZE = get_recommended_batch_size(frame_size_mb=frame_size_mb)
        print(f"[Compressor] Batch Size AUTOMÁTICO: {BATCH_SIZE} frames", file=sys.stderr)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        next_frame_to_write = 0
        
        current_batch = []
        batch_counter = 0
        pending_results = {}
        
        for frame_id, frame_data in read_stdin_frames(frame_size):
            if len(frame_data) == 0:
                continue
            
            current_batch.append((frame_id, frame_data))
            
            if len(current_batch) >= BATCH_SIZE:
                f = executor.submit(compress_batch_task, list(current_batch))
                futures[batch_counter] = f
                batch_counter += 1
                current_batch = []
                
                # Backpressure
                max_pending_batches = num_workers * 2
                while len(futures) > max_pending_batches:
                    done_batches = [bid for bid, fut in futures.items() if fut.done()]
                    for bid in done_batches:
                        batch_res = futures[bid].result()
                        del futures[bid]
                        for res in batch_res:
                            pending_results[res["frame_id"]] = res
                    
                    if not done_batches:
                        time.sleep(0.01)
            
            # Escrita periódica
            if len(pending_results) > 20:
                while next_frame_to_write in pending_results:
                    res = pending_results.pop(next_frame_to_write)
                    
                    # Processar resultado
                    best_mode = res["mode"]
                    best_bytes = res["bytes"]
                    orig_size = res["orig_size"]
                    gpu_idx = res.get("gpu_index", -1)
                    
                    if gpu_idx >= 0 and gpu_idx < len(gpu_frame_counts):
                        gpu_frame_counts[gpu_idx] += 1
                        gpu_last_frames[gpu_idx] = res["frame_id"]
                    
                    if best_mode == "lz_ext3_gpu":
                        global_stats["lz_ext3_gpu"] += 1
                    else:
                        global_stats["raw"] += 1
                    
                    global_stats["total_orig_bytes"] += orig_size
                    global_stats["total_compressed_bytes"] += res["size"]
                    
                    meta = writer.write_frame(
                        frame_id=res["frame_id"],
                        uncompressed_size=orig_size,
                        compressed_bytes=best_bytes,
                    )
                    
                    if current_vol_name is None:
                        current_vol_name = meta.volume_name
                    
                    if meta.volume_name != current_vol_name:
                        print_stats()
                        current_vol_name = meta.volume_name
                    
                    all_frames.append(meta)
                    frame_modes[res["frame_id"]] = best_mode
                    
                    next_frame_to_write += 1
        
        # Submeter último batch
        if current_batch:
            f = executor.submit(compress_batch_task, list(current_batch))
            futures[batch_counter] = f
        
        # Aguardar todos os batches
        while futures or pending_results:
            done_batches = [bid for bid, fut in futures.items() if fut.done()]
            for bid in done_batches:
                batch_res = futures[bid].result()
                del futures[bid]
                for res in batch_res:
                    pending_results[res["frame_id"]] = res
            
            while next_frame_to_write in pending_results:
                res = pending_results.pop(next_frame_to_write)
                
                best_mode = res["mode"]
                best_bytes = res["bytes"]
                orig_size = res["orig_size"]
                gpu_idx = res.get("gpu_index", -1)
                
                if gpu_idx >= 0 and gpu_idx < len(gpu_frame_counts):
                    gpu_frame_counts[gpu_idx] += 1
                    gpu_last_frames[gpu_idx] = res["frame_id"]
                
                if best_mode == "lz_ext3_gpu":
                    global_stats["lz_ext3_gpu"] += 1
                else:
                    global_stats["raw"] += 1
                
                global_stats["total_orig_bytes"] += orig_size
                global_stats["total_compressed_bytes"] += res["size"]
                
                meta = writer.write_frame(
                    frame_id=res["frame_id"],
                    uncompressed_size=orig_size,
                    compressed_bytes=best_bytes,
                )
                
                if current_vol_name is None:
                    current_vol_name = meta.volume_name
                
                if meta.volume_name != current_vol_name:
                    print_stats()
                    current_vol_name = meta.volume_name
                
                all_frames.append(meta)
                frame_modes[res["frame_id"]] = best_mode
                
                next_frame_to_write += 1
            
            if not futures and not pending_results:
                break
            
            if not done_batches:
                time.sleep(0.01)
    
    # Stats finais
    if current_vol_name:
        print_stats()
    
    writer.close()
    
    # Criar índice simplificado para stream
    from iotools import embed_index_file
    
    # Criar FileEntry fictício para o stream
    class StreamFileEntry:
        def __init__(self, size):
            self.path = Path("<stdin>")
            self.size = size
            self.offset = 0
            self.num_frames = len(all_frames)
            self.is_duplicate = False
    
    stream_entry = StreamFileEntry(global_stats["total_orig_bytes"])
    
    last_vol = all_frames[-1].volume_name if all_frames else ""
    
    index_params = {
        "frame_size": frame_size,
        "max_volume_size": max_volume_size,
        "source": "stdin",
        "compression_mode": "lz_ext3_gpu",
        "frame_modes": frame_modes,
        "compressor": "lz4_gpu_stream"
    }
    
    embed_index_file(
        output_base=output_base,
        last_volume_name=last_vol,
        files=[stream_entry],
        frames=all_frames,
        dictionary=None,
        params=index_params
    )
    
    return all_frames

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compressor LZ4 GPU para Streams (stdin)",
        epilog="Exemplo: cat file.bin | python compressor_lz4_stream.py -o output.gpu"
    )
    parser.add_argument("-o", "--output", required=True, help="Base de saída para volumes")
    parser.add_argument("--frame-size-mb", type=int, default=16, help="Tamanho do frame (MB)")
    parser.add_argument("--volume-size-mb", type=int, default=98, help="Tamanho do volume (MB)")
    
    args = parser.parse_args()
    
    output_base = Path(args.output).resolve()
    frame_size = args.frame_size_mb * 1024 * 1024
    max_volume_size = args.volume_size_mb * 1024 * 1024
    
    print("=" * 70, file=sys.stderr)
    print("COMPRESSOR LZ4 GPU STREAM", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Fonte: stdin (stream)", file=sys.stderr)
    print(f"Saída: {output_base}", file=sys.stderr)
    print(f"Frame: {args.frame_size_mb}MB | Volume: {args.volume_size_mb}MB", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    compress_stream_lz4(
        frame_size=frame_size,
        max_volume_size=max_volume_size,
        output_base=output_base,
    )
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("✅ Compressão de Stream Finalizada!", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
