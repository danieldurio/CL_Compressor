"""
Compressor Otimizado com Deduplicação GPU e LZ4 GPU.

Pipeline Simplificado:
1. Deduplicação GPU: Identifica e remove arquivos idênticos.
2. Compressão LZ4 GPU: Alta performance para dados compressíveis.
3. Fallback RAW: Para dados incompressíveis.

Uso:
    python compressor_lz4_dedup.py F:\\Radio -o F:\\saida_final --frame-size-mb 1
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import time

from iotools import (
    FileEntry, VolumeWriter, generate_frames, estimate_total_size,
    scan_directory, FrameMeta
)
from deduplicator import GPUFileDeduplicator
from gpu_lz4_compressor import GPU_LZ4_Compressor

def compress_directory_lz4(
    entries: Iterable[FileEntry],
    frame_size: int,
    max_volume_size: int,
    output_base: Path,
    index_params: Dict[str, Any],
    total_size_to_compress: int = 0,
) -> List[FrameMeta]:
    """
    Compressor focado em LZ4 GPU e Deduplicação.
    Suporta Múltiplas GPUs para paralelismo.
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
        print("[Compressor] Nenhuma GPU detectada para compressão!")
    else:
        print(f"[Compressor] Detectadas {len(devices)} GPUs. Inicializando compressores...")
        for i in range(len(devices)):
            # Passar frame_size para alocação de buffers persistentes
            comp = GPU_LZ4_Compressor(device_index=i, max_input_size=frame_size)
            if comp.enabled:
                compressors.append(comp)
            else:
                print(f"[Compressor] Falha ao inicializar compressor na GPU {i}")

    if not compressors:
        print("[Compressor] AVISO: Nenhum compressor GPU disponível. Fallback para RAW (sem compressão).")

    # Fila de compressores disponíveis para os workers
    import queue
    from concurrent.futures import ThreadPoolExecutor, wait
    
    compressor_queue = queue.Queue()
    for c in compressors:
        compressor_queue.put(c)
        
    print(f"[Compressor] Iniciando compressão com {len(compressors)} workers GPU...")
    
    index_params = dict(index_params)
    index_params["compression_mode"] = "lz_ext3_gpu"
    
    frame_modes: Dict[int, str] = {}
    
    # Stats accumulators
    vol_stats = {
        "orig": 0,
        "lz_ext3_gpu": 0,
        "raw": 0,
        "hits": {"lz_ext3_gpu": 0, "raw": 0}
    }
    # Estatísticas GLOBAIS (acumuladas - persistem entre volumes)
    global_stats = {
        "lz_ext3_gpu": 0,
        "raw": 0,
        "total_orig_bytes": 0,
        "total_compressed_bytes": 0
    }
    gpu_frame_counts = [0] * len(compressors) if compressors else [0]
    gpu_last_frames = [-1] * len(compressors) if compressors else [-1]
    current_vol_name = None  # Para rastrear mudanças de volume

    def print_vol_stats(vol_name, stats):
        """Imprime estatísticas globais acumuladas"""
        if not vol_name: return
        
        # Calcular totais globais
        total_compressed_count = global_stats["lz_ext3_gpu"] + global_stats["raw"]
        if total_compressed_count == 0:
            return
            
        lz4_pct = (global_stats["lz_ext3_gpu"] / total_compressed_count) * 100
        raw_pct = (global_stats["raw"] / total_compressed_count) * 100
        
        total_orig_mb = global_stats["total_orig_bytes"] / 1024 / 1024
        total_comp_mb = global_stats["total_compressed_bytes"] / 1024 / 1024
        
        reduction_pct = 0
        if global_stats["total_orig_bytes"] > 0:
            reduction_pct = (1 - (global_stats["total_compressed_bytes"] / global_stats["total_orig_bytes"])) * 100
            
        progress_pct = 0
        if total_size_to_compress > 0:
            progress_pct = (global_stats["total_orig_bytes"] / total_size_to_compress) * 100
        
        # Linha 1: Compressor stats + Redução
        print(
            f"[Compressor] | "
            f"LZ_EXT3_GPU={global_stats['lz_ext3_gpu']} ({lz4_pct:.2f}%) | "
            f"RAW={global_stats['raw']} ({raw_pct:.2f}%) | "
            f"Redução = {reduction_pct:.1f}%"
        )
        
        # Linha 2: Estatísticas por GPU + Tamanhos
        if compressors and len(compressors) >= 1:
            gpu_work_str = " | ".join([f"GPU{i+1} = {gpu_frame_counts[i]}" for i in range(len(compressors))])
            print(f"HIT: {gpu_work_str} | Tamanho real = {total_orig_mb:.0f}MB | Tamanho atual = {total_comp_mb:.0f}MB")
            
            # Linha 3: Último frame + Progresso
            gpu_last_str = " | ".join([f"GPU{i+1} = {gpu_last_frames[i]}" for i in range(len(compressors))])
            print(f"Last Frame - {gpu_last_str} | Progresso atual = {progress_pct:.1f}%")

    def reset_stats(stats):
        stats["orig"] = 0
        stats["lz_ext3_gpu"] = 0
        stats["raw"] = 0
        for k in stats["hits"]:
            stats["hits"][k] = 0

    # Função Worker para compressão paralela (BATCH)
    def compress_batch_task(batch_data):
        # batch_data é uma lista de tuplas (frame_id, frame_bytes)
        frames = [b[1] for b in batch_data]
        frame_ids = [b[0] for b in batch_data]
        
        # Resultado padrão (RAW) para todos
        results = []
        
        if not compressors:
             # Fallback RAW imediato
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

        # Adquirir compressor
        comp = compressor_queue.get()
        gpu_idx = comp.device_index
        
        try:
            # Executar compressão em batch na GPU
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
                    
                    # Log periódico
                    if fid % 100 == 0:
                        ratio = (res_size / orig_size) * 100
                        saved = orig_size - res_size
                        print(f"[LZ4] Frame {fid} GPU{gpu_idx+1}: {orig_size} -> {res_size} ({ratio:.1f}%) | Economia: {saved}")
                
                results.append(res)
                
        except Exception as e:
            print(f"[Compressor] Erro Batch GPU {gpu_idx}: {e}")
            # Fallback RAW para todo o batch em caso de erro
            results = []
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

    # Execução Paralela com Batching
    num_workers = len(compressors) if compressors else 1
    BATCH_SIZE = 24 # Tamanho do lote para GPU
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {} # batch_id -> Future
        next_frame_to_write = 0
        
        current_batch = []
        batch_counter = 0
        
        # Buffer de resultados pendentes para escrita ordenada
        pending_results = {} # frame_id -> result_dict
        
        for frame_id, frame_data in generate_frames(entries, frame_size):
            if len(frame_data) == 0: continue
            
            current_batch.append((frame_id, frame_data))
            
            if len(current_batch) >= BATCH_SIZE:
                # Submeter batch
                f = executor.submit(compress_batch_task, list(current_batch))
                futures[batch_counter] = f
                batch_counter += 1
                current_batch = []
                
                # Controle de Backpressure (limitar batches em voo)
                # Manter ~2 batches por GPU para evitar OOM com arquivos incompressíveis
                max_pending_batches = num_workers * 2
                while len(futures) > max_pending_batches:
                    # Remover futures concluídos
                    done_batches = [bid for bid, fut in futures.items() if fut.done()]
                    for bid in done_batches:
                        batch_res = futures[bid].result()
                        del futures[bid]
                        for res in batch_res:
                            pending_results[res["frame_id"]] = res
                    
                    if not done_batches:
                        time.sleep(0.01)
            
            # Forçar escrita periódica para evitar acúmulo excessivo em RAM
            # Processar resultados pendentes a cada 20 frames acumulados
            if len(pending_results) > 20:
                while next_frame_to_write in pending_results:
                    res = pending_results.pop(next_frame_to_write)
                    
                    # Processar resultado (escrita e stats) - Lógica idêntica à anterior
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
                    
                    # Validação para prevenir overflow (C long limit) - 2GB para segurança  
                    res_size_value = res["size"]
                    if isinstance(res_size_value, int) and res_size_value > 0 and res_size_value < (2 * 1024 * 1024 * 1024):
                        try:
                            global_stats["total_compressed_bytes"] += res_size_value
                        except OverflowError as e:
                            print(f"[ERRO] Overflow ao somar estatísticas no frame {res['frame_id']}: total={global_stats['total_compressed_bytes']}, adding={res_size_value}")
                            # Resetar estatística para continuar sem crash
                            global_stats["total_compressed_bytes"] = res_size_value
                    else:
                        if res_size_value >= (2 * 1024 * 1024 * 1024):
                            print(f"[AVISO] Frame {res['frame_id']}: tamanho excessivo {res_size_value} bytes (>2GB), ignorando estatística")
                    
                    meta = writer.write_frame(
                        frame_id=res["frame_id"],
                        uncompressed_size=orig_size,
                        compressed_bytes=best_bytes,
                    )
                    
                    if current_vol_name is None:
                        current_vol_name = meta.volume_name
                        
                    if meta.volume_name != current_vol_name:
                        print_vol_stats(current_vol_name, vol_stats)
                        reset_stats(vol_stats)
                        current_vol_name = meta.volume_name
                        
                    vol_stats["orig"] += orig_size
                    if best_mode == "lz_ext3_gpu":
                        vol_stats["lz_ext3_gpu"] += res["size"]
                    else:
                        vol_stats["raw"] += res["size"]
                    vol_stats["hits"][best_mode] += 1
                    
                    all_frames.append(meta)
                    frame_modes[res["frame_id"]] = best_mode
                    
                    next_frame_to_write += 1


        # Submeter último batch parcial
        if current_batch:
            f = executor.submit(compress_batch_task, list(current_batch))
            futures[batch_counter] = f
            batch_counter += 1
        
        # Aguardar todos os batches
        while futures or pending_results:
            # Coletar futures prontos
            done_batches = [bid for bid, fut in futures.items() if fut.done()]
            for bid in done_batches:
                batch_res = futures[bid].result()
                del futures[bid]
                for res in batch_res:
                    pending_results[res["frame_id"]] = res
            
            # Escrever resultados ordenados
            while next_frame_to_write in pending_results:
                res = pending_results.pop(next_frame_to_write)
                
                # Processar resultado (escrita e stats) - Cópia da lógica acima
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
                    
                # Validação para prevenir overflow (C long limit) - 2GB para segurança
                res_size_value = res["size"]
                if isinstance(res_size_value, int) and res_size_value > 0 and res_size_value < (2 * 1024 * 1024 * 1024):
                    try:
                        global_stats["total_compressed_bytes"] += res_size_value
                    except OverflowError as e:
                        print(f"[ERRO] Overflow ao somar: current={global_stats['total_compressed_bytes']}, adding={res_size_value}")
                        # Resetar estatística para continuar
                        global_stats["total_compressed_bytes"] = res_size_value
                else:
                    print(f"[Bloco] Frame {res['frame_id']}: tamanho ao limite {res_size_value}, capturando o raw")
                
                meta = writer.write_frame(
                    frame_id=res["frame_id"],
                    uncompressed_size=orig_size,
                    compressed_bytes=best_bytes,
                )
                
                if current_vol_name is None:
                    current_vol_name = meta.volume_name
                    
                if meta.volume_name != current_vol_name:
                    print_vol_stats(current_vol_name, vol_stats)
                    reset_stats(vol_stats)
                    current_vol_name = meta.volume_name
                    
                vol_stats["orig"] += orig_size
                if best_mode == "lz_ext3_gpu":
                    vol_stats["lz_ext3_gpu"] += res["size"]
                else:
                    vol_stats["raw"] += res["size"]
                vol_stats["hits"][best_mode] += 1
                
                all_frames.append(meta)
                frame_modes[res["frame_id"]] = best_mode
                
                next_frame_to_write += 1
            
            if not futures and not pending_results:
                break
            
            if not done_batches:
                time.sleep(0.01)
    
    # Imprimir stats do último volume
    if current_vol_name:
        print_vol_stats(current_vol_name, vol_stats)
    
    writer.close()
    index_params["frame_modes"] = frame_modes
    
    # Embed index file
    from iotools import embed_index_file
    last_vol = all_frames[-1].volume_name if all_frames else ""
    
    embed_index_file(
        output_base=output_base,
        last_volume_name=last_vol,
        files=list(entries),
        frames=all_frames,
        dictionary=None, # Sem dicionário
        params=index_params
    )
    
    return all_frames

def main() -> int:
    parser = argparse.ArgumentParser(description="Compressor LZ4 GPU + Deduplicação")
    parser.add_argument("source", help="Pasta de origem")
    parser.add_argument("-o", "--output", default="archive_lz4.gpu", help="Base de saída")
    parser.add_argument("--frame-size-mb", type=int, default=16, help="Tamanho do frame (MB) - recomendado: 16-32 para janela de 16MB")
    parser.add_argument("--volume-size-mb", type=int, default=98, help="Tamanho do volume (MB)")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    if not source_dir.is_dir():
        print(f"Erro: '{source_dir}' não é um diretório.")
        return 1
    
    output_base = Path(args.output).resolve()
    frame_size = args.frame_size_mb * 1024 * 1024
    max_volume_size = args.volume_size_mb * 1024 * 1024
    
    print("="*70)
    print("COMPRESSOR LZ4 GPU + DEDUP")
    print("="*70)
    print(f"Fonte: {source_dir}")
    print(f"Saída: {output_base}")
    print("="*70)
    
    # 1. Scan
    entries = scan_directory(source_dir)
    if not entries:
        print("Nenhum arquivo encontrado.")
        return 0
    
    total_size_orig = estimate_total_size(entries)
    print(f"Arquivos: {len(entries)} ({total_size_orig} bytes)")
    
    # 2. Deduplicação GPU
    print("\n[Fase 1] Deduplicação GPU...")
    deduplicator = GPUFileDeduplicator()
    entries = deduplicator.find_duplicates(entries)
    
    unique_entries = [e for e in entries if not e.is_duplicate]
    total_size_dedup = estimate_total_size(unique_entries)
    dedup_saved = total_size_orig - total_size_dedup
    
    print(f"Tamanho efetivo: {total_size_dedup} bytes")
    if dedup_saved > 0:
        print(f"Economia Dedup: {dedup_saved / 1024 / 1024:.2f} MB ({(dedup_saved/total_size_orig)*100:.1f}%)")
    
    # 3. Compressão
    print("\n[Fase 2] Compressão LZ4 GPU...")
    print(f"[Compressor] Parâmetros: frame_size={args.frame_size_mb}MB, max_volume={args.volume_size_mb}MB")
    
    index_params = {
        "frame_size": frame_size,
        "max_volume_size": max_volume_size,
        "source_root": str(source_dir),
        "deduplication": True,
        "dedup_saved_bytes": dedup_saved,
        "compressor": "lz4_gpu"
    }
    
    compress_directory_lz4(
        entries=entries,
        frame_size=frame_size,
        max_volume_size=max_volume_size,
        output_base=output_base,
        index_params=index_params,
        total_size_to_compress=total_size_dedup,
    )
    
    print("\n" + "="*70)
    print("✅ Processo Completo Finalizado!")
    print("="*70)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
