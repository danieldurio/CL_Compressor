"""
Compressor Otimizado com Deduplicação GPU e LZ4 GPU.

Pipeline Simplificado:
1. Deduplicação GPU: Identifica e remove arquivos idênticos.
2. Compressão LZ4 GPU: Alta performance para dados compressíveis.
3. Fallback RAW: Para dados incompressíveis.

Uso:
    python compressor_lz4_dedup.py F:\\Radio -o F:\\saida_final 
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

# Carregar configurações do config.txt centralizado
import config_loader

# Tentar importar GPU compressor, fallback para CPU se OpenCL não disponível
OPENCL_AVAILABLE = False
GPU_LZ4_Compressor = None

# Flag global para forçar modo CPU (via --cpu ou config.txt)
FORCE_CPU_MODE = config_loader.is_force_cpu_mode()

try:
    import pyopencl as cl
    from gpu_lz4_compressor import GPU_LZ4_Compressor
    from gpu_capabilities import get_recommended_batch_size
    from gpu_selector import prompt_gpu_selection, get_filtered_devices, get_excluded_indices
    OPENCL_AVAILABLE = True
except ImportError as e:
    print(f"[Compressor] OpenCL não disponível: {e}")
    print("[Compressor] Usando fallback CPU para compressão LZ4.")
    cl = None

# CPU Fallback compressor (sempre disponível)
from compressor_fallback import CPU_LZ4_Compressor

# ============================================================
# CONFIGURAÇÃO (carregada de config.txt)
# ============================================================
BATCH_SIZE_OVERRIDE = config_loader.get_compressor_batch_size()
READ_BUFFER_BATCHES = config_loader.get_read_buffer_batches()
WRITE_BUFFER_BATCHES = config_loader.get_write_buffer_batches()
# ============================================================

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
    
    # 0. Determinar batch size e modo de operação
    USE_CPU_FALLBACK = False
    BATCH_SIZE = BATCH_SIZE_OVERRIDE if BATCH_SIZE_OVERRIDE is not None else 24  # Default
    
    # Verificar se modo CPU foi forçado via --cpu
    if FORCE_CPU_MODE:
        USE_CPU_FALLBACK = True
        import multiprocessing
        BATCH_SIZE = multiprocessing.cpu_count() * 2
        print(f"[Compressor] 🖥️ MODO CPU FORÇADO (--cpu): Batch Size = {BATCH_SIZE} frames ({multiprocessing.cpu_count()} CPUs)")
    elif not OPENCL_AVAILABLE:
        # Sem OpenCL - usar CPU fallback
        USE_CPU_FALLBACK = True
        import multiprocessing
        BATCH_SIZE = multiprocessing.cpu_count() * 2  # Batch = 2x CPUs para melhor throughput
        print(f"[Compressor] Modo CPU (OpenCL indisponível): Batch Size = {BATCH_SIZE} frames ({multiprocessing.cpu_count()} CPUs)")
    elif BATCH_SIZE_OVERRIDE is not None:
        BATCH_SIZE = BATCH_SIZE_OVERRIDE
        print(f"[Compressor] Batch Size FIXO (definido pelo usuário): {BATCH_SIZE} frames")
    else:
        # Calcular batch size baseado nas capacidades da GPU
        frame_size_for_calc = frame_size // (1024 * 1024)  # Converter para MB
        BATCH_SIZE = get_recommended_batch_size(frame_size_mb=frame_size_for_calc)
        print(f"[Compressor] Batch Size AUTOMÁTICO: {BATCH_SIZE} frames (baseado em GPU capabilities)")
    
    # 1. Inicializar Compressores (GPU ou CPU)
    compressors = []
    
    if USE_CPU_FALLBACK:
        # Modo CPU - usar um único compressor CPU com workers internos
        cpu_comp = CPU_LZ4_Compressor()
        compressors.append(cpu_comp)
        print(f"[Compressor] Usando 1 compressor CPU com {cpu_comp.cpu_count} workers paralelos")
    else:
        # Modo GPU - prompt para seleção de GPUs
        excluded_indices = prompt_gpu_selection()
        
        # Obter dispositivos filtrados
        devices = get_filtered_devices(excluded_indices)
        
        if not devices:
            print("[Compressor] Nenhuma GPU disponível! Ativando fallback CPU...")
            USE_CPU_FALLBACK = True
            cpu_comp = CPU_LZ4_Compressor()
            compressors.append(cpu_comp)
        else:
            # Obter índices das GPUs habilitadas
            enabled_indices = []
            all_platforms = cl.get_platforms()
            global_idx = 0
            for p in all_platforms:
                try:
                    platform_devices = p.get_devices(device_type=cl.device_type.GPU)
                    for _ in platform_devices:
                        if global_idx not in excluded_indices:
                            enabled_indices.append(global_idx)
                        global_idx += 1
                except:
                    pass
            
            print(f"[Compressor] Inicializando {len(enabled_indices)} GPUs...")
            for gpu_index in enabled_indices:
                comp = GPU_LZ4_Compressor(device_index=gpu_index, max_input_size=frame_size, batch_size=BATCH_SIZE)
                if comp.enabled:
                    compressors.append(comp)
                else:
                    print(f"[Compressor] Falha ao inicializar compressor na GPU {gpu_index}")
            
            # Se nenhuma GPU funcionou, fallback para CPU
            if not compressors:
                print("[Compressor] Nenhuma GPU inicializada! Ativando fallback CPU...")
                USE_CPU_FALLBACK = True
                cpu_comp = CPU_LZ4_Compressor()
                compressors.append(cpu_comp)

    # Fila de compressores disponíveis para os workers
    import queue
    from concurrent.futures import ThreadPoolExecutor, wait
    
    compressor_queue = queue.Queue()
    for c in compressors:
        compressor_queue.put(c)
    
    mode_str = "CPU" if USE_CPU_FALLBACK else "GPU"
    print(f"[Compressor] Iniciando compressão com {len(compressors)} workers {mode_str}...")
    
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
    def compress_batch_task(batch_data, batch_id=None):
        # batch_data é uma lista de tuplas (frame_id, frame_bytes)
        frames = [b[1] for b in batch_data]
        frame_ids = [b[0] for b in batch_data]
        
        # Informar início do batch
        batch_size = len(frames)
        if batch_id is not None:
            print(f"\n[Batch {batch_id}] Iniciando processamento de {batch_size} frames ...")
        
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
            
            # Processar resultados com progresso
            for i, (res_bytes, res_size, _) in enumerate(batch_results):
                orig_data = frames[i]
                orig_size = len(orig_data)
                fid = frame_ids[i]
                
                # Mostrar progresso a cada 20% do batch
                progress_pct = int((i + 1) / batch_size * 100)
                if batch_id is not None and progress_pct in [20, 40, 60, 80] and (i + 1) == int(batch_size * progress_pct / 100):
                    print(f"[Batch {batch_id}] {progress_pct}% processado ({i + 1}/{batch_size} frames)")
                
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

    # Execução Paralela com Pipeline I/O
    num_workers = len(compressors) if compressors else 1
    # BATCH_SIZE já foi definido no início da função
    
    import threading
    
    # Filas do pipeline com tamanhos configuráveis
    read_queue_size = BATCH_SIZE * READ_BUFFER_BATCHES
    # A fila de escrita armazena BATCHES inteiros, não frames individuais
    write_queue_size = WRITE_BUFFER_BATCHES if WRITE_BUFFER_BATCHES > 0 else 0
    
    frame_queue = queue.Queue(maxsize=read_queue_size)  # Frames lidos aguardando processamento
    write_queue = queue.Queue(maxsize=write_queue_size)  # Resultados aguardando escrita
    
    # Eventos de controle
    reader_done = threading.Event()
    processing_done = threading.Event()
    pipeline_error = threading.Event()
    
    print(f"[Pipeline] Buffers: Read={READ_BUFFER_BATCHES} batches, Write={WRITE_BUFFER_BATCHES} batches")
    
    # ================================================================
    # THREAD 1: READER (Produtor - lê frames do disco)
    # ================================================================
    def reader_thread():
        """Lê frames em background e enfileira para processamento."""
        try:
            for frame_id, frame_data in generate_frames(entries, frame_size):
                if pipeline_error.is_set():
                    break
                if len(frame_data) == 0:
                    continue
                frame_queue.put((frame_id, frame_data))
        except Exception as e:
            print(f"[Reader] Erro: {e}")
            pipeline_error.set()
        finally:
            reader_done.set()
    
    # ================================================================
    # THREAD 3: WRITER (Consumidor - escreve resultados em ordem)
    # ================================================================
    pending_results = {}  # frame_id -> result_dict
    next_frame_to_write = [0]  # Usar lista para permitir modificação em closure
    
    def process_result(res):
        """Processa um resultado: atualiza stats e escreve no volume."""
        nonlocal current_vol_name
        
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
        
        # Validação para prevenir overflow
        res_size_value = res["size"]
        if isinstance(res_size_value, int) and res_size_value > 0 and res_size_value < (2 * 1024 * 1024 * 1024):
            try:
                global_stats["total_compressed_bytes"] += res_size_value
            except OverflowError:
                global_stats["total_compressed_bytes"] = res_size_value
        
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
    
    def writer_thread():
        """Escreve resultados em ordem de frame_id."""
        try:
            while not (processing_done.is_set() and write_queue.empty() and not pending_results):
                if pipeline_error.is_set():
                    break
                
                # Coletar resultados da fila
                try:
                    batch_results = write_queue.get(timeout=0.1)
                    for res in batch_results:
                        pending_results[res["frame_id"]] = res
                except queue.Empty:
                    pass
                
                # Escrever em ordem
                while next_frame_to_write[0] in pending_results:
                    res = pending_results.pop(next_frame_to_write[0])
                    process_result(res)
                    next_frame_to_write[0] += 1
                    
        except Exception as e:
            print(f"[Writer] Erro: {e}")
            pipeline_error.set()
    
    # ================================================================
    # THREAD 2: COMPRESSOR (Workers - processa batches)
    # ================================================================
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}  # batch_id -> Future
        current_batch = []
        batch_counter = 0
        
        # Iniciar threads de I/O
        reader = threading.Thread(target=reader_thread, name="FrameReader", daemon=True)
        writer_th = threading.Thread(target=writer_thread, name="FrameWriter", daemon=True)
        
        reader.start()
        writer_th.start()
        
        # Loop principal: coleta frames e submete batches
        while not pipeline_error.is_set():
            # Tentar coletar frame da fila
            try:
                frame_id, frame_data = frame_queue.get(timeout=0.1)
                current_batch.append((frame_id, frame_data))
            except queue.Empty:
                # Fila vazia - verificar se reader terminou
                if reader_done.is_set() and frame_queue.empty():
                    break
                continue
            
            # Submeter batch quando atingir tamanho
            if len(current_batch) >= BATCH_SIZE:
                f = executor.submit(compress_batch_task, list(current_batch), batch_counter)
                futures[batch_counter] = f
                batch_counter += 1
                current_batch = []
                
                # Backpressure: limitar batches em voo
                max_pending_batches = num_workers * 2
                while len(futures) > max_pending_batches and not pipeline_error.is_set():
                    done_batches = [bid for bid, fut in futures.items() if fut.done()]
                    for bid in done_batches:
                        try:
                            batch_res = futures[bid].result()
                            del futures[bid]
                            write_queue.put(batch_res)
                        except Exception as e:
                            print(f"[Compressor] Erro no batch {bid}: {e}")
                            del futures[bid]
                    
                    if not done_batches:
                        time.sleep(0.01)
        
        # Submeter último batch parcial
        if current_batch:
            f = executor.submit(compress_batch_task, list(current_batch), batch_counter)
            futures[batch_counter] = f
            batch_counter += 1
        
        # Aguardar todos os batches de compressão
        while futures and not pipeline_error.is_set():
            done_batches = [bid for bid, fut in futures.items() if fut.done()]
            for bid in done_batches:
                try:
                    batch_res = futures[bid].result()
                    del futures[bid]
                    write_queue.put(batch_res)
                except Exception as e:
                    print(f"[Compressor] Erro no batch {bid}: {e}")
                    del futures[bid]
            
            if not done_batches:
                time.sleep(0.01)
        
        # Sinalizar fim do processamento
        processing_done.set()
        
        # Aguardar writer terminar
        writer_th.join(timeout=60)
        
        if writer_th.is_alive():
            print("[Pipeline] AVISO: Writer thread ainda ativo após timeout")
    
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
    global FORCE_CPU_MODE
    
    parser = argparse.ArgumentParser(description="Compressor LZ4 GPU + Deduplicação")
    parser.add_argument("source", help="Pasta de origem")
    parser.add_argument("-o", "--output", default="archive_lz4.gpu", help="Base de saída")
    parser.add_argument("--frame-size-mb", type=int, default=16, help="Tamanho do frame (MB) - recomendado: 16-32 para janela de 16MB")
    parser.add_argument("--volume-size-mb", type=int, default=98, help="Tamanho do volume (MB)")
    parser.add_argument("--cpu", action="store_true", help="Forçar modo CPU (ignorar GPU/OpenCL)")
    
    args = parser.parse_args()
    
    # Aplicar flag --cpu
    if args.cpu:
        FORCE_CPU_MODE = True
    
    source_dir = Path(args.source).resolve()
    if not source_dir.is_dir():
        print(f"Erro: '{source_dir}' não é um diretório.")
        return 1
    
    output_base = Path(args.output).resolve()
    frame_size = args.frame_size_mb * 1024 * 1024
    max_volume_size = args.volume_size_mb * 1024 * 1024
    
    print("="*70)
    if FORCE_CPU_MODE:
        print("COMPRESSOR LZ4 CPU (--cpu)")
    else:
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
    
    # 2. Deduplicação (GPU ou CPU dependendo do modo)
    if FORCE_CPU_MODE:
        print("\n[Fase 1] Deduplicação CPU...")
    else:
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
    if FORCE_CPU_MODE:
        print("\n[Fase 2] Compressão LZ4 CPU...")
    else:
        print("\n[Fase 2] Compressão LZ4 GPU...")
    print(f"[Compressor] Parâmetros: frame_size={args.frame_size_mb}MB, max_volume={args.volume_size_mb}MB")
    
    index_params = {
        "frame_size": frame_size,
        "max_volume_size": max_volume_size,
        "source_root": str(source_dir),
        "deduplication": True,
        "dedup_saved_bytes": dedup_saved,
        "compressor": "lz4_cpu" if FORCE_CPU_MODE else "lz4_gpu"
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
