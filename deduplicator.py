"""
Módulo de deduplicação de arquivos usando GPU.

Identifica arquivos idênticos para economizar espaço no arquivo final.
Estratégia:
1. Agrupa arquivos por tamanho.
2. Para grupos com > 1 arquivo, calcula hash (FNV-1a 64-bit) em GPU.
3. Marca arquivos duplicados no FileEntry.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from iotools import FileEntry

class GPUFileDeduplicator:
    """
    Identifica arquivos duplicados usando hashing em GPU.
    Suporta Multi-GPU.
    """
    
    def __init__(self):
        self.enabled = False
        self._contexts = [] # List of (ctx, queue, program) per device
        self._num_devices = 0
        self._current_device_idx = 0
        self._initialize_opencl()
        
    def _initialize_opencl(self):
        try:
            import pyopencl as cl
            self._cl = cl
        except ImportError:
            print("[Deduplicator] pyopencl não disponível. Deduplicação GPU desativada.")
            return

        try:
            platforms = cl.get_platforms()
            devices = []
            for plat in platforms:
                devices.extend(plat.get_devices(device_type=cl.device_type.GPU))
            
            if not devices:
                print("[Deduplicator] Nenhuma GPU encontrada.")
                return

            # Kernel FNV-1a 64-bit (Vectorized / Stride 8)
            # Processa 8 bytes por vez (ulong) para maior throughput de memória
            kernel_src = """
            __kernel void block_fnv1a(
                __global const ulong* data,  // Lendo como ulong (8 bytes)
                const uint block_size_bytes,
                const uint num_blocks,
                __global ulong* block_hashes
            ) {
                uint gid = get_global_id(0);
                if (gid >= num_blocks) return;
                
                // Cada thread processa um bloco de 'block_size_bytes'
                // O input 'data' é ulong*, então o índice deve ser ajustado
                // block_size_bytes deve ser múltiplo de 8
                
                uint num_ulongs = block_size_bytes / 8;
                uint start_idx = gid * num_ulongs;
                
                ulong hash = 0xcbf29ce484222325UL;
                ulong prime = 0x100000001b3UL;
                
                // Loop principal otimizado (8 bytes por iteração)
                for (uint i = 0; i < num_ulongs; i++) {
                    ulong chunk = data[start_idx + i];
                    
                    // Misturar chunk (simulando FNV-1a mas em 64 bits)
                    // Para ser idêntico ao FNV-1a byte-a-byte seria mais lento.
                    // Aqui fazemos uma variante "FNV-1a-64-chunked" que é rápida e boa para dedup.
                    hash ^= chunk;
                    hash *= prime;
                }
                
                block_hashes[gid] = hash;
            }
            """
            
            for dev in devices:
                try:
                    ctx = cl.Context(devices=[dev])
                    queue = cl.CommandQueue(ctx, dev)
                    program = cl.Program(ctx, kernel_src).build()
                    self._contexts.append({
                        "device": dev,
                        "ctx": ctx,
                        "queue": queue,
                        "program": program
                    })
                    print(f"[Deduplicator] GPU Hashing ativado: {dev.name}")
                except Exception as e:
                    print(f"[Deduplicator] Falha ao inicializar device {dev.name}: {e}")
            
            self._num_devices = len(self._contexts)
            self.enabled = self._num_devices > 0
            
        except Exception as e:
            print(f"[Deduplicator] Falha na inicialização OpenCL: {e}")
            self.enabled = False

    def compute_file_hash(self, filepath: str, size: int) -> int:
        """
        Calcula hash do arquivo usando GPU (estratégia de blocos).
        Usa Round-Robin para distribuir entre GPUs.
        """
        if not self.enabled:
            return self._cpu_hash(filepath)
        
        # Validação prévia: arquivos vazios ou inválidos
        if size == 0:
            return self._cpu_hash(filepath)
            
        BLOCK_SIZE = 256 * 1024 # 256KB por bloco
        
        # Selecionar GPU (Round-Robin)
        gpu_res = self._contexts[self._current_device_idx]
        self._current_device_idx = (self._current_device_idx + 1) % self._num_devices
        
        ctx = gpu_res["ctx"]
        queue = gpu_res["queue"]
        program = gpu_res["program"]
        
        try:
            # Verificar se o arquivo é acessível e regular
            import os
            if not os.path.isfile(filepath):
                # Não é arquivo regular (pode ser link simbólico quebrado, dispositivo, etc.)
                return self._cpu_hash(filepath)
            
            # Verificar permissões de leitura
            if not os.access(filepath, os.R_OK):
                print(f"[Deduplicator] Sem permissão de leitura: {filepath}")
                return hash(filepath)  # Hash dummy para evitar duplicata acidental
            
            with open(filepath, "rb") as f:
                data = f.read() # Ler tudo para memória (cuidado com arquivos gigantes)
                
                # Validação adicional: arquivo vazio ou leitura falhou
                if len(data) == 0:
                    return self._cpu_hash(filepath)
                
                # Para arquivos gigantes, teríamos que ler em chunks. 
                # MVP: ler tudo se couber na RAM, senão fallback CPU.
                if len(data) > 100 * 1024 * 1024: # > 100MB
                     return self._cpu_hash(filepath) # Fallback para não estourar VRAM/RAM
                
            # Padding para alinhar com BLOCK_SIZE e garantir múltiplo de 8
            n = len(data)
            
            # BLOCK_SIZE deve ser múltiplo de 8 (256KB é)
            if BLOCK_SIZE % 8 != 0:
                BLOCK_SIZE = ((BLOCK_SIZE + 7) // 8) * 8
                
            num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            padded_len = num_blocks * BLOCK_SIZE
            
            if n < padded_len:
                data += b'\0' * (padded_len - n)
                
            arr_in = np.frombuffer(data, dtype=np.uint8)
            
            # O kernel espera ulong*, mas podemos passar buffer de bytes.
            # O cast é feito no kernel arguments ou implicitamente.
            # Importante: O buffer deve estar alinhado.
            
            mf = self._cl.mem_flags
            buf_in = self._cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_in)
            
            hashes = np.zeros(num_blocks, dtype=np.uint64)
            buf_hashes = self._cl.Buffer(ctx, mf.WRITE_ONLY, hashes.nbytes)
            
            kernel = self._cl.Kernel(program, "block_fnv1a")
            # Args: data (ulong*), block_size_bytes (uint), num_blocks (uint), hashes (ulong*)
            kernel.set_args(buf_in, np.uint32(BLOCK_SIZE), np.uint32(num_blocks), buf_hashes)
            
            self._cl.enqueue_nd_range_kernel(queue, kernel, (num_blocks,), None)
            queue.finish()
            
            self._cl.enqueue_copy(queue, hashes, buf_hashes).wait()
            
            # Combinar hashes dos blocos na CPU (ordem importa)
            final_hash = 0xcbf29ce484222325
            prime = 0x100000001b3
            
            for h in hashes:
                final_hash ^= int(h)
                final_hash *= prime
                final_hash &= 0xFFFFFFFFFFFFFFFF # Manter 64-bit
                
            return final_hash
            
        except (OSError, IOError, PermissionError) as e:
            # Erros de I/O: arquivo não existe, sem permissão, etc.
            print(f"[Deduplicator] Erro I/O ({filepath}): {e}. Usando CPU.")
            return self._cpu_hash(filepath)
        except Exception as e:
            print(f"[Deduplicator] Erro GPU ({gpu_res['device'].name}): {e}. Usando CPU.")
            return self._cpu_hash(filepath)

    def compute_data_hash(self, data: bytes) -> int:
        """
        Calcula hash de dados já em memória usando GPU.
        Versão otimizada para Double Buffer (evita re-leitura de arquivo).
        """
        if not self.enabled or len(data) == 0:
            return self._hash_bytes(data)
        
        # Para arquivos grandes, fallback para CPU (evitar estouro VRAM)
        if len(data) > 100 * 1024 * 1024:
            return self._hash_bytes(data)
            
        BLOCK_SIZE = 256 * 1024  # 256KB por bloco
        
        # Selecionar GPU (Round-Robin)
        gpu_res = self._contexts[self._current_device_idx]
        self._current_device_idx = (self._current_device_idx + 1) % self._num_devices
        
        ctx = gpu_res["ctx"]
        queue = gpu_res["queue"]
        program = gpu_res["program"]
        
        try:
            n = len(data)
            num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            padded_len = num_blocks * BLOCK_SIZE
            
            if n < padded_len:
                data = data + b'\0' * (padded_len - n)
                
            arr_in = np.frombuffer(data, dtype=np.uint8)
            
            mf = self._cl.mem_flags
            buf_in = self._cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_in)
            
            hashes = np.zeros(num_blocks, dtype=np.uint64)
            buf_hashes = self._cl.Buffer(ctx, mf.WRITE_ONLY, hashes.nbytes)
            
            kernel = self._cl.Kernel(program, "block_fnv1a")
            kernel.set_args(buf_in, np.uint32(BLOCK_SIZE), np.uint32(num_blocks), buf_hashes)
            
            self._cl.enqueue_nd_range_kernel(queue, kernel, (num_blocks,), None)
            queue.finish()
            
            self._cl.enqueue_copy(queue, hashes, buf_hashes).wait()
            
            # Combinar hashes dos blocos
            final_hash = 0xcbf29ce484222325
            prime = 0x100000001b3
            
            for h in hashes:
                final_hash ^= int(h)
                final_hash *= prime
                final_hash &= 0xFFFFFFFFFFFFFFFF
                
            return final_hash
            
        except Exception as e:
            # Fallback CPU em caso de erro
            return self._hash_bytes(data)

    def _cpu_hash(self, filepath: str) -> int:
        """Fallback CPU (FNV-1a simples)."""
        h = 0xcbf29ce484222325
        prime = 0x100000001b3
        mask = 0xFFFFFFFFFFFFFFFF
        
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                for b in chunk:
                    h ^= b
                    h *= prime
                    h &= mask
        return h

    def _hash_bytes(self, data: bytes) -> int:
        """Hash rápido de bytes usando hashlib (C nativo)."""
        import hashlib
        # MD5 é muito mais rápido que FNV byte-a-byte em Python
        # Retorna int para compatibilidade com o código existente
        return int(hashlib.md5(data).hexdigest(), 16)


    def find_duplicates(self, entries: List[FileEntry]) -> List[FileEntry]:
        """
        Processa a lista de arquivos e marca duplicatas usando filtro de 4 estágios.
        
        Estágios:
        1. Tamanho (Size)
        2. Primeiro Byte (First Byte)
        3. Último Byte (Last Byte)
        4. Hash Completo (Full GPU Hash)
        """
        print(f"[Deduplicator] Iniciando busca por duplicatas em {len(entries)} arquivos...")
        
        # --- ESTÁGIO 1: Agrupar por Tamanho ---
        by_size = defaultdict(list)
        for e in entries:
            if e.size > 0:
                by_size[e.size].append(e)
        
        # Filtrar apenas grupos com colisão de tamanho
        candidates_s1 = []
        for size, group in by_size.items():
            if len(group) > 1:
                candidates_s1.append(group)
                
        count_s1 = sum(len(g) for g in candidates_s1)
        print(f"[Dedup Stage 1] Size Filter: {len(by_size)} grupos únicos. {count_s1} arquivos candidatos a duplicata.")
        
        if count_s1 == 0:
            return entries

        # --- ESTÁGIO 2: Primeiros 2 Bytes ---
        candidates_s2 = []
        removed_s2 = 0
        
        for group in candidates_s1:
            by_first = defaultdict(list)
            for entry in group:
                try:
                    with open(entry.path_abs, "rb") as f:
                        first = f.read(2)
                    by_first[first].append(entry)
                except Exception:
                    # Se der erro de leitura, ignora (trata como único)
                    pass
            
            for subgroup in by_first.values():
                if len(subgroup) > 1:
                    candidates_s2.append(subgroup)
                else:
                    removed_s2 += 1
                    
        count_s2 = sum(len(g) for g in candidates_s2)
        print(f"[Dedup Stage 2] First 2 Bytes: {removed_s2} removidos. {count_s2} restantes.")

        # --- ESTÁGIO 3: Últimos 2 Bytes ---
        candidates_s3 = []
        removed_s3 = 0
        
        for group in candidates_s2:
            by_last = defaultdict(list)
            size = group[0].size # Todos no grupo têm mesmo tamanho
            
            for entry in group:
                try:
                    with open(entry.path_abs, "rb") as f:
                        if size >= 2:
                            f.seek(-2, 2) # Ir para os últimos 2 bytes
                            last = f.read(2)
                        else:
                            # Se tamanho < 2, lê o que tem (já verificado no estágio 2, mas ok)
                            f.seek(0)
                            last = f.read(2)
                    by_last[last].append(entry)
                except Exception:
                    pass
                    
            for subgroup in by_last.values():
                if len(subgroup) > 1:
                    candidates_s3.append(subgroup)
                else:
                    removed_s3 += 1
                    
        count_s3 = sum(len(g) for g in candidates_s3)
        print(f"[Dedup Stage 3] Last 2 Bytes:  {removed_s3} removidos. {count_s3} restantes.")

        # --- ESTÁGIO 4: 8 Bytes do Centro ---
        candidates_s4 = []
        removed_s4 = 0
        
        for group in candidates_s3:
            by_center = defaultdict(list)
            size = group[0].size
            
            for entry in group:
                try:
                    with open(entry.path_abs, "rb") as f:
                        if size >= 8:
                            # Posição central e adjacentes - ler 8 bytes
                            center = size // 2
                            # Começar 4 bytes antes do centro para pegar 8 bytes balanceados
                            f.seek(center - 4)
                            center_bytes = f.read(8)
                        else:
                            # Para arquivos muito pequenos, lê o que tem
                            f.seek(0)
                            center_bytes = f.read(size)
                    by_center[center_bytes].append(entry)
                except Exception:
                    pass
                    
            for subgroup in by_center.values():
                if len(subgroup) > 1:
                    candidates_s4.append(subgroup)
                else:
                    removed_s4 += 1
                    
        count_s4 = sum(len(g) for g in candidates_s4)
        print(f"[Dedup Stage 4] Center 8 Bytes: {removed_s4} removidos.")

        # --- ESTÁGIO 5+: Progressive Scan ---
        # Stage 5: Arquivos pequenos (≤50MB) - hash completo
        # Stage 6+: Arquivos grandes - scan progressivo em chunks de 50MB
        
        CHUNK_SIZE = 50 * 1024 * 1024  # 50MB
        dupes_count = 0
        bytes_saved = 0
        MAX_WORKERS = 8
        
        # Separar em pequenos (≤50MB) e grandes (>50MB)
        small_files_groups = []
        large_files_groups = []
        
        for group in candidates_s4:
            size = group[0].size
            if size <= CHUNK_SIZE:
                small_files_groups.append(group)
            else:
                large_files_groups.append(group)
        
        # --- Stage 5: Arquivos pequenos - Hash Completo via GPU (Double Buffer) ---
        small_count = sum(len(g) for g in small_files_groups)
        small_removed = 0
        small_remaining = 0
        processed_count = 0
        
        if small_count > 0:
            import queue
            import threading
            
            # Output inicial ANTES de começar
            print(f"[Dedup Stage 5] Progressive scan activated: Small files ({small_count} arquivos)")
            print(f"[Dedup Stage 5] Double Buffer: I/O + GPU em paralelo")
            
            # ===================================================================
            # DOUBLE BUFFER: Producer-Consumer com fila limitada
            # ===================================================================
            # - Reader Thread: Lê arquivos do disco para memória
            # - GPU Workers: Calculam hash dos dados já carregados
            # - Fila de 128 itens: ~2-4GB RAM max (assumindo média de 20-30MB por arquivo)
            # - Para ajustar: BUFFER_SIZE * tamanho médio dos arquivos = RAM usada
            # ===================================================================
            
            BUFFER_SIZE = 128  # Itens na fila - seguro para sistemas com 16GB+ RAM
            data_queue = queue.Queue(maxsize=BUFFER_SIZE)
            reader_done = threading.Event()
            
            # Flatten all entries for sequential reading
            all_entries = []
            for group in small_files_groups:
                for entry in group:
                    all_entries.append((entry, group[0].size))
            
            def reader_thread():
                """Lê arquivos do disco e coloca na fila (Producer)."""
                for entry, size in all_entries:
                    try:
                        with open(entry.path_abs, "rb") as f:
                            file_data = f.read()
                        data_queue.put((entry, size, file_data))
                    except Exception:
                        data_queue.put((entry, size, None))  # Erro de leitura
                reader_done.set()
            
            # Iniciar thread de leitura
            reader = threading.Thread(target=reader_thread, name="FileReader", daemon=True)
            reader.start()
            
            # Hash tracking por grupo
            group_hashes = defaultdict(dict)  # group_id -> {hash: entry}
            entry_to_group = {}
            for gid, group in enumerate(small_files_groups):
                for entry in group:
                    entry_to_group[entry.path_abs] = gid
            
            # Processar com GPU (Consumer)
            while True:
                try:
                    # Tentar pegar item da fila (com timeout para checar se reader terminou)
                    item = data_queue.get(timeout=0.1)
                    entry, size, file_data = item
                    
                    processed_count += 1
                    
                    # Feedback a cada 1000 arquivos
                    if processed_count % 1000 == 0:
                        pct = (processed_count / small_count) * 100
                        qsize = data_queue.qsize()
                        print(f"[Dedup Stage 5] Progresso: {processed_count}/{small_count} ({pct:.1f}%) - Duplicatas: {small_removed} | Buffer: {qsize}")
                    
                    gid = entry_to_group[entry.path_abs]
                    hashes = group_hashes[gid]
                    
                    if file_data is None:
                        # Erro de leitura - usar hash único
                        hashes[hash(entry.path_abs)] = entry
                        small_remaining += 1
                    else:
                        # Calcular hash via GPU usando dados já em memória
                        h = self.compute_data_hash(file_data)
                        
                        if h in hashes:
                            original = hashes[h]
                            entry.is_duplicate = True
                            entry.original_path_rel = original.path_rel
                            dupes_count += 1
                            bytes_saved += size
                            small_removed += 1
                        else:
                            hashes[h] = entry
                            small_remaining += 1
                    
                except queue.Empty:
                    # Fila vazia - verificar se reader terminou
                    if reader_done.is_set() and data_queue.empty():
                        break
            
            # Aguardar reader terminar (safety)
            reader.join(timeout=5)
            
            print(f"[Dedup Stage 5] Concluído: {small_removed} removidos. {small_remaining} restantes")
        
        # --- Stage 6+: Arquivos grandes - Progressive Scan ---
        if large_files_groups:
            stage_num = 6
            current_offset = 0
            pending_groups = large_files_groups[:]
            
            while pending_groups:
                chunk_start = current_offset
                chunk_end = current_offset + CHUNK_SIZE
                end_mb = chunk_end // (1024 * 1024)
                
                next_pending = []
                stage_removed = 0
                stage_remaining = 0
                
                for group in pending_groups:
                    size = group[0].size
                    is_final_chunk = chunk_end >= size
                    
                    by_chunk_hash = defaultdict(list)
                    
                    for entry in group:
                        if entry.is_duplicate:
                            continue
                        try:
                            with open(entry.path_abs, "rb") as f:
                                f.seek(chunk_start)
                                chunk_data = f.read(CHUNK_SIZE)
                            chunk_hash = self._hash_bytes(chunk_data)
                            by_chunk_hash[chunk_hash].append(entry)
                        except Exception:
                            by_chunk_hash[hash(entry.path_abs)].append(entry)
                    
                    for subgroup in by_chunk_hash.values():
                        if len(subgroup) == 1:
                            stage_remaining += 1
                        elif len(subgroup) > 1:
                            if is_final_chunk:
                                first = subgroup[0]
                                for dup in subgroup[1:]:
                                    dup.is_duplicate = True
                                    dup.original_path_rel = first.path_rel
                                    dupes_count += 1
                                    bytes_saved += dup.size
                                    stage_removed += 1
                                stage_remaining += 1
                            else:
                                next_pending.append(subgroup)
                                stage_remaining += len(subgroup)
                
                print(f"[Dedup Stage {stage_num}] Progressive {end_mb}MB: {stage_removed} removidos. {stage_remaining} restantes")
                
                pending_groups = next_pending
                current_offset = chunk_end
                stage_num += 1
                
                if stage_num > 200:
                    print(f"[Dedup] AVISO: Limite de stages atingido.")
                    break
        
        print(f"[Dedup Final] Encontradas {dupes_count} duplicatas reais.")
        print(f"[Dedup Final] Economia potencial: {bytes_saved / 1024 / 1024:.2f} MB")
                    
        print(f"[Dedup Final] Encontradas {dupes_count} duplicatas reais.")
        print(f"[Dedup Final] Economia potencial: {bytes_saved / 1024 / 1024:.2f} MB")
        
        return entries
