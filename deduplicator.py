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
        print(f"[Dedup Stage 4] Center 8 Bytes: {removed_s4} removidos. {count_s4} restantes para Hash Completo.")

        # --- ESTÁGIO 5: Hash Completo (GPU) ---
        dupes_count = 0
        bytes_saved = 0
        
        # Definir o número máximo de workers (threads) para I/O e GPU
        # Um valor razoável é o número de núcleos da CPU ou um pouco mais.
        MAX_WORKERS = 8 
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for group in candidates_s4:
                hashes = {}
                size = group[0].size
                
                # 1. Submeter todas as tarefas de hashing do grupo para o executor
                future_to_entry = {
                    executor.submit(self.compute_file_hash, str(entry.path_abs), size): entry
                    for entry in group
                }
                
                # 2. Processar os resultados à medida que ficam prontos (paralelo)
                for future in as_completed(future_to_entry):
                    entry = future_to_entry[future]
                    try:
                        # O hash é calculado na thread, liberando o GIL para I/O e OpenCL
                        h = future.result()
                        
                        # 3. Lógica de deduplicação (sequencial, para thread safety)
                        if h in hashes:
                            # Duplicata confirmada!
                            original = hashes[h]
                            entry.is_duplicate = True
                            entry.original_path_rel = original.path_rel
                            dupes_count += 1
                            bytes_saved += size
                        else:
                            hashes[h] = entry
                            
                    except Exception as exc:
                        print(f"[Deduplicator] Erro ao processar hash de {entry.path_abs}: {exc}")
                        # Tratar como arquivo único em caso de erro
                        hashes[hash(entry.path_abs)] = entry # Garante que não será duplicado por acidente
                        
        # O restante do código (linhas 340-343) permanece o mesmo
        # ...
        
        print(f"[Dedup Final] Encontradas {dupes_count} duplicatas reais.")
        print(f"[Dedup Final] Economia potencial: {bytes_saved / 1024 / 1024:.2f} MB")
                    
        print(f"[Dedup Final] Encontradas {dupes_count} duplicatas reais.")
        print(f"[Dedup Final] Economia potencial: {bytes_saved / 1024 / 1024:.2f} MB")
        
        return entries
