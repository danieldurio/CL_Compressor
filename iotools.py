from __future__ import annotations

import json
import os
import threading
import shutil
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Iterable, Iterator, List, Dict, Any, Optional, Tuple, BinaryIO


# ---------------------------------------------------------------------------
# Representação de arquivos de entrada
# ---------------------------------------------------------------------------
    
@dataclass
class FileEntry:
    """
    Representa um arquivo de entrada para o compressor.

    Campos esperados pelo restante do sistema (main, preprocessor, compressor):

      - path_abs: caminho absoluto do arquivo (Path)
      - path_rel: caminho relativo à pasta raiz (str)
      - size:     tamanho em bytes (int)
    """
    path_abs: Path
    path_rel: str
    size: int
    is_duplicate: bool = False
    original_path_rel: Optional[str] = None  # Caminho relativo do arquivo original (se for duplicata)
    
    # Atributos ADICIONAIS para indexação completa (GPU_IDX3)
    # Estes podem ser preenchidos por módulos como acls.py ou preprocessor.
    sddl: Optional[str] = None
    attrs: Optional[int] = None
    ctime: Optional[float] = None
    mtime: Optional[float] = None
    atime: Optional[float] = None
    original_size: Optional[int] = None # Para duplicatas ou arquivos reprocessados
    
    # Rastreamento de Frame
    start_frame_id: Optional[int] = None # Frame onde o arquivo começa

    # Helpers opcionais (caso em algum ponto usem .path ou .rel_path)
    @property
    def path(self) -> Path:
        return self.path_abs

    @property
    def rel_path(self) -> str:
        return self.path_rel

# Carregar configurações do config.txt centralizado
try:
    import config_loader
    # Configuração do scanner paralelo (carregada de config.txt)
    NUM_SCAN_WORKERS = config_loader.get_num_scan_workers()
    PRE_SCAN_TARGET_DIRS = config_loader.get_pre_scan_target_dirs()
    SCAN_STATUS_INTERVAL = config_loader.get_scan_status_interval()
except ImportError:
    NUM_SCAN_WORKERS = 4
    PRE_SCAN_TARGET_DIRS = 1000
    SCAN_STATUS_INTERVAL = 5.0


def scan_directory_parallel(root: Path) -> List[FileEntry]:
    """
    Scanner paralelo otimizado com Arquitetura Producer-Consumer (3-Stage).
    
    Stages:
    1. Discovery Thread (Producer): Varre estrutura de diretórios e popula 'pending_dirs'.
    2. Distributor Thread: Monitora workers e distribui lotes da 'pending_dirs' para 'worker_queues'.
    3. Worker Threads (Consumers): Processam suas filas locais de forma atômica e isolada.
    """
    root = root.resolve()
    
    # --- Estruturas de Dados ---
    
    # Fila de diretórios descobertos aguardando distribuição
    # Discovery -> Distributor
    pending_dirs_queue: Queue[Path] = Queue()
    
    # Filas individuais por worker
    # Distributor -> Worker[i]
    worker_queues: List[Queue[List[Path]]] = [Queue() for _ in range(NUM_SCAN_WORKERS)]
    
    # Resultados consolidados
    results: List[FileEntry] = []
    results_lock = threading.Lock()
    
    # Stats por Worker
    # worker_stats[id] = {"dirs": 0, "files": 0, "active": False, "status": "Idle"}
    worker_stats = [{"dirs": 0, "files": 0, "active": False, "status": "Idle"} for _ in range(NUM_SCAN_WORKERS)]
    stats_lock = threading.Lock()
    
    # Global Stats
    global_stats = {
        "discovery_count": 0,  # Total de dirs encontrados pelo Discovery
        "distributed_count": 0, # Total de dirs entregues aos workers
        "processed_dirs": 0,    # Total processado efetivamente
        "processed_files": 0
    }
    
    # Eventos de Controle
    discovery_done = threading.Event()
    distributor_done = threading.Event()
    
    # Batch Size para distribuição
    TASK_BATCH_SIZE = 1000
    
    # -----------------------------------------------------------------------
    # 1. Discovery Thread (Producer)
    # -----------------------------------------------------------------------
    def discovery_task():
        try:
            count = 0
            # Usar os.walk é eficiente o suficiente para listar diretórios?
            # Ou BFS manual? O usuário pediu "avanca de subdir".
            # os.walk é DFS bottom-up ou top-down. Top-down é melhor para stream.
            # Vamos usar os.walk(topdown=True) e apenas coletar os nomes de diretórios para scan.
            
            # Nota: Scan paralelo pressupõe que workers vão ler arquivos.
            # Se discovery entregar a pasta, o worker faz scandir nela.
            
            # Otimização: Discovery apenas navega na estrutura. Worker lista arquivos.
            
            # Adicionar ROOT inicial
            pending_dirs_queue.put(root)
            count += 1
            
            # BFS Manual ou Walk? 
            # Se usarmos os.walk, ele já faz o scandir. Se o worker fizer scandir de novo, é trabalho duplicado?
            # Sim. Mas os.walk retorna (root, dirs, files).
            # Podemos enviar (root, files) para o worker? Sim, mas serialized é ruim.
            # Melhor: Discovery faz apenas a navegação estrutural rápida (os.scandir recursivo ou stack).
            
            # Vamos usar uma stack manual para controle total e evitar lock de GIL longo do os.walk
            stack = [root]
            while stack:
                current_dir = stack.pop()
                
                # Scan para achar subdirs
                try:
                    subdirs = []
                    # Worker vai precisar processar este diretório também para pegar arquivos?
                    # Sim. Então cada diretório visitado é uma tarefa para workers e fonte de novos dirs.
                    
                    # Vamos otimizar: Discovery faz scandir, separa subdirs para stack e
                    # manda o prorprio current_dir para processamento de arquivos.
                    # Mas se Discovery faz scandir, ele já viu os arquivos.
                    # Se não aproveitarmos isso, o worker vai fazer scandir de novo -> I/O duplicado.
                    
                    # Solução Híbrida Performance:
                    # Discovery navega. Mas para evitar duplicar I/O, Discovery poderia listar tudo.
                    # MAS, o gargalo é "stat" em milhões de arquivos.
                    # os.scandir é rápido (só nomes). stat é lento.
                    # Então: Discovery faz os.scandir APENAS para achar DIRs (skip files stat).
                    # Worker faz os.scandir NO MESMO dir para pegar FILEs e fazer stat.
                    # O "custo" de abrir a pasta 2x é menor que serializar milhões de file entries entre threads.
                    # O cache do SO (MFT) ajuda na segunda leitura.
                    
                    # Implementação: Stack DFS
                    with os.scandir(current_dir) as it:
                        for entry in it:
                            if entry.is_dir(follow_symlinks=False):
                                path_obj = Path(entry.path)
                                subdirs.append(path_obj)
                                # Adiciona à fila de trabalho (pending)
                                pending_dirs_queue.put(path_obj)
                                count += 1
                                
                                # Flush periódico no count
                                if count % 100 == 0:
                                    with stats_lock:
                                        global_stats["discovery_count"] = count
                    
                    # Adicionar subdirs à stack para continuar descoberta
                    # Reverse para DFS ou normal? Tanto faz.
                    stack.extend(subdirs)
                    
                except OSError:
                    pass
        finally:
            with stats_lock:
                global_stats["discovery_count"] = count
            discovery_done.set()
            # print("[Discovery] Finalizado.")

    # -----------------------------------------------------------------------
    # 2. Distributor Thread
    # -----------------------------------------------------------------------
    def distributor_task():
        while not discovery_done.is_set() or not pending_dirs_queue.empty():
            # Estratégia:
            # 1. Verificar workers ociosos (queue vazia ou baixa)
            # 2. Se tiver pending, puxar batch e entregar
            
            try:
                # Sleep curto para não busy-wait se não tiver trabalho
                if pending_dirs_queue.empty():
                    import time
                    time.sleep(0.05)
                    continue
                
                # Check workers que precisam de trabalho
                target_workers = []
                for i in range(NUM_SCAN_WORKERS):
                    # Critério: Fila vazia ou quase vazia?
                    # Se tiver menos que um batch, manda mais.
                    if worker_queues[i].qsize() < 2: 
                        target_workers.append(i)
                
                if not target_workers:
                    # Todos ocupados, aguarda um pouco
                    import time
                    time.sleep(0.1)
                    continue
                
                # Temos workers famintos e trabalho pendente.
                # Distribui Round-Robin para os famintos
                for worker_id in target_workers:
                    if pending_dirs_queue.empty():
                        break
                        
                    # Montar Batch
                    batch = []
                    try:
                        while len(batch) < TASK_BATCH_SIZE:
                            item = pending_dirs_queue.get_nowait()
                            batch.append(item)
                    except Empty:
                        pass
                    
                    if batch:
                        worker_queues[worker_id].put(batch)
                        with stats_lock:
                            global_stats["distributed_count"] += len(batch)
                            
            except Exception as e:
                print(f"[Distributor] Error: {e}")
        
        distributor_done.set()
        # print("[Distributor] Finalizado.")

    # -----------------------------------------------------------------------
    # 3. Worker Threads (Consumers)
    # -----------------------------------------------------------------------
    def worker_task(worker_id):
        my_queue = worker_queues[worker_id]
        local_entries = []
        
        w_dirs = 0
        w_files = 0
        
        while True:
            # Check de terminação
            if my_queue.empty():
                if distributor_done.is_set():
                    # Se distributor acabou e minha fila ta vazia, acabou.
                    break
                else:
                    # Espera trabalho chegar
                    with stats_lock:
                       worker_stats[worker_id]["status"] = "Wait"
                       worker_stats[worker_id]["active"] = False
                    
                    try:
                        # Blocking get com timeout para checar flag de saida
                        batch = my_queue.get(timeout=0.5)
                    except Empty:
                        continue
            else:
                # Tem trabalho, pega non-blocking (já sabemos que tem)
                try:
                    batch = my_queue.get_nowait()
                except Empty:
                    continue
            
            # Processar Batch
            with stats_lock:
                worker_stats[worker_id]["status"] = "Work"
                worker_stats[worker_id]["active"] = True
            
            for dir_path in batch:
                w_dirs += 1
                try:
                    # Scan apenas de ARQUIVOS neste diretório (não recursivo)
                    with os.scandir(dir_path) as it:
                        for entry in it:
                            if entry.is_file(follow_symlinks=False):
                                try:
                                    path_obj = Path(entry.path)
                                    # size = entry.stat().st_size
                                    stat_res = entry.stat()
                                    size = stat_res.st_size
                                    
                                    path_rel = str(path_obj.relative_to(root)).replace("\\", "/")
                                    local_entries.append(FileEntry(path_abs=path_obj, path_rel=path_rel, size=size))
                                    w_files += 1
                                except OSError:
                                    pass
                except OSError:
                    pass
                
                # Flush periódico
                if w_dirs >= 50:
                    with stats_lock:
                         worker_stats[worker_id]["dirs"] += w_dirs
                         worker_stats[worker_id]["files"] += w_files
                         global_stats["processed_dirs"] += w_dirs # Global aggregate
                         global_stats["processed_files"] += w_files
                    w_dirs = 0
                    w_files = 0
            
            # Flush fim do Batch
            if w_dirs > 0 or w_files > 0:
                with stats_lock:
                     worker_stats[worker_id]["dirs"] += w_dirs
                     worker_stats[worker_id]["files"] += w_files
                     global_stats["processed_dirs"] += w_dirs
                     global_stats["processed_files"] += w_files
                w_dirs = 0
                w_files = 0
            
            my_queue.task_done()
        
        # Persistir resultados finais
        with stats_lock:
             worker_stats[worker_id]["status"] = "Done"
             worker_stats[worker_id]["active"] = False
             
        if local_entries:
            with results_lock:
                results.extend(local_entries)

    # -----------------------------------------------------------------------
    # 4. Monitor Thread
    # -----------------------------------------------------------------------
    def monitor_task():
        import time
        while not distributor_done.is_set(): # Monitora até distributor acabar (workers acabam logo depois)
             time.sleep(SCAN_STATUS_INTERVAL)
             # Se distributor acabou, sai do loop na proxima
             
             with stats_lock:
                 disc = global_stats["discovery_count"]
                 proc_d = global_stats["processed_dirs"]
                 proc_f = global_stats["processed_files"]
                 
                 threads_str_list = []
                 active_count = 0
                 
                 for i, ws in enumerate(worker_stats):
                     s = ws["status"][0] # W, I, D
                     d = ws["dirs"]
                     f = ws["files"]
                     if ws["active"]: active_count += 1
                     threads_str_list.append(f"T{i}({s}):{d}D")
                 
             thread_line = " | ".join(threads_str_list)
             pending = pending_dirs_queue.qsize()
             
             print(f"\n[Scan] Disc:{disc} | Proc:{proc_d} Dirs | {proc_f} Files | Pending:{pending}")
             print(f"[Work] {thread_line}\n")
             
             if distributor_done.is_set(): break

    # -----------------------------------------------------------------------
    # Iniciar Threads
    # -----------------------------------------------------------------------
    print(f"[Scan] Iniciando Producer-Consumer Scan ({NUM_SCAN_WORKERS} workers)...")
    
    # 1. Discovery
    t_disc = threading.Thread(target=discovery_task, name="Discovery", daemon=True)
    t_disc.start()
    
    # 2. Distributor
    t_dist = threading.Thread(target=distributor_task, name="Distributor", daemon=True)
    t_dist.start()
    
    # 3. Workers
    t_workers = []
    for i in range(NUM_SCAN_WORKERS):
        t = threading.Thread(target=worker_task, args=(i,), name=f"Worker-{i}")
        t.start()
        t_workers.append(t)
        
    # 4. Monitor
    t_mon = threading.Thread(target=monitor_task, name="Monitor", daemon=True)
    t_mon.start()
    
    # Joins
    t_disc.join()
    t_dist.join()
    for t in t_workers:
        t.join()
    t_mon.join(timeout=1.0)
    
    # Sort results
    results.sort(key=lambda e: e.path_rel.lower())
    
    print(f"[Scan] Finalizado. Total: {len(results)} arquivos.")
    return results


def scan_directory(root: Path, parallel: bool = True) -> List[FileEntry]:
    """
    Varre recursivamente a pasta `root` e monta a lista de FileEntry.
    """
    if parallel:
        return scan_directory_parallel(root)
    entries: List[FileEntry] = []
    root = root.resolve()
    
    skipped_count = 0
    skipped_reasons = {}

    try:
        all_paths = list(root.rglob("*"))
    except PermissionError as e:
        print(f"[SCAN] [WARN] SKIP: Sem permissao para acessar '{root}': {e}")
        return entries
    except OSError as e:
        print(f"[SCAN] [WARN] SKIP: Erro ao acessar '{root}': {e}")
        return entries

    for path_abs in all_paths:
        try:
            if not path_abs.is_file():
                continue

            size = path_abs.stat().st_size
            path_rel = str(path_abs.relative_to(root)).replace("\\", "/")
            entries.append(FileEntry(path_abs=path_abs, path_rel=path_rel, size=size))
            
        except PermissionError:
            skipped_count += 1
            reason = "Permissão negada"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue
        except OSError as e:
            skipped_count += 1
            reason = f"Erro de I/O"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue
        except Exception as e:
            skipped_count += 1
            reason = f"Erro inesperado: {type(e).__name__}"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

    if skipped_count > 0:
        print(f"\n[SCAN] Resumo: {skipped_count} arquivo(s) ignorado(s)")

    entries.sort(key=lambda e: e.path_rel.lower())
    return entries


def estimate_total_size(entries: Iterable[FileEntry]) -> int:
    return sum(e.size for e in entries)


# ---------------------------------------------------------------------------
# Detecção de dispositivos OpenCL
# ---------------------------------------------------------------------------

@dataclass
class OpenCLInfo:
    available: bool
    devices: List[Tuple[str, str]]  # (platform_name, device_name)

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def device_names(self) -> List[str]:
        return [f"{plat} :: {dev}" for (plat, dev) in self.devices]


def detect_opencl_devices() -> OpenCLInfo:
    try:
        import pyopencl as cl  # type: ignore
    except ImportError:
        return OpenCLInfo(available=False, devices=[])

    try:
        platforms = cl.get_platforms()
    except Exception:
        return OpenCLInfo(available=False, devices=[])

    devs: List[Tuple[str, str]] = []
    for plat in platforms:
        try:
            for dev in plat.get_devices():
                devs.append((plat.name, dev.name))
        except Exception:
            continue

    return OpenCLInfo(available=bool(devs), devices=devs)


# ---------------------------------------------------------------------------
# Metadados de frames e escrita de volumes
# ---------------------------------------------------------------------------

@dataclass
class FrameMeta:
    frame_id: int
    volume_name: str
    offset: int
    compressed_size: int
    uncompressed_size: int


class VolumeWriter:
    """
    Responsável por gravar os frames comprimidos em volumes do tipo:
        <base>.001, <base>.002, ...
    Disponibiliza métricas de velocidade.
    """

    def __init__(self, base_path: Path, max_volume_size: int) -> None:
        self.base_path = base_path
        self.max_volume_size = int(max_volume_size)
        self.current_volume_index = 0
        self.current_fp: Optional[BinaryIO] = None
        self.current_size = 0
        self.current_volume_name: Optional[str] = None
        
        self.volume_start_time: Optional[float] = None
        self.volume_bytes_written = 0

    def _open_new_volume(self) -> None:
        speed_str = ""
        if self.current_fp is not None:
            self.current_fp.close()
            
            if self.volume_start_time and self.volume_bytes_written > 0:
                import time
                elapsed = time.time() - self.volume_start_time
                if elapsed > 0:
                    speed_mbps = (self.volume_bytes_written / (1024 * 1024)) / elapsed
                    speed_str = f" | {speed_mbps:.1f} MB/s"

        self.current_volume_index += 1
        vol_path = Path(f"{self.base_path}.{self.current_volume_index:03d}")
        # Buffer de escrita de 1MB
        self.current_fp = open(vol_path, "wb", buffering=1024*1024)
        self.current_size = 0
        self.current_volume_name = vol_path.name
        
        import time
        self.volume_start_time = time.time()
        self.volume_bytes_written = 0
        
        print(f"[VolumeWriter] Abrindo novo volume: {vol_path}{speed_str}")

    def write_frame(
        self,
        frame_id: int,
        uncompressed_size: int,
        compressed_bytes: bytes,
    ) -> FrameMeta:
        size = len(compressed_bytes)

        if self.current_fp is None or (self.current_size + size > self.max_volume_size):
            self._open_new_volume()

        offset = self.current_size
        self.current_fp.write(compressed_bytes)
        self.current_size += size
        self.volume_bytes_written += size

        return FrameMeta(
            frame_id=frame_id,
            volume_name=self.current_volume_name or "",
            offset=offset,
            compressed_size=size,
            uncompressed_size=uncompressed_size,
        )

    def close(self) -> None:
        if self.current_fp is not None:
            self.current_fp.close()
            self.current_fp = None


# ---------------------------------------------------------------------------
# Geração de frames lógicos
# ---------------------------------------------------------------------------

def generate_frames(entries: Iterable[FileEntry], frame_size: int) -> Iterator[tuple[int, bytes | memoryview]]:
    """
    Gera frames lógicos de tamanho máximo `frame_size` bytes.
    Usa small_files_buffer para concatenar pequenos arquivos e memoryview para grandes.
    """
    frame_size = int(frame_size)
    frame_id = 0
    small_files_buffer = bytearray()
    buffer_entries: List[FileEntry] = []  # Arquivos atualmente no buffer
    
    skipped_files = []

    for entry in entries:
        if entry.is_duplicate:
            continue

        try:
            file_size = entry.size
            if file_size == 0:
                continue
            
            with open(entry.path_abs, "rb") as f:
                # Fase 1: Completar buffer existente
                if len(small_files_buffer) > 0 or file_size < frame_size:
                    wanted = frame_size - len(small_files_buffer)
                    chunk = f.read(wanted)
                    small_files_buffer.extend(chunk)
                    
                    if len(small_files_buffer) >= frame_size:
                        # Marcar arquivos do buffer
                        for e in buffer_entries:
                            if e.start_frame_id is None:
                                e.start_frame_id = frame_id
                        # Se o arquivo atual contribuiu, ele também começa aqui (se ainda não marcado)
                        if entry.start_frame_id is None:
                            entry.start_frame_id = frame_id
                            
                        buffer_entries = [] # Buffer esvaziado (exceto sobra deste arquivo, handled below)
                        
                        yield frame_id, bytes(small_files_buffer[:frame_size])
                        del small_files_buffer[:frame_size]
                        frame_id += 1
                        
                        if remaining <= 0:
                            continue
                    else:
                        # Buffer ainda não cheio: adicionar este arquivo à lista de pendentes
                        buffer_entries.append(entry)
                
                # Fase 2: Processar restante
                while True:
                    curr_pos = f.tell()
                    remaining_file = file_size - curr_pos
                    
                    if remaining_file == 0:
                        break

                    if remaining_file < frame_size:
                        chunk = f.read()
                        small_files_buffer.extend(chunk)
                        if entry.start_frame_id is None:
                             entry.start_frame_id = frame_id # Começa no frame atual (que está sendo enchido)
                        buffer_entries.append(entry)
                        break
                    
                    # Frame inteiro direto
                    if entry.start_frame_id is None:
                        entry.start_frame_id = frame_id
                    
                    chunk = f.read(frame_size)
                    if not chunk: break
                    yield frame_id, chunk
                    frame_id += 1
            
        except Exception as e:
            skipped_files.append((entry.path_rel, str(e)))
            continue

    if skipped_files:
        print(f"\n[IO] Resumo: {len(skipped_files)} arquivo(s) pulado(s) durante leitura")
        
    if small_files_buffer:
        # Marcar restantes
        for e in buffer_entries:
            if e.start_frame_id is None:
                e.start_frame_id = frame_id
        yield frame_id, bytes(small_files_buffer)


# ---------------------------------------------------------------------------
# Escrita do arquivo de índice
# ---------------------------------------------------------------------------

def _dictionary_to_serializable(dictionary: Any) -> Any:
    if dictionary is None:
        return None
    to_dict = getattr(dictionary, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    if hasattr(dictionary, "__dict__"):
        try:
            return dict(dictionary.__dict__)
        except Exception:
            pass
    return str(dictionary)


def write_index_file(
    index_path: Path,
    files: List[FileEntry],
    frames: List[FrameMeta],
    dictionary: Any,
    params: Dict[str, Any],
) -> None:
    """
    Grava o arquivo de índice JSON contendo metadados (Legacy/Debug).
    """
    files_list = [
        {
            "path_rel": f.path_rel,
            "size": f.size,
            "is_duplicate": f.is_duplicate,
            "original": f.original_path_rel,
        }
        for f in files
    ]

    frames_list = [
        {
            "frame_id": fr.frame_id,
            "volume_name": fr.volume_name,
            "offset": fr.offset,
            "compressed_size": fr.compressed_size,
            "uncompressed_size": fr.uncompressed_size,
        }
        for fr in frames
    ]

    index_obj: Dict[str, Any] = {
        "files": files_list,
        "frames": frames_list,
        "dictionary": _dictionary_to_serializable(dictionary),
        "params": dict(params),
    }

    index_path = index_path.resolve()
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, ensure_ascii=False, indent=2)

    print(f"[Index] Arquivo de índice gravado em: {index_path}")


def embed_index_file(
    output_base: Path,
    last_volume_name: str,
    files: List[FileEntry],
    frames: List[FrameMeta],
    dictionary: Any,
    params: Dict[str, Any],
    max_volume_size: Optional[int] = None,
) -> str:
    """
    Gera o índice SQLite (GPU_IDX3), comprime com gzip (stream) e anexa ao final do último volume.
    
    Se o último volume estiver >40% ocupado e max_volume_size for especificado,
    cria um novo volume para evitar extrapolar o limite.
    
    Returns:
        Nome do volume onde o índice foi gravado (pode ser diferente de last_volume_name)
    """
    import struct
    import gzip
    import sqlite3
    import shutil
    import json
    import time
    import re
    
    last_vol_path = output_base.parent / last_volume_name
    
    # Verificar ocupação do último volume e decidir se precisa criar novo
    target_vol_path = last_vol_path
    target_vol_name = last_volume_name
    
    if max_volume_size and last_vol_path.exists():
        current_size = last_vol_path.stat().st_size
        occupancy = current_size / max_volume_size
        
        if occupancy > 0.4:  # Se >40% ocupado, criar novo volume
            # Extrair número do volume atual e incrementar
            match = re.match(r"(.*)\.(\d{3})$", last_volume_name)
            if match:
                prefix = match.group(1)
                current_num = int(match.group(2))
                new_num = current_num + 1
                new_vol_name = f"{prefix}.{new_num:03d}"
                new_vol_path = output_base.parent / new_vol_name
                
                print(f"[Index] Volume atual em {occupancy*100:.1f}% ocupado (>{40}%). Criando novo volume: {new_vol_name}")
                
                # Criar o novo volume vazio
                with open(new_vol_path, "wb") as f:
                    pass
                
                target_vol_path = new_vol_path
                target_vol_name = new_vol_name
    
    print(f"[Index] Gerando índice SQLite (GPU_IDX3) em {target_vol_name}...")

    tmp_dir = Path("tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = tmp_dir / f"index_{int(time.time())}.db"
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("PRAGMA journal_mode = OFF")
        c.execute("PRAGMA synchronous = OFF")
        
        c.execute("CREATE TABLE sddls (id INTEGER PRIMARY KEY, sddl TEXT UNIQUE)")
        
        c.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path_rel TEXT,
                size INTEGER,
                compressed_size INTEGER,
                offset INTEGER,
                volume_name TEXT,
                is_duplicate BOOLEAN,
                original_path TEXT,
                parent_path TEXT,
                sddl_id INTEGER,
                attrs INTEGER,
                ctime REAL,
                mtime REAL,
                atime REAL,
                sframe_id INTEGER
            )
        """)
        c.execute("CREATE INDEX idx_path ON files(path_rel)")
        c.execute("CREATE INDEX idx_parent ON files(parent_path)")
        
        c.execute("CREATE TABLE frames (id INTEGER PRIMARY KEY, vol TEXT, offset INTEGER, size INTEGER, original_size INTEGER)")
        
        c.execute("CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT)")
        
        sddl_map = {} 
        next_sddl_id = 1
        files_batch = []
        BATCH_SIZE = 10000
        
        for f in files:
            sddl_text = getattr(f, 'sddl', None)
            sddl_id = None
            
            if sddl_text:
                if sddl_text not in sddl_map:
                    c.execute("INSERT INTO sddls (id, sddl) VALUES (?, ?)", (next_sddl_id, sddl_text))
                    sddl_map[sddl_text] = next_sddl_id
                    next_sddl_id += 1
                sddl_id = sddl_map[sddl_text]
            
            # Helper seguro para atributos (pode vir de dict ou objeto)
            def get_attr(obj, name, default):
                return getattr(obj, name, default) if not isinstance(obj, dict) else obj.get(name, default)

            attrs = get_attr(f, 'attrs', 0)
            ctime = get_attr(f, 'ctime', 0.0)
            mtime = get_attr(f, 'mtime', 0.0)
            atime = get_attr(f, 'atime', 0.0)
            
            path_str = str(f.path) if hasattr(f, 'path') else f['path']
            path_str = path_str.replace('\\', '/')
            
            if '/' in path_str:
                parent_str = path_str.rsplit('/', 1)[0]
            else:
                parent_str = ""
            
            size_val = f.original_size if getattr(f, 'original_size', None) is not None else get_attr(f, 'size', 0)
            offset_val = get_attr(f, 'offset', 0)
            vol_val = get_attr(f, 'volume', "")
            is_dup = get_attr(f, 'is_duplicate', False)
            comp_size = get_attr(f, 'compressed_size', 0)
            orig_path = get_attr(f, 'original_path_rel', None)
            
            sframe_id = get_attr(f, 'start_frame_id', None)
            
            files_batch.append((
                path_str, size_val, comp_size, offset_val, vol_val, is_dup, orig_path, parent_str,
                sddl_id, attrs, ctime, mtime, atime, sframe_id
            ))
            
            if len(files_batch) >= BATCH_SIZE:
                c.executemany("""
                    INSERT INTO files (path_rel, size, compressed_size, offset, volume_name, 
                                     is_duplicate, original_path, parent_path, sddl_id, attrs, ctime, mtime, atime, sframe_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, files_batch)
                files_batch = []
        
        if files_batch:
            c.executemany("""
                    INSERT INTO files (path_rel, size, compressed_size, offset, volume_name, 
                                     is_duplicate, original_path, parent_path, sddl_id, attrs, ctime, mtime, atime, sframe_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, files_batch)
        
        if frames:
            c.executemany("INSERT INTO frames VALUES (?,?,?,?,?)", [
                (fr.frame_id, fr.volume_name, fr.offset, fr.compressed_size, fr.uncompressed_size) 
                if hasattr(fr, 'frame_id') else 
                (fr['id'], fr['vol'], fr['off'], fr['size'], fr['osize']) 
                for fr in frames
            ])
            
        if params:
            c.execute("INSERT INTO kv_store VALUES (?, ?)", ("params", json.dumps(params)))
        if dictionary:
            c.execute("INSERT INTO kv_store VALUES (?, ?)", ("dictionary", json.dumps(dictionary)))
            
        conn.commit()
        conn.close()
        
        if not target_vol_path.exists():
             with open(target_vol_path, "wb") as f: pass
             
        start_offset = target_vol_path.stat().st_size
        
        with open(db_path, "rb") as f_in:
            with gzip.open(target_vol_path, "ab", compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        final_offset = target_vol_path.stat().st_size
        index_size = final_offset - start_offset
        
        footer = struct.pack('<QQ8s', start_offset, index_size, b'GPU_IDX3')
        
        with open(target_vol_path, "ab") as f:
            f.write(footer)
            
        print(f"[Index] Footer GPU_IDX3 gravado em {target_vol_name}. Offset={start_offset}, Size={index_size}")
        
        return target_vol_name
        
    except Exception as e:
        print(f"Erro ao gerar índice SQLite: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if db_path.exists():
            try:
                db_path.unlink()
            except:
                pass


def _enable_privilege(priv_name: str) -> bool:
    import sys
    if sys.platform != 'win32': return False

    import ctypes
    from ctypes import wintypes
    
    TOKEN_ADJUST_PRIVILEGES = 0x0020
    TOKEN_QUERY = 0x0008
    SE_PRIVILEGE_ENABLED = 0x00000002
    
    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    try:
        advapi32 = ctypes.windll.advapi32
        kernel32 = ctypes.windll.kernel32
        
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        
        token = wintypes.HANDLE()
        current_process = kernel32.GetCurrentProcess()
        
        if not advapi32.OpenProcessToken(current_process, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(token)):
             return False

        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, priv_name, ctypes.byref(luid)):
             kernel32.CloseHandle(token)
             return False
             
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0].Luid = luid
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
        
        if not advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
             kernel32.CloseHandle(token)
             return False
             
        error = kernel32.GetLastError()
        kernel32.CloseHandle(token)
        return (error != 1300)
        
    except Exception:
        return False

def enable_se_backup_privilege() -> bool:
    return _enable_privilege("SeBackupPrivilege")

def enable_se_restore_privilege() -> bool:
    return _enable_privilege("SeRestorePrivilege")

def enable_se_security_privilege() -> bool:
    return _enable_privilege("SeSecurityPrivilege")
