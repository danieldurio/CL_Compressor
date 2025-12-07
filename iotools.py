from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any, Optional, Tuple
import json
import os
import threading
from queue import Queue, Empty


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

    # Helpers opcionais (caso em algum ponto usem .path ou .rel_path)
    @property
    def path(self) -> Path:
        return self.path_abs

    @property
    def rel_path(self) -> str:
        return self.path_rel

# Carregar configurações do config.txt centralizado
import config_loader

# Configuração do scanner paralelo (carregada de config.txt)
NUM_SCAN_WORKERS = config_loader.get_num_scan_workers()


def scan_directory_parallel(root: Path) -> List[FileEntry]:
    """
    Scanner paralelo com work-stealing.
    Usa 3 workers fixos que subdividem diretórios grandes dinamicamente.
    
    Cada worker processa apenas 1 nível por vez:
    - Arquivos → adiciona à lista local
    - Subdiretórios → adiciona à fila para outros workers
    
    Isso garante balanceamento de carga mesmo quando um subdir é gigante.
    """
    root = root.resolve()
    
    # Fila thread-safe de diretórios a processar
    work_queue: Queue[Path] = Queue()
    
    # Lock para contagem de workers ativos (para detectar término)
    active_workers = [0]
    active_lock = threading.Lock()
    all_done = threading.Event()
    
    # Resultados por worker (sem compartilhamento durante processamento)
    results: List[List[FileEntry]] = [[] for _ in range(NUM_SCAN_WORKERS)]
    
    # Estatísticas thread-safe
    stats = {"dirs_processed": 0, "files_found": 0}
    stats_lock = threading.Lock()
    
    def process_path(path_abs: Path) -> Optional[FileEntry]:
        """Converte path em FileEntry."""
        try:
            if not path_abs.is_file():
                return None
            size = path_abs.stat().st_size
            path_rel = str(path_abs.relative_to(root)).replace("\\", "/")
            return FileEntry(path_abs=path_abs, path_rel=path_rel, size=size)
        except (PermissionError, FileNotFoundError, OSError):
            return None
    
    def worker(worker_id: int):
        """Worker que processa diretórios da fila."""
        local_entries = results[worker_id]
        local_dirs_count = 0
        local_files_count = 0
        
        while not all_done.is_set():
            try:
                # Tentar pegar trabalho da fila
                current_dir = work_queue.get(timeout=0.05)
            except Empty:
                # Fila vazia - verificar se todos terminaram
                with active_lock:
                    if active_workers[0] == 0 and work_queue.empty():
                        all_done.set()
                continue
            
            # Marcar como ativo
            with active_lock:
                active_workers[0] += 1
            
            try:
                # Processar diretório atual (apenas 1 nível)
                with os.scandir(current_dir) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                # Subdiretório → adicionar à fila para outros workers
                                work_queue.put(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                # Arquivo → processar imediatamente
                                file_entry = process_path(Path(entry.path))
                                if file_entry:
                                    local_entries.append(file_entry)
                                    local_files_count += 1
                        except (PermissionError, OSError):
                            pass
                local_dirs_count += 1
            except (PermissionError, OSError):
                pass
            finally:
                # Marcar como inativo
                with active_lock:
                    active_workers[0] -= 1
                work_queue.task_done()
        
        # Atualizar estatísticas globais
        with stats_lock:
            stats["dirs_processed"] += local_dirs_count
            stats["files_found"] += local_files_count
    
    # Inicializar fila com diretório raiz
    work_queue.put(root)
    
    # Iniciar workers
    threads = []
    for i in range(NUM_SCAN_WORKERS):
        t = threading.Thread(target=worker, args=(i,), name=f"ScanWorker-{i}", daemon=True)
        t.start()
        threads.append(t)
    
    # Aguardar término de todos workers
    for t in threads:
        t.join()
    
    # Merge: combinar resultados de todos workers
    all_entries: List[FileEntry] = []
    for worker_entries in results:
        all_entries.extend(worker_entries)
    
    # Ordenar para consistência
    all_entries.sort(key=lambda e: e.path_rel.lower())
    
    print(f"[SCAN] ✓ {len(all_entries)} arquivos encontrados | {stats['dirs_processed']} dirs | {NUM_SCAN_WORKERS} workers")
    return all_entries


def scan_directory(root: Path, parallel: bool = True) -> List[FileEntry]:
    """
    Varre recursivamente a pasta `root` e monta a lista de FileEntry.
    
    Args:
        root: Diretório raiz para escanear
        parallel: Se True (padrão), usa scanning paralelo com work-stealing.
                  Se False, usa scanning serial tradicional.
    
    Arquivos/pastas inacessíveis são ignorados com aviso no console.
    Motivos de skip: permissão negada, arquivo em uso, caminho inválido, etc.
    """
    if parallel:
        return scan_directory_parallel(root)
    entries: List[FileEntry] = []
    root = root.resolve()
    
    # Estatísticas de erros
    skipped_count = 0
    skipped_reasons = {}

    try:
        all_paths = list(root.rglob("*"))
    except PermissionError as e:
        print(f"[SCAN] ⚠️ SKIP: Sem permissão para acessar '{root}': {e}")
        return entries
    except OSError as e:
        print(f"[SCAN] ⚠️ SKIP: Erro ao acessar '{root}': {e}")
        return entries

    for path_abs in all_paths:
        try:
            # Verificar se é arquivo (pode falhar se path inacessível)
            if not path_abs.is_file():
                continue

            # Tentar obter tamanho (pode falhar em arquivos em uso/bloqueados)
            size = path_abs.stat().st_size
            
            # Caminho relativo (sempre com /) em relação à raiz
            path_rel = str(path_abs.relative_to(root)).replace("\\", "/")
            entries.append(FileEntry(path_abs=path_abs, path_rel=path_rel, size=size))
            
        except PermissionError:
            skipped_count += 1
            reason = "Permissão negada"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            print(f"[SCAN] ⚠️ SKIP: {path_abs.name} - {reason}")
            continue
            
        except FileNotFoundError:
            skipped_count += 1
            reason = "Arquivo não encontrado/movido"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            print(f"[SCAN] ⚠️ SKIP: {path_abs.name} - {reason}")
            continue
            
        except OSError as e:
            skipped_count += 1
            # Detectar motivos específicos do Windows
            error_code = getattr(e, 'winerror', None) or e.errno
            
            if error_code == 32:  # ERROR_SHARING_VIOLATION
                reason = "Arquivo em uso por outro processo"
            elif error_code == 5:  # ERROR_ACCESS_DENIED
                reason = "Acesso negado pelo sistema"
            elif error_code == 123:  # ERROR_INVALID_NAME
                reason = "Nome de arquivo inválido"
            elif error_code == 1920:  # ERROR_CANT_ACCESS_FILE
                reason = "Arquivo não pode ser acessado"
            else:
                reason = f"Erro de I/O ({error_code}): {e}"
                
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            print(f"[SCAN] ⚠️ SKIP: {path_abs.name} - {reason}")
            continue
            
        except Exception as e:
            skipped_count += 1
            reason = f"Erro inesperado: {type(e).__name__}"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            print(f"[SCAN] ⚠️ SKIP: {path_abs.name} - {reason}: {e}")
            continue

    # Relatório de arquivos pulados
    if skipped_count > 0:
        print(f"\n[SCAN] 📊 Resumo: {skipped_count} arquivo(s) ignorado(s)")
        for reason, count in skipped_reasons.items():
            print(f"       - {reason}: {count}")
        print()

    # Ordena para estabilidade (opcional, mas ajuda em testes)
    entries.sort(key=lambda e: e.path_rel.lower())
    return entries


def estimate_total_size(entries: Iterable[FileEntry]) -> int:
    """
    Soma o tamanho de todos os arquivos da lista.
    Usado pelo preprocessor para saber o total de bytes.
    """
    return sum(e.size for e in entries)


# ---------------------------------------------------------------------------
# Detecção de dispositivos OpenCL (para o main.py)
# ---------------------------------------------------------------------------

@dataclass
@dataclass
class OpenCLInfo:
    available: bool
    devices: List[Tuple[str, str]]  # (platform_name, device_name)

    @property
    def num_devices(self) -> int:
        """
        Compatível com main.py:
            print(f"... {cl_info.num_devices} device(s).")
        """
        return len(self.devices)

    @property
    def device_names(self) -> List[str]:
        """
        Devolve uma lista de strings prontas para print, ex:
            "NVIDIA CUDA :: NVIDIA GeForce RTX 3070"
        Compatível com main.py:
            for name in cl_info.device_names:
                print("        -", name)
        """
        return [f"{plat} :: {dev}" for (plat, dev) in self.devices]


def detect_opencl_devices() -> OpenCLInfo:
    """
    Detecta dispositivos OpenCL disponíveis.

    Retorna um OpenCLInfo:

        - available: bool
        - devices: List[(platform_name, device_name)]

    Compatível com o main.py:
        cl_info = detect_opencl_devices()
        if cl_info.available:
            ...
    """
    try:
        import pyopencl as cl  # type: ignore
    except ImportError:
        # Sem pyopencl, sem OpenCL :)
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
# Metadados de frames e escrita de volumes .001, .002, ...
# ---------------------------------------------------------------------------


@dataclass
class FrameMeta:
    """
    Metadados de um frame dentro de um volume.

    - frame_id:          índice lógico do frame (0,1,2,...)
    - volume_name:       nome do arquivo de volume (ex: "saida.001")
    - offset:            posição (em bytes) no volume onde o frame começa
    - compressed_size:   tamanho em bytes do frame comprimido
    - uncompressed_size: tamanho original em bytes do frame
    """
    frame_id: int
    volume_name: str
    offset: int
    compressed_size: int
    uncompressed_size: int


class VolumeWriter:
    """
    Responsável por gravar os frames comprimidos em volumes do tipo:

        <base>.001, <base>.002, ...

    respeitando o limite de tamanho em bytes (max_volume_size).
    """

    def __init__(self, base_path: Path, max_volume_size: int) -> None:
        self.base_path = base_path              # ex: F:\...\compressor\saida
        self.max_volume_size = int(max_volume_size)
        self.current_volume_index = 0           # 1 -> .001, 2 -> .002, ...
        self.current_fp: Optional[object] = None
        self.current_size = 0                   # bytes já escritos no volume atual
        self.current_volume_name: Optional[str] = None
        
        # Para cálculo de velocidade
        self.volume_start_time: Optional[float] = None
        self.volume_bytes_written = 0

    def _open_new_volume(self) -> None:
        # Calcular velocidade do volume anterior (se houver)
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
        # Buffer de escrita de 1MB (otimização para throughput)
        self.current_fp = open(vol_path, "wb", buffering=1024*1024)
        self.current_size = 0
        self.current_volume_name = vol_path.name  # só o nome do arquivo
        
        # Reset timing para o novo volume
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
        """
        Escreve um frame comprimido no volume atual, ou em um novo volume
        se o próximo frame iria estourar o limite de tamanho.
        """
        size = len(compressed_bytes)

        # Se não há volume aberto OU este frame faria passar do limite,
        # abre um novo volume ANTES de escrever.
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
# Geração de frames lógicos a partir dos arquivos de entrada
# ---------------------------------------------------------------------------


def generate_frames(entries: Iterable[FileEntry], frame_size: int) -> Iterator[tuple[int, bytes | memoryview]]:
    """
    Gera frames lógicos de tamanho máximo `frame_size` bytes a partir dos
    arquivos de entrada.

    OTIMIZAÇÃO DE PERFORMANCE:
    - Minimiza cópias usando `memoryview` e `mmap`.
    - Mantém mmaps vivos em um deque para evitar segfaults ao cruzar fronteiras de arquivos
      enquanto o consumidor (compressor) ainda processa o buffer anterior.
    
    Retorna tuplas (frame_id, data).
    `data` pode ser `bytes` (para arquivos pequenos/bufferizados) ou `memoryview` (zero-copy).
    """
    frame_size = int(frame_size)
    frame_id = 0
    
    # Buffer para acumular arquivos pequenos (< frame_size)
    # Quando atinge frame_size, emite como bytes
    small_files_buffer = bytearray()
    
    import mmap
    from collections import deque
    
    # Keep-Alive Deque: (mmap_obj, file_obj)
    # Mantém referências para evitar que o Python feche o arquivo/mmap prematuramente
    # enquanto o consumidor (compressor) está lendo o memoryview.
    # Tamanho 64 cobre ~2-3 batches de compressão (assumindo batch 24-32).
    mmap_keep_alive = deque(maxlen=64)
    
    # Estatísticas de erros
    skipped_files = []

    for entry in entries:
        if entry.is_duplicate:
            continue

        try:
            # Estratégia híbrida baseada no tamanho do arquivo
            file_size = entry.size
            if file_size == 0:
                continue
            
            # Se arquivo é muito pequeno OU já temos dados parciais no buffer
            # usamos estratégia de cópia para concatenar
            USE_MMAP_THRESHOLD = 50 * 1024 * 1024  # Apenas mmap arquivos > 50MB
            # Se o arquivo for menor que o frame_size, ele certamente vai pro buffer,
            # então nem adianta mmap.
            
            # Decisão Simplificada:
            # Se tivermos buffer pendente, lemos para completar o buffer.
            # Se o buffer esvaziar e o arquivo ainda tiver muito dados, fazemos mmap do restante?
            # Mmap é complexo de misturar com buffer.
            # Nova Estratégia:
            # 1. Tentar completar 'small_files_buffer' com read() normal.
            # 2. Se buffer completou -> yield bytes.
            # 3. Se sobrou muito dado no arquivo (> frame_size), usar mmap para gerar frames diretos.
            
            with open(entry.path_abs, "rb") as f:
                
                # Fase 1: Completar buffer existente (ou encher se arquivo pequeno)
                if len(small_files_buffer) > 0 or file_size < frame_size:
                    wanted = frame_size - len(small_files_buffer)
                    # Lê o que der (limitado pelo tamanho do arquivo)
                    chunk = f.read(wanted)
                    small_files_buffer.extend(chunk)
                    
                    if len(small_files_buffer) >= frame_size:
                        yield frame_id, bytes(small_files_buffer[:frame_size])
                        del small_files_buffer[:frame_size]
                        frame_id += 1
                        
                        # Se ainda tem dados no arquivo após completar buffer...
                        remaining = file_size - len(chunk)
                        if remaining > 0:
                            # O ponteiro do arquivo já avançou 'len(chunk)'.
                            # Podemos continuar lendo ou mudar para mmap.
                            pass
                        else:
                            continue # Arquivo acabou
                
                # Fase 2: Processar restante do arquivo (se houver)
                # Se chegamos aqui, small_files_buffer está vazio (ou foi emitido).
                # E o arquivo ainda tem conteúdo.
                
                curr_pos = f.tell()
                remaining_file = file_size - curr_pos
                
                if remaining_file == 0:
                    continue

                if remaining_file < frame_size:
                    # Sobra insuficiente para um frame inteiro -> joga no buffer
                    chunk = f.read()
                    small_files_buffer.extend(chunk)
                    continue
                
                # Se tem pelo menos um frame inteiro, vale a pena mmap (zero-copy)
                try:
                    # Windows: mmap requer size > 0. Se size=0 já filtramos.
                    # length=0 mapreia todo o arquivo.
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    
                    # Precisamos manter o arquivo E o mmap abertos
                    # O truque é 'transferir' a posse deles para o deque
                    # Para isso, duplicamos o handle ou simplesmente evitamos o close automático do 'with open'?
                    # O 'with open' vai fechar no final do bloco.
                    # Mmap requer file descriptor aberto? No Windows, mmap mantém o handle do arquivo,
                    # mas se fecharmos o objeto python 'f', o fd subjacente fecha? Sim.
                    # Então não podemos usar 'with open' se queremos yieldar mmap vivos.
                    
                    # CORREÇÃO: Não podemos usar 'with open' aqui se formos colocar no keep_alive.
                    # Mas já abrimos lá em cima.
                    # Solução: Clonar o file handle seria ideal, mas complexo inter-plataforma.
                    # Workaround: Usar slice *copiado* se o mmap for arriscado? Não, queremos zero-copy.
                    
                    # Vamos assumir que processamos o mmap COMPLETO aqui dentro.
                    # O problema é: yield pausa a função. O 'with open' NÃO fecha enquanto a função está pausada.
                    # O 'with open' só fecha quando o bloco termina.
                    # O bloco só termina quando o loop avança para o próximo arquivo.
                    # Quando avança, o consumidor pode ainda ter a referência ao mmap anterior.
                    
                    # SOLUÇÃO REAL:
                    # Em vez de Yieldar slices do mmap que dependem do arquivo aberto,
                    # nós criamos um mmap_obj que *preserva* o arquivo.
                    # No Python, mmap.mmap(f.fileno(), ...) duplica o handle no OS (Windows/Linux).
                    # Então se f.close() for chamado, o mmap continua válido?
                    # Testes dizem: No Windows, sim. No Linux, sim (ref count).
                    # Mas o objeto Python 'mmap' precisa estar vivo.
                    
                    # Então:
                    # 1. Criar mmap
                    # 2. Adicionar mmap ao keep_alive
                    # 3. Yield slices
                    # 4. DEIXAR o 'with open' fechar 'f'. O mmap mantém o arquivo vivo no OS.
                    
                    # Mover o cursor do mmap para onde o 'f' parou
                    cursor = curr_pos
                    
                    while remaining_file >= frame_size:
                        # Yield slice (zero-copy view)
                        # Yielding slicing de mmap cria bytes? NÃO. Cria bytes no Python < 3.
                        # No Python 3, mm[x:y] retorna bytes (cópia).
                        # Para zero-copy, precisamos memoryview(mm)[x:y].
                        mv = memoryview(mm)
                        yield frame_id, mv[cursor:cursor+frame_size]
                        
                        frame_id += 1
                        cursor += frame_size
                        remaining_file -= frame_size
                    
                    # Adicionar ao keep_alive para o mmap não ser GC'ed imediatamente
                    mmap_keep_alive.append(mm)
                    
                    # Sobra final -> buffer
                    if remaining_file > 0:
                        small_files_buffer.extend(mm[cursor:cursor+remaining_file])
                        
                except (ValueError, OSError):
                    # Fallback se mmap falhar (ex: FS não suporta, 0 bytes, etc)
                    # Voltar para o ponto correto
                    f.seek(curr_pos)
                    while True:
                        chunk = f.read(frame_size) # Lê bloco (cópia)
                        if not chunk: break
                        
                        if len(chunk) == frame_size:
                            yield frame_id, chunk
                            frame_id += 1
                        else:
                            small_files_buffer.extend(chunk)
                            break
                            
        except PermissionError:
            reason = "Permissão negada"
            skipped_files.append((entry.path_rel, reason))
            # print(f"[IO] ⚠️ SKIP: {entry.path_rel} - {reason}")
            continue
            
        except FileNotFoundError:
            reason = "Arquivo não encontrado/movido"
            skipped_files.append((entry.path_rel, reason))
            continue
            
        except Exception as e:
            reason = f"Erro inesperado ({type(e).__name__}): {e}"
            skipped_files.append((entry.path_rel, reason))
            print(f"[IO] ⚠️ SKIP: {entry.path_rel} - {reason}")
            continue

    # Relatório final de arquivos pulados
    if skipped_files:
        print(f"\n[IO] 📊 Resumo: {len(skipped_files)} arquivo(s) pulado(s) durante leitura")
        
    # Sobrou algo no buffer (último frame parcial)
    if small_files_buffer:
        yield frame_id, bytes(small_files_buffer)


# ---------------------------------------------------------------------------
# Escrita do arquivo de índice (.index.json)
# ---------------------------------------------------------------------------


def _dictionary_to_serializable(dictionary: Any) -> Any:
    """
    Converte o objeto Dictionary para algo serializável em JSON.

    - Se tiver método to_dict(), usa.
    - Senão, tenta usar __dict__.
    - Senão, converte para str().
    """
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

    # Fallback bem genérico
    return str(dictionary)


def write_index_file(
    index_path: Path,
    files: List[FileEntry],
    frames: List[FrameMeta],
    dictionary: Any,
    params: Dict[str, Any],
) -> None:
    """
    Grava o arquivo de índice JSON contendo metadados.
    (Mantido para compatibilidade ou debug)
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
) -> None:
    """
    Gera o índice STREAMING (GPU_IDX2), comprime com gzip (stream) e anexa ao final do último volume.
    
    Formato GPU_IDX2:
    - Linha 1: Header JSON (version, params, dictionary)
    - Linhas 2..N: FileEntries (JSONL)
    - Linhas N+1..M: FrameMeta (JSONL)
    
    Isso evita carregar listas gigantes em memória para o json.dumps(),
    mantendo O(1) de memória independente do número de arquivos.
    
    Footer (24 bytes):
    [Offset (8)] [Size (8)] [Magic (8)]
    Magic = b'GPU_IDX2'
    """
    import struct
    import gzip
    
    # Se last_volume_name for vazio (nenhum frame gerado?), cria um .001
    if not last_volume_name:
        last_volume_name = f"{output_base.name}.001"
        
    last_vol_path = output_base.parent / last_volume_name
    
    # Se o arquivo não existir (caso raro de 0 arquivos), cria
    if not last_vol_path.exists():
        with open(last_vol_path, "wb") as f:
            pass
            
    # Obter offset atual (onde começa o índice)
    start_offset = last_vol_path.stat().st_size
    
    print(f"[Index] Incorporando índice STREAMING (GPU_IDX2) em {last_volume_name}...", flush=True)
    
    # Abrir arquivo de saída em modo append
    with open(last_vol_path, "ab") as f_out:
        # Usar GzipFile envolvendo o file handle para streamar a compressão
        # mtime=0 para saída determinística
        with gzip.GzipFile(fileobj=f_out, mode="wb", compresslevel=9, mtime=0) as gz:
            
            # 1. Header
            header = {
                "version": 2,
                "type": "GPU_IDX2",
                "params": dict(params),
                "dictionary": _dictionary_to_serializable(dictionary),
                "count_files": len(files),
                "count_frames": len(frames)
            }
            gz.write(json.dumps(header, ensure_ascii=False).encode('utf-8'))
            gz.write(b'\n')
            
            # 2. Files Stream
            # Escreve um objeto JSON por linha
            for fp in files:
                # Otimização: Criar dict manualmente é mais rápido que overhead de dataclasses.asdict
                # Compact key names para economizar espaço? Não, melhor legibilidade/compatibilidade
                # "t": "f" (type file) markers não são estritamente necessários se a ordem for garantida,
                # mas ajudam no parse robusto. Vamos assumir ordem rígida: Header -> Files -> Frames.
                entry = {
                    "path_rel": fp.path_rel,
                    "size": fp.size,
                    "is_duplicate": fp.is_duplicate,
                    "original": fp.original_path_rel
                }
                gz.write(json.dumps(entry, ensure_ascii=False).encode('utf-8'))
                gz.write(b'\n')
                
            # 3. Frames Stream
            for fr in frames:
                entry = {
                    "frame_id": fr.frame_id,
                    "volume_name": fr.volume_name,
                    "offset": fr.offset,
                    "compressed_size": fr.compressed_size,
                    "uncompressed_size": fr.uncompressed_size
                }
                gz.write(json.dumps(entry).encode('utf-8'))
                gz.write(b'\n')
        
        # Após fechar o gz, ele escreveu bytes comprimidos no f_out.
        # Agora escrevemos o footer no f_out.
        
        final_offset = f_out.tell()
        index_size = final_offset - start_offset
        
        # 4. Footer GPU_IDX2
        magic = b'GPU_IDX2' # Novo Magic
        footer = struct.pack('<QQ8s', start_offset, index_size, magic)
        f_out.write(footer)
        
    print(f"[Index] Footer GPU_IDX2 gravado. Offset={start_offset}, Size={index_size}")

# ---------------------------------------------------------------------------
# Privilégios do Sistema (Windows)
# ---------------------------------------------------------------------------

def _enable_privilege(priv_name: str) -> bool:
    """
    Função genérica para habilitar privilégios no Windows.
    """
    import sys
    if sys.platform != 'win32':
        return False

    import ctypes
    from ctypes import wintypes
    
    # Constantes
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
        
        # DEFINIR TIPOS (CRUCIAL PARA 64-BIT)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        
        # 1. Obter handle do token do processo
        token = wintypes.HANDLE()
        current_process = kernel32.GetCurrentProcess()
        
        if not advapi32.OpenProcessToken(current_process, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(token)):
             err = kernel32.GetLastError()
             print(f"[System] Falha ao abrir token do processo (OpenProcessToken). Error Code: {err}")
             return False

        # 2. Obter LUID do privilégio
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, priv_name, ctypes.byref(luid)):
             err = kernel32.GetLastError()
             print(f"[System] Falha ao buscar LUID ({priv_name}). Error Code: {err}")
             kernel32.CloseHandle(token)
             return False
             
        # 3. Ajustar privilégio
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0].Luid = luid
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
        
        if not advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
             err = kernel32.GetLastError()
             print(f"[System] Falha ao ajustar Token ({priv_name}). Error Code: {err}")
             kernel32.CloseHandle(token)
             return False
             
        error = kernel32.GetLastError()
        if error == 1300: # ERROR_NOT_ALL_ASSIGNED
             print(f"[System] AVISO: O usuário não possui '{priv_name}' (Execute como Admin).")
             kernel32.CloseHandle(token)
             return False
             
        kernel32.CloseHandle(token)
        print(f"[System] Privilégio '{priv_name}' habilitado com sucesso.")
        return True
        
    except Exception as e:
        print(f"[System] Erro ao tentar habilitar privilégios: {e}")
        return False

def enable_se_backup_privilege() -> bool:
    return _enable_privilege("SeBackupPrivilege")

def enable_se_restore_privilege() -> bool:
    return _enable_privilege("SeRestorePrivilege")

def enable_se_security_privilege() -> bool:
    return _enable_privilege("SeSecurityPrivilege") # Necessário para SACL
