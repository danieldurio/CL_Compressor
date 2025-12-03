from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any, Optional, Tuple
import json


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


def scan_directory(root: Path) -> List[FileEntry]:
    """
    Varre recursivamente a pasta `root` e monta a lista de FileEntry.
    """
    entries: List[FileEntry] = []
    root = root.resolve()

    for path_abs in root.rglob("*"):
        if not path_abs.is_file():
            continue

        try:
            size = path_abs.stat().st_size
        except OSError:
            continue

        # Caminho relativo (sempre com /) em relação à raiz
        path_rel = str(path_abs.relative_to(root)).replace("\\", "/")
        entries.append(FileEntry(path_abs=path_abs, path_rel=path_rel, size=size))

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
        self.current_fp = open(vol_path, "wb")
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


def generate_frames(entries: Iterable[FileEntry], frame_size: int) -> Iterator[tuple[int, bytes]]:
    """
    Gera frames lógicos de tamanho máximo `frame_size` bytes a partir dos
    arquivos de entrada.

    Cada frame é simplesmente um pedaço do "stream" concatenado de todos os
    arquivos, na ordem de `entries`.

    Retorna tuplas (frame_id, frame_data).
    """
    frame_size = int(frame_size)
    buffer = bytearray()
    frame_id = 0

    for entry in entries:
        if entry.is_duplicate:
            continue

        with open(entry.path_abs, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)  # lê em blocos de 64 KB
                if not chunk:
                    break

                buffer.extend(chunk)

                while len(buffer) >= frame_size:
                    out = bytes(buffer[:frame_size])
                    del buffer[:frame_size]
                    yield frame_id, out
                    frame_id += 1

    # Sobrou algo no buffer (último frame parcial)
    if buffer:
        yield frame_id, bytes(buffer)


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
    Gera o índice JSON, comprime com zlib e anexa ao final do último volume.
    Adiciona um rodapé (Footer) fixo para localização.
    
    Footer Format (24 bytes):
    [Offset Index (8 bytes)] [Size Index (8 bytes)] [Magic (8 bytes)]
    Magic = b'GPU_IDX1'
    """
    import zlib
    import struct
    
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

    # 1. Serializar JSON
    json_bytes = json.dumps(index_obj, ensure_ascii=False).encode('utf-8')
    
    # 2. Comprimir
    compressed_index = zlib.compress(json_bytes, level=9)
    c_size = len(compressed_index)
    
    # 3. Anexar ao último volume
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
    
    print(f"[Index] Incorporando índice comprimido ({c_size} bytes) em {last_volume_name}...")
    
    with open(last_vol_path, "ab") as f:
        f.write(compressed_index)
        
        # 4. Escrever Footer
        # Offset (8), Size (8), Magic (8)
        magic = b'GPU_IDX1'
        footer = struct.pack('<QQ8s', start_offset, c_size, magic)
        f.write(footer)
        
    print(f"[Index] Footer gravado. Offset={start_offset}, Size={c_size}")
