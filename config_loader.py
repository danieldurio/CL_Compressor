"""
Módulo para carregar configurações do config.txt

Carrega as variáveis de configuração de forma centralizada para todos os módulos.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

# Cache das configurações carregadas
_config_cache: Optional[Dict[str, Any]] = None

def _get_config_path() -> Path:
    """Retorna o caminho para o config.txt (mesmo diretório deste módulo)."""
    return Path(__file__).parent / "config.txt"

def _parse_value(value_str: str) -> Any:
    """Converte string para o tipo Python apropriado."""
    value_str = value_str.strip()
    
    # None
    if value_str.lower() == 'none':
        return None
    
    # Booleanos
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    
    # Inteiros
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Floats
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # String (fallback)
    return value_str

def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Carrega as configurações do config.txt.
    
    Args:
        force_reload: Se True, recarrega do arquivo mesmo se já estiver em cache
        
    Returns:
        Dict com as configurações (chave: valor)
    """
    global _config_cache
    
    if _config_cache is not None and not force_reload:
        return _config_cache
    
    config: Dict[str, Any] = {}
    config_path = _get_config_path()
    
    if not config_path.exists():
        print(f"[Config] config.txt não encontrado em {config_path}. Usando valores padrão.")
        return _get_defaults()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Remover espaços e quebras de linha
                line = line.strip()
                
                # Ignorar linhas vazias e comentários
                if not line or line.startswith('#'):
                    continue
                
                # Ignorar linhas decorativas (═, ╔, ╚, ║, etc.)
                if line[0] in '═╔╚╗╝║├┤┬┴┼─│':
                    continue
                
                # Parsear CHAVE = VALOR
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remover comentário inline (se houver)
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    config[key] = _parse_value(value)
                    
    except Exception as e:
        print(f"[Config] Erro ao ler config.txt: {e}. Usando valores padrão.")
        return _get_defaults()
    
    # Mesclar com defaults para valores não especificados
    defaults = _get_defaults()
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    _config_cache = config
    return config

def _get_defaults() -> Dict[str, Any]:
    """Retorna valores padrão caso o config.txt não exista ou esteja incompleto."""
    return {
        # Compressor
        'FORCE_CPU_MODE': False,
        'COMPRESSOR_BATCH_SIZE': 105,
        'READ_BUFFER_BATCHES': 1,
        'WRITE_BUFFER_BATCHES': 1,
        
        # Decompressor
        'DECOMPRESSOR_BATCH_SIZE': None,
        'MAX_WORKER_THREADS': 2,
        'GPU_FALLBACK_ENABLED': True,
        
        # IO Tools
        'NUM_SCAN_WORKERS': 4,
        'PRE_SCAN_TARGET_DIRS': 1000,
        'SCAN_STATUS_INTERVAL': 5,
        
        # Deduplicator
        'NUM_IO_WORKERS': 4,
        'NUM_READERS': 4,
        'BUFFER_SIZE': 256,
        'HASH_BATCH_SIZE': 128,     # Arquivos por batch de hash GPU (Stage 5)
        'DEDUP_GPU_WORKERS': 8,     # Workers paralelos consumindo GPU
        
        # GPU Kernel
        'HASH_LOG': 20,
        'HASH_CANDIDATES': 7,
        'GOOD_ENOUGH_MATCH': 128,
    }

def get(key: str, default: Any = None) -> Any:
    """
    Obtém um valor de configuração pelo nome.
    
    Args:
        key: Nome da configuração
        default: Valor padrão se não encontrado
        
    Returns:
        Valor da configuração ou default
    """
    config = load_config()
    return config.get(key, default)

# Aliases convenientes para uso direto
def get_compressor_batch_size() -> Optional[int]:
    """Retorna COMPRESSOR_BATCH_SIZE (None = automático)."""
    return get('COMPRESSOR_BATCH_SIZE')

def get_decompressor_batch_size() -> Optional[int]:
    """Retorna DECOMPRESSOR_BATCH_SIZE (None = automático)."""
    return get('DECOMPRESSOR_BATCH_SIZE')

def is_force_cpu_mode() -> bool:
    """Retorna True se modo CPU forçado."""
    return bool(get('FORCE_CPU_MODE', False))

def get_read_buffer_batches() -> int:
    """Retorna READ_BUFFER_BATCHES."""
    return int(get('READ_BUFFER_BATCHES', 1))

def get_write_buffer_batches() -> int:
    """Retorna WRITE_BUFFER_BATCHES."""
    return int(get('WRITE_BUFFER_BATCHES', 1))

def get_num_scan_workers() -> int:
    """Retorna NUM_SCAN_WORKERS."""
    return int(get('NUM_SCAN_WORKERS', 4))

def get_max_worker_threads() -> int:
    """Retorna MAX_WORKER_THREADS."""
    return int(get('MAX_WORKER_THREADS', 2))

def is_gpu_fallback_enabled() -> bool:
    """Retorna True se fallback GPU→CPU está habilitado."""
    return bool(get('GPU_FALLBACK_ENABLED', True))

def get_num_io_workers() -> int:
    """Retorna NUM_IO_WORKERS."""
    return int(get('NUM_IO_WORKERS', 4))

def get_num_readers() -> int:
    """Retorna NUM_READERS."""
    return int(get('NUM_READERS', 4))

def get_buffer_size() -> int:
    """Retorna BUFFER_SIZE."""
    return int(get('BUFFER_SIZE', 256))

def get_hash_log() -> int:
    """Retorna HASH_LOG."""
    return int(get('HASH_LOG', 20))

def get_hash_candidates() -> int:
    """Retorna HASH_CANDIDATES."""
    return int(get('HASH_CANDIDATES', 7))

def get_good_enough_match() -> int:
    """Retorna GOOD_ENOUGH_MATCH."""
    return int(get('GOOD_ENOUGH_MATCH', 128))

def get_hash_batch_size() -> int:
    """Retorna HASH_BATCH_SIZE - arquivos por batch de hash GPU."""
    return int(get('HASH_BATCH_SIZE', 128))

def get_dedup_gpu_workers() -> int:
    """Retorna DEDUP_GPU_WORKERS - workers paralelos de hash GPU."""
    return int(get('DEDUP_GPU_WORKERS', 2))

def get_pre_scan_target_dirs() -> int:
    """Retorna PRE_SCAN_TARGET_DIRS - número alvo de diretórios para Pre-Scan BFS."""
    return int(get('PRE_SCAN_TARGET_DIRS', 1000))

def get_scan_status_interval() -> float:
    """Retorna SCAN_STATUS_INTERVAL - intervalo de atualização do console durante scan (segundos)."""
    return float(get('SCAN_STATUS_INTERVAL', 5))

