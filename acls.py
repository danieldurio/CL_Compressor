"""
Módulo para backup e restauração de metadata NTFS (ACLs, Atributos, Timestamps).
Executa via ctypes para manipular SDDL e Atributos.
Saída/Entrada comprimida em GZIP (JSONL).
"""

import os
import ctypes
from ctypes import wintypes
import json
import gzip
from pathlib import Path
from datetime import datetime

# Win32 Constants
FILE_ATTRIBUTE_READONLY = 0x01
FILE_ATTRIBUTE_HIDDEN = 0x02
FILE_ATTRIBUTE_SYSTEM = 0x04
FILE_ATTRIBUTE_ARCHIVE = 0x20
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_ATTRIBUTE_TEMPORARY = 0x100
FILE_ATTRIBUTE_COMPRESSED = 0x800
FILE_ATTRIBUTE_ENCRYPTED = 0x4000

# Security Information Constants
OWNER_SECURITY_INFORMATION = 0x00000001
GROUP_SECURITY_INFORMATION = 0x00000002
DACL_SECURITY_INFORMATION = 0x00000004
SACL_SECURITY_INFORMATION = 0x00000008
LABEL_SECURITY_INFORMATION = 0x00000010

def get_file_sddl(path: str) -> str:
    """Retorna a string SDDL do arquivo (Owner, Group, DACL)."""
    advapi32 = ctypes.windll.advapi32
    kernel32 = ctypes.windll.kernel32
    
    # Define argtypes for 64-bit safety
    advapi32.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)
    ]
    
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.ULONG)
    ]
    
    # 1. Get Size needed
    size_needed = wintypes.DWORD(0)
    # Requested Info: Owner, Group, DACL. (SACL requires high privilege/audit)
    security_info = OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
    
    advapi32.GetFileSecurityW(path, security_info, None, 0, ctypes.byref(size_needed))
    
    if size_needed.value == 0:
        return "" # Fail or empty
        
    # 2. Get Descriptor
    sd_buffer = ctypes.create_string_buffer(size_needed.value)
    if not advapi32.GetFileSecurityW(path, security_info, sd_buffer, size_needed, ctypes.byref(size_needed)):
        return ""
        
    # 3. Convert to String SDDL
    sddl_ptr = wintypes.LPWSTR()
    sddl_len = wintypes.ULONG()
    
    if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
        sd_buffer, 
        1, # SDDL_REVISION_1
        security_info, 
        ctypes.byref(sddl_ptr), 
        ctypes.byref(sddl_len)
    ):
        return ""
        
    # Copy from pointer
    try:
        sddl_str = ctypes.wstring_at(sddl_ptr)
    finally:
        kernel32.LocalFree(sddl_ptr)
        
    return sddl_str

def set_file_sddl(path: str, sddl: str) -> bool:
    """Aplica uma string SDDL a um arquivo (Owner, Group, DACL)."""
    if not sddl:
        return False
        
    advapi32 = ctypes.windll.advapi32
    kernel32 = ctypes.windll.kernel32
    
    # Types
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.ULONG)
    ]
    advapi32.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID
    ]
    
    # 1. Convert SDDL String -> Security Descriptor Binary
    sd_ptr = wintypes.LPVOID()
    sd_size = wintypes.ULONG()
    
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl,
        1, # SDDL_REVISION_1
        ctypes.byref(sd_ptr),
        ctypes.byref(sd_size)
    ):
        # error = kernel32.GetLastError()
        # print(f"Erro convertendo SDDL: {error}")
        return False
        
    try:
        # 2. Apply Security Descriptor
        # Try to apply Owner, Group, DACL
        security_info = OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
        
        if not advapi32.SetFileSecurityW(path, security_info, sd_ptr):
             # Se falhar (ex: sem privilégio de Owner), tentar apenas DACL
             security_info = DACL_SECURITY_INFORMATION
             if not advapi32.SetFileSecurityW(path, security_info, sd_ptr):
                 return False
                 
        return True
    finally:
        kernel32.LocalFree(sd_ptr)


def get_win_attributes(path: str) -> int:
    try:
        return ctypes.windll.kernel32.GetFileAttributesW(path)
    except:
        return 0

def set_win_attributes(path: str, attrs: int) -> bool:
    try:
        if attrs == 0: return True
        # Filter out some attributes we shouldn't force (like Compressed/Encrypted/ReparsePoint if not handled)
        # But generally, just trying to set basics: ReadOnly, Hidden, System, Archive
        # Mask: readonly=1, hidden=2, system=4, archive=32, temp=256
        mask = 0x01 | 0x02 | 0x04 | 0x20 
        attrs_to_set = attrs & mask
        
        # If the file is currently ReadOnly, we might need to clear it first to modify other things? 
        # Actually SetFileAttributes overrides.
        
        return ctypes.windll.kernel32.SetFileAttributesW(path, attrs_to_set) != 0
    except:
        return False

def backup_acls(source_root: Path, output_file: Path):
    """
    Varre source_root e salva metadados de TODOS arquivos/pastas.
    """
    source_root = source_root.resolve()
    print(f"\n[ACLS] Iniciando captura de metadados para: {source_root}")
    print(f"[ACLS] Saída: {output_file}")
    
    count = 0
    start_time = datetime.now()
    
    try:
        # Abrir compressão gzip
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            # Escrever Header
            header = {
                "version": 1, 
                "source": str(source_root),
                "start_time": start_time.isoformat()
            }
            f.write(json.dumps(header) + "\n")
            
            # Walk manual para garantir ordem e completude
            for root, dirs, files in os.walk(source_root):
                # Processar diretório atual (root)
                # entry para o próprio diretório
                try:
                    rel_path = str(Path(root).relative_to(source_root)).replace("\\", "/")
                    if rel_path == ".": rel_path = "" # Root
                    
                    sddl = get_file_sddl(root)
                    attrs = get_win_attributes(root)
                    st = os.stat(root)
                    
                    entry = {
                        "p": rel_path,
                        "t": "d", # dir
                        "s": sddl,
                        "a": attrs,
                        "ct": st.st_ctime,
                        "mt": st.st_mtime,
                        "at": st.st_atime
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1
                except Exception as e:
                    pass
                
                # Processar arquivos
                for name in files:
                    try:
                        abs_path = os.path.join(root, name)
                        rel_path = str(Path(abs_path).relative_to(source_root)).replace("\\", "/")
                        
                        sddl = get_file_sddl(abs_path)
                        attrs = get_win_attributes(abs_path)
                        st = os.stat(abs_path)
                        
                        entry = {
                            "p": rel_path,
                            "t": "f", # file
                            "s": sddl,
                            "a": attrs,
                            "ct": st.st_ctime,
                            "mt": st.st_mtime,
                            "at": st.st_atime
                        }
                        f.write(json.dumps(entry) + "\n")
                        count += 1
                    except Exception:
                        pass
                        
    except Exception as e:
        print(f"[ACLS] Erro fatal durante backup de ACLs: {e}")
        return

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n[ACLS] Concluído. {count} itens processados em {duration:.1f}s.")
    print(f"[ACLS] Arquivo salvo em: {output_file}")

def restore_acls(acls_file: Path, target_root: Path):
    """
    Lê arquivo .acls e aplica metadados em target_root.
    Versão Otimizada (Sequencial + String Path operations).
    Evita overhead de Threads e Pathlib para máxima velocidade em items pequenos.
    """
    import os
    
    target_root_path = Path(target_root).resolve()
    target_root_str = str(target_root_path)
    acls_file = Path(acls_file)
    
    if not acls_file.exists():
        print(f"[ACLS] Arquivo de metadados não encontrado: {acls_file}")
        return
        
    print(f"\n[ACLS] Restaurando metadados de: {acls_file}")
    print(f"[ACLS] Alvo: {target_root_str}")
    
    start_time = datetime.now()
    count = 0
    errors = 0
    
    try:
        with gzip.open(acls_file, 'rt', encoding='utf-8') as f:
            # Read Header
            try:
                header_line = f.readline()
                header = json.loads(header_line)
            except:
                print("[ACLS] Erro lendo header (arquivo inválido?)")
                return
            
            print(f"[ACLS] Processando sequencialmente (Otimizado)...")
            
            # Hot Loop Optimization: Pre-lookup methods
            json_loads = json.loads
            path_join = os.path.join
            path_exists = os.path.exists
            utime = os.utime
            
            for line in f:
                try:
                    entry = json_loads(line)
                    
                    # Usa operações de string (muito mais rápido que pathlib)
                    rel_path = entry["p"]
                    # Normalizar separadores se necessário, mas json source já deve estar ok
                    
                    # Construir caminho absoluto
                    # obs: rel_path pode ser vazio para a raiz
                    if rel_path:
                        full_path = path_join(target_root_str, rel_path)
                    else:
                        full_path = target_root_str
                        
                    if not path_exists(full_path):
                        continue
                        
                    # Extrair dados
                    sddl = entry.get("s")
                    attrs = entry.get("a", 0)
                    at = entry.get("at")
                    mt = entry.get("mt")
                    
                    # 1. Timestamps (syscall rápida)
                    if at is not None and mt is not None:
                         # utime pode falhar se arquivo estiver travado
                         try:
                             utime(full_path, (at, mt))
                         except:
                             pass
                             
                    # 2. Attributes
                    if attrs != 0:
                        set_win_attributes(full_path, attrs)
                        
                    # 3. Security (Mais lento, mas necessário)
                    if sddl:
                        set_file_sddl(full_path, sddl)
                        
                    count += 1
                    
                    if count % 5000 == 0:
                         print(f"[ACLS] {count} itens processados...", end='\r')
                         
                except Exception:
                    errors += 1
                    pass

    except Exception as e:
        print(f"[ACLS] Erro fatal durante restore: {e}")
        
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n[ACLS] Restore Concluído. {count} itens processados em {duration:.1f}s (Sequencial Otimizado).")

