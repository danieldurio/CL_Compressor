"""
Módulo para gerenciamento de Volume Shadow Copy (VSS) no Windows.

Estratégia:
- Wrapper de Batch (.bat).
- Caminho ABSOLUTO para WMIC.
- Fix de Quoting: Remove backslash final do Volume para evitar escape de aspas.

Funcionalidades:
- Exclusivo para Windows.
- Requer privilégios de Administrador.
"""

import subprocess
import sys
import os
import re
import tempfile
from contextlib import contextmanager
import ctypes
import shutil

class VSSException(Exception):
    pass

@contextmanager
def disable_fs_redirection():
    if sys.platform != 'win32':
        yield
        return
    is_64bit = sys.maxsize > 2**32
    if is_64bit:
        yield
        return
    wow64 = ctypes.windll.kernel32
    old_value = ctypes.c_void_p()
    if not hasattr(wow64, 'Wow64DisableWow64FsRedirection'):
        yield
        return
    success = False
    try:
        if wow64.Wow64DisableWow64FsRedirection(ctypes.byref(old_value)):
            success = True
        yield
    finally:
        if success:
            wow64.Wow64RevertWow64FsRedirection(old_value)

def _is_admin():
    try:
        return os.getuid() == 0
    except AttributeError:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

def _get_volume_path(path: str) -> str:
    # Retorna "E:\"
    return os.path.splitdrive(os.path.abspath(path))[0] + "\\"

def _find_wmic():
    """Retorna caminho absoluto do WMIC."""
    system_root = os.environ.get('SystemRoot', 'C:\\Windows')
    candidates = [
        os.path.join(system_root, 'System32', 'wbem', 'WMIC.exe'),
        os.path.join(system_root, 'System32', 'WMIC.exe'),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    which = shutil.which("wmic")
    if which: return which
    return None

def _run_bat(commands):
    """Escreve comandos em um bat e executa."""
    fd, bat_path = tempfile.mkstemp(suffix=".bat", text=True)
    os.close(fd)
    
    full_content = "@echo off\r\nchcp 65001 > nul\r\n" + commands
    
    try:
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(full_content)
            
        cmd = ["cmd.exe", "/c", bat_path]
        
        with disable_fs_redirection():
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                encoding='utf-8',
                errors='replace'
            )
            
        output = process.stdout
        error = process.stderr
        return output, error, process.returncode
    finally:
        if os.path.exists(bat_path):
            try: os.remove(bat_path)
            except: pass

@contextmanager
def vss_shadow(source_path: str):
    volume = _get_volume_path(source_path)
    # volume é "E:\"
    
    # Para o WMIC comando create, se usarmos Volume="E:\"", a barra escapa a aspa.
    # Vamos remover a barra final para o comando create.
    # WMIC aceita "E:" normalmente.
    volume_arg = volume.rstrip("\\") # "E:"
    
    wmic_exe = _find_wmic()
    if not wmic_exe:
        raise VSSException("WMIC não encontrado em System32\\wbem nem no PATH.")
        
    print(f"[VSS] Criando shadow para '{volume}' (Batch + Safe Quote)...")
    
    # Try with explicit namespace
    # wmic ... create Volume="E:" ...
    create_cmd = f'"{wmic_exe}" /namespace:\\\\root\\cimv2 shadowcopy call create Volume="{volume_arg}" Context="ClientAccessible"'
    
    output, error, code = _run_bat(create_cmd)
    
    if "Method execution successful" not in output:
         # Tentar sem namespace
         create_cmd_2 = f'"{wmic_exe}" shadowcopy call create Volume="{volume_arg}" Context="ClientAccessible"'
         out2, err2, code2 = _run_bat(create_cmd_2)
         if "Method execution successful" in out2:
             output = out2
         else:
             combined = f"CMD1:\n{output}\nERR1:{error}\nCMD2:\n{out2}\nERR2:{err2}"
             raise VSSException(f"Falha ao executar WMIC (Create).\n{combined}")

    # 2. PARSE ID
    id_match = re.search(r'ShadowID\s*=\s*"(\{[0-9a-fA-F\-]+\})";', output)
    if not id_match:
        raise VSSException(f"ShadowID não encontrado na saída.\n{output}")
    
    shadow_id = id_match.group(1)
    print(f"[VSS] Shadow {shadow_id} criado.")

    # 3. GET DEVICE OBJECT
    get_cmd = f'"{wmic_exe}" /namespace:\\\\root\\cimv2 path Win32_ShadowCopy where "ID=\'{shadow_id}\'" get DeviceObject /value'
    out_get, _, _ = _run_bat(get_cmd)
    
    dev_match = re.search(r'DeviceObject=(.*)', out_get)
    if not dev_match:
         # Fallback
         get_cmd_2 = f'"{wmic_exe}" shadowcopy where "ID=\'{shadow_id}\'" get DeviceObject /value'
         out_get_2, _, _ = _run_bat(get_cmd_2)
         dev_match = re.search(r'DeviceObject=(.*)', out_get_2)
         
    if not dev_match:
        _run_bat(f'"{wmic_exe}" shadowcopy where "ID=\'{shadow_id}\'" delete')
        raise VSSException(f"DeviceObject falhou.\n{out_get}")
        
    shadow_device_path = dev_match.group(1).strip()
    print(f"[VSS] Mapeado em {shadow_device_path}")
    
    try:
        abs_source = os.path.abspath(source_path)
        _, tail = os.path.splitdrive(abs_source)
        if not tail.startswith("\\"): tail = "\\" + tail
        
        vss_path = f"{shadow_device_path}{tail}"
        yield vss_path

    finally:
        print(f"[VSS] Removendo shadow {shadow_id}...")
        del_cmd = f'"{wmic_exe}" shadowcopy where "ID=\'{shadow_id}\'" delete'
        _run_bat(del_cmd)


if __name__ == '__main__':
    if _is_admin():
        print("Teste VSS (Safe Quote)...")
        test_path = os.path.dirname(os.path.abspath(__file__))
        try:
            with vss_shadow(test_path) as sp:
                print(f"Sucesso: {sp}")
        except Exception as e:
            print(f"ERRO: {e}")
    else:
        print("Requer Admin.")
