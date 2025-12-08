"""
Explorer - Navegador de Arquivos para Archives Compactados (GPU_LZ4)

Suporta índices:
- GPU_IDX1 (Sólido/JSON)
- GPU_IDX2 (Streaming/JSONL)
- GPU_IDX3 (SQLite Embedded)

Uso:
    python explorer.py arquivo.001
"""

from __future__ import annotations
import struct
import zlib
import json
import argparse
import os
import re
import gzip
import io
import time
import shutil
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Set, Union, Optional, Any

def format_bytes(size: int) -> str:
    """Formata bytes para formato legível."""
    if size is None: return "0.0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


# -----------------------------------------------------------------------------
# Backends de Dados (Abstração)
# -----------------------------------------------------------------------------

class FileItem:
    def __init__(self, name: str, size: int, original_entry: Dict):
        self.name = name
        self.size = size
        self.is_folder = False
        self.original_entry = original_entry # Dict completo para extração

class FolderItem:
    def __init__(self, name: str):
        self.name = name
        self.is_folder = True
        self.size = 0 # Pode ser calculado on-demand
        self.file_count = 0

class ArchiveBackend(ABC):
    @abstractmethod
    def list_dir(self, path: str) -> List[Union[FileItem, FolderItem]]:
        """Lista conteúdo de uma pasta (path relativo, sem / inicial)."""
        pass
    
    @abstractmethod
    def get_info(self) -> str:
        """Retorna info do archive."""
        pass
    
    @abstractmethod
    def close(self):
        pass


class InMemoryBackend(ArchiveBackend):
    """
    Backend para GPU_IDX1/2 onde tudo é carregado na RAM.
    Constrói uma árvore de objetos TreeNode uma única vez.
    """
    class TreeNode:
        def __init__(self, name: str, is_folder: bool = True):
            self.name = name
            self.is_folder = is_folder
            self.size = 0
            self.children: Dict[str, InMemoryBackend.TreeNode] = {}
            self.entry: Optional[Dict] = None

    def __init__(self, index_dict: Dict):
        self.index = index_dict
        self.root = self._build_tree(index_dict.get('files', []))
        self.files_count = len(index_dict.get('files', []))

    def _build_tree(self, files: List[Dict]) -> TreeNode:
        root = self.TreeNode("", is_folder=True)
        for f in files:
            if f.get('is_duplicate'): continue
            
            path = f['path_rel'].replace('\\', '/')
            parts = [p for p in path.split('/') if p]
            
            curr = root
            for i, part in enumerate(parts):
                is_last = (i == len(parts) - 1)
                
                if part not in curr.children:
                    node = self.TreeNode(part, is_folder=not is_last)
                    curr.children[part] = node
                
                curr = curr.children[part]
                if is_last:
                    curr.size = f.get('size', 0)
                    curr.entry = f
            
            # Propagar tamanhos para cima (opcional, pesado se muitos arquivos)
            # Simplificação: não propagar neste backend simples para economizar tempo start
            
        return root

    def list_dir(self, path: str) -> List[Union[FileItem, FolderItem]]:
        # Navegar na árvore
        curr = self.root
        
        # Tratar root
        if path == "" or path == ".":
            pass
        else:
            parts = [p for p in path.replace('\\', '/').split('/') if p]
            for part in parts:
                if part in curr.children:
                    curr = curr.children[part]
                else:
                    return [] # Path não encontrado
        
        if not curr.is_folder:
            return []

        items = []
        for name, node in curr.children.items():
            if node.is_folder:
                items.append(FolderItem(name))
            else:
                items.append(FileItem(name, node.size, node.entry))
        
        # Ordenar: Pastas primeiro
        items.sort(key=lambda x: (not x.is_folder, x.name.lower()))
        return items

    def get_info(self) -> str:
        return f"In-Memory (Legacy) | {self.files_count} arquivos"

    def close(self):
        pass


class SQLiteBackend(ArchiveBackend):
    """
    Backend para GPU_IDX3 usando SQLite.
    Queries diretas no banco de dados.
    """
    def __init__(self, db_path: Path, archive_path: Path):
        self.db_path = db_path
        self.archive_path = archive_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Contagem rápida
        self.files_count = self.conn.execute("SELECT count(*) FROM files").fetchone()[0]
        
        # Descobrir volumes disponíveis para extração
        parent_dir = archive_path.parent
        name_stem = archive_path.name
        match = re.match(r"(.*)\.(\d{3})$", name_stem)
        prefix = match.group(1) if match else name_stem
        self.volumes = sorted([v for v in parent_dir.glob(f"{prefix}.*") if re.match(r".*\.\d{3}$", v.name)])
        self.parent_dir = parent_dir
        self.prefix = prefix

    def list_dir(self, path: str) -> List[Union[FileItem, FolderItem]]:
        # Normalizar path para coincidir com parent_path no DB
        # parent_path não tem slash no final, e é vazio para root.
        # Ex: "a/b" para arquivos dentro de "a/b".
        
        target_parent = path.replace('\\', '/').strip('/')
        
        items = []
        c = self.conn.cursor()
        
        # 1. Buscar Arquivos diretos
        # SELECT * FROM files WHERE parent_path = ?
        c.execute("""
            SELECT path_rel, size, is_duplicate, original_path, volume_name, offset, compressed_size 
            FROM files 
            WHERE parent_path = ?
            ORDER BY path_rel
        """, (target_parent,))
        
        for row in c.fetchall():
            name = row['path_rel'].rsplit('/', 1)[-1]
            entry = dict(row) # Converter row para dict
            # Ajustar dict para formato esperado pelo extrator se necessário
            items.append(FileItem(name, row['size'], entry))
            
        # 2. Buscar Subpastas
        # SELECT DISTINCT parent_path FROM files WHERE parent_path LIKE target_parent || '/%'
        # Mas isso pega TODOS os descendentes.
        # Ex: target="a". Resultados: "a/b", "a/b/c", "a/d".
        # Queremos apenas os filhos imediatos: "b", "d".
        
        prefix = target_parent + "/" if target_parent else ""
        query = "SELECT DISTINCT parent_path FROM files WHERE parent_path LIKE ? || '%'"
        
        # Se target_parent for vazio (root), LIKE '%' pega tudo. Otimizar?
        # Sim. Se empty, parent_path LIKE '%'.
        # Mas queremos apenas os que não contêm '/' (se estamos na root)?
        # Não, 'a/b' tem parent 'a'. 'a' tem parent ''.
        # Na root, queremos achar quem tem parent_path = "x" (onde x não tem /)
        # OU "x/y" ? Não.
        # Qualquer arquivo em "x/y.txt" tem parent "x". Então existe a pasta "x".
        # Se estamos na root, queremos listar todas as pastas "x" que aparecem como parent_path.
        # E se tivermos "x/y/z.txt"? parent="x/y". Isso implica pasta "x/y", que implica pasta "x".
        
        # A lógica correta é:
        # Quais são os primeiros componentes de caminho que residem sob 'target_parent'?
        # Ex: target="". Paths: "a.txt", "b/c.txt" (parent="b").
        # "b" é uma pasta na root.
        # O problema é que a tabela files indexa o PAI direto.
        # Se tenho apenas "a/b/c.txt" (parent="a/b"), a pasta "a" NÃO aparece explicitamente como parent de ninguém se não houver arquivos diretos em "a".
        # Isso significa que pastas vazias de arquivos (mas contendo pastas) não apareceriam se olharmos só 'parent_path = target'.
        
        # Solução Robusta com o que temos:
        # Pegar todos os parent_path que começam com o prefixo.
        # Para cada parent_path único encontrado, extrair o próximo segmento após o prefixo.
        # Adicionar esse segmento como pasta.
        # Usar Python set para uniquificar.
        
        # Isso pode trazer muitos resultados?
        # SELECT DISTINCT parent_path é razoável.
        
        if target_parent:
            sql_pattern = target_parent + "/%"
            c.execute("SELECT DISTINCT parent_path FROM files WHERE parent_path LIKE ?", (sql_pattern,))
        else:
            # Root: pegar tudo que não seja vazio
            c.execute("SELECT DISTINCT parent_path FROM files WHERE parent_path <> ''")
            
        found_folders = set()
        prefix_len = len(prefix)
        
        for row in c.fetchall():
            p_path = row['parent_path']
            # Remover prefixo
            if p_path.startswith(prefix):
                suffix = p_path[prefix_len:]
                # Pegar primeiro componente
                first_comp = suffix.split('/')[0]
                if first_comp:
                    found_folders.add(first_comp)
        
        for f_name in sorted(found_folders):
            items.append(FolderItem(f_name))
        
        # Ordenar final (folders first)
        items.sort(key=lambda x: (not x.is_folder, x.name.lower()))
        
        return items

    def search(self, pattern: str, limit: int = 100) -> List[Dict]:
        """
        Busca arquivos por padrão SQL LIKE.
        Retorna lista de dicts com informações dos arquivos encontrados.
        
        Args:
            pattern: Padrão de busca (ex: "%config%" ou "%.txt")
            limit: Máximo de resultados
            
        Returns:
            Lista de dicionários com path_rel, size, is_duplicate, etc.
        """
        c = self.conn.cursor()
        
        # Buscar com LIKE (case-insensitive no SQLite por padrão para ASCII)
        c.execute("""
            SELECT path_rel, size, is_duplicate, original_path, volume_name, offset, compressed_size
            FROM files
            WHERE path_rel LIKE ?
            ORDER BY path_rel
            LIMIT ?
        """, (pattern, limit))
        
        results = []
        for row in c.fetchall():
            results.append(dict(row))
        
        return results

    def get_file_entry(self, path_rel: str) -> Optional[Dict]:
        """
        Obtém entrada de arquivo pelo caminho relativo.
        
        Args:
            path_rel: Caminho relativo do arquivo (normalizado com /)
            
        Returns:
            Dict com informações do arquivo ou None se não encontrado
        """
        c = self.conn.cursor()
        
        # Normalizar path
        path_rel = path_rel.replace('\\', '/')
        
        c.execute("""
            SELECT path_rel, size, is_duplicate, original_path, volume_name, offset, compressed_size
            FROM files
            WHERE path_rel = ?
        """, (path_rel,))
        
        row = c.fetchone()
        return dict(row) if row else None

    def get_all_files_in_folder(self, folder_path: str) -> List[Dict]:
        """
        Obtém todos os arquivos dentro de uma pasta (recursivo).
        
        Args:
            folder_path: Caminho da pasta (normalizado com /)
            
        Returns:
            Lista de dicts com informações dos arquivos
        """
        c = self.conn.cursor()
        
        # Normalizar path
        folder_path = folder_path.replace('\\', '/').strip('/')
        
        if folder_path:
            pattern = folder_path + "/%"
            c.execute("""
                SELECT path_rel, size, is_duplicate, original_path, volume_name, offset, compressed_size
                FROM files
                WHERE path_rel LIKE ? OR parent_path = ?
                ORDER BY path_rel
            """, (pattern, folder_path))
        else:
            # Root: pegar tudo
            c.execute("""
                SELECT path_rel, size, is_duplicate, original_path, volume_name, offset, compressed_size
                FROM files
                ORDER BY path_rel
            """)
        
        results = []
        for row in c.fetchall():
            results.append(dict(row))
        
        return results

    def get_frames(self) -> Dict[int, Dict]:
        """
        Obtém todas as informações de frames do arquivo.
        
        Returns:
            Dict mapeando frame_id para informações do frame
        """
        c = self.conn.cursor()
        
        c.execute("SELECT id, vol, offset, size, original_size FROM frames ORDER BY id")
        
        frames = {}
        for row in c.fetchall():
            frames[row[0]] = {
                "frame_id": row[0],
                "volume_name": row[1],
                "offset": row[2],
                "compressed_size": row[3],
                "uncompressed_size": row[4]
            }
        
        return frames

    def get_file_details(self, path_rel: str) -> Optional[Dict]:
        """
        Obtém informações detalhadas de um arquivo incluindo:
        - Frame onde está localizado
        - Volume onde está
        - Se tem duplicatas (outros arquivos apontando para ele)
        - Informações de ACLs se disponíveis
        
        Args:
            path_rel: Caminho relativo do arquivo
            
        Returns:
            Dict com informações detalhadas ou None
        """
        c = self.conn.cursor()
        path_rel = path_rel.replace('\\', '/')
        
        # 1. Buscar informações básicas + ACLs
        c.execute("""
            SELECT f.path_rel, f.size, f.is_duplicate, f.original_path,
                   f.attrs, f.ctime, f.mtime, f.atime, s.sddl
            FROM files f
            LEFT JOIN sddls s ON f.sddl_id = s.id
            WHERE f.path_rel = ?
        """, (path_rel,))
        
        row = c.fetchone()
        if not row:
            return None
        
        details = {
            "path_rel": row[0],
            "size": row[1] if row[1] else 0,
            "is_duplicate": bool(row[2]),
            "original_path": row[3],
            "attrs": row[4],
            "ctime": row[5],
            "mtime": row[6],
            "atime": row[7],
            "sddl": row[8],
            "frame_id": None,
            "volume_name": None,
            "has_duplicates": False,
            "duplicate_count": 0
        }
        
        # 2. Verificar se tem duplicatas (outros arquivos apontando para este)
        c.execute("""
            SELECT COUNT(*) FROM files 
            WHERE original_path = ? AND is_duplicate = 1
        """, (path_rel,))
        dup_count = c.fetchone()[0]
        details["has_duplicates"] = dup_count > 0
        details["duplicate_count"] = dup_count
        
        # 3. Calcular Frame e Volume (apenas para arquivos não-duplicados)
        if not details["is_duplicate"]:
            # Precisamos calcular a posição global e encontrar o frame
            c.execute("""
                SELECT id, path_rel, size, is_duplicate FROM files ORDER BY id
            """)
            
            global_offset = 0
            file_start = None
            
            for f_row in c.fetchall():
                f_path = f_row[1]
                f_size = f_row[2] if f_row[2] else 0
                f_is_dup = bool(f_row[3])
                
                if f_is_dup:
                    continue
                
                if f_path == path_rel:
                    file_start = global_offset
                    break
                
                global_offset += f_size
            
            if file_start is not None:
                # Encontrar o frame que contém este offset
                frames = self.get_frames()
                frame_offset = 0
                
                for frame_id in sorted(frames.keys()):
                    fr = frames[frame_id]
                    fr_end = frame_offset + fr['uncompressed_size']
                    
                    if file_start < fr_end:
                        details["frame_id"] = frame_id
                        details["volume_name"] = fr['volume_name']
                        break
                    
                    frame_offset = fr_end
        
        return details

    def get_info(self) -> str:
        return f"SQLite (GPU_IDX3) | {self.files_count} arquivos"

    def close(self):
        self.conn.close()
        # Não deletar o arquivo DB aqui, pois o explorer pode ser reaberto?
        # explorer roda uma vez e sai. Deletar no exit do programa.
        pass

# -----------------------------------------------------------------------------
# Lógica de Leitura do Índice
# -----------------------------------------------------------------------------

def load_backend(archive_path: Path) -> Optional[ArchiveBackend]:
    """Descobre formato e retorna Backend apropriado."""
    parent_dir = archive_path.parent
    name_stem = archive_path.name
    
    # Resolver prefixo (ex: data.001 -> data)
    match = re.match(r"(.*)\.(\d{3})$", name_stem)
    prefix = match.group(1) if match else name_stem
    
    volumes = sorted([v for v in parent_dir.glob(f"{prefix}.*") if re.match(r".*\.\d{3}$", v.name)])
    if not volumes:
        print("Nenhum volume encontrado.")
        return None
        
    last_vol = volumes[-1]
    
    try:
        with open(last_vol, 'rb') as f:
            f.seek(0, 2)
            if f.tell() < 24: return None
            f.seek(-24, 2)
            offset, size, magic = struct.unpack('<QQ8s', f.read(24))
            
            if magic == b'GPU_IDX1':
                print("Carregando índice GPU_IDX1 (Sólido)...")
                f.seek(offset)
                data = zlib.decompress(f.read(size))
                return InMemoryBackend(json.loads(data.decode('utf-8')))
                
            elif magic == b'GPU_IDX2':
                print("Carregando índice GPU_IDX2 (Streaming)...")
                # Carregar para Memória (Fallback)
                f.seek(offset)
                compressed = f.read(size)
                with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as gz:
                    line = gz.readline()
                    header = json.loads(line)
                    files = []
                    for _ in range(header.get('count_files', 0)):
                        files.append(json.loads(gz.readline()))
                return InMemoryBackend({'files': files})
                
            elif magic == b'GPU_IDX3':
                print("Carregando índice GPU_IDX3 (SQLite)...")
                tmp_dir = Path("tmp").resolve()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                db_path = tmp_dir / f"explorer_{int(time.time())}.db"
                
                f.seek(offset)
                compressed = f.read(size)
                with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as gz:
                    with open(db_path, "wb") as db_out:
                        shutil.copyfileobj(gz, db_out)
                        
                return SQLiteBackend(db_path, archive_path)
            
            else:
                print(f"Magic desconhecido: {magic}")
                return None
                
    except Exception as e:
        print(f"Erro ao abrir índice: {e}")
        return None


# -----------------------------------------------------------------------------
# Interface do Terminal
# -----------------------------------------------------------------------------

class TerminalExplorer:
    def __init__(self, backend: ArchiveBackend):
        self.backend = backend
        self.current_path = "" # relative path from root, no leading/trailing slashes
        self.selected_paths: Set[str] = set()
        self.history: List[str] = [] # stack of paths
        self.search_results = None  # Lista de resultados de busca (ou None)
        
    def run(self):
        while True:
            self.clear_screen()
            self.print_header()
            
            # Mostrar resultados de busca ou listagem normal
            if self.search_results is not None:
                items = self.print_search_results()
            else:
                items = self.backend.list_dir(self.current_path)
                self.print_contents(items)
            
            self.print_commands()
            
            cmd = input("Comando > ").strip()
            
            if cmd == 'q':
                break
            elif cmd == '':
                if self.search_results is not None:
                    self.search_results = None  # Voltar da busca
                else:
                    self.go_up()
            elif cmd.isdigit():
                idx = int(cmd)
                if self.search_results is not None:
                    # Navegar até o arquivo dos resultados de busca
                    if 1 <= idx <= len(self.search_results):
                        result = self.search_results[idx-1]
                        path = result['path_rel'].replace('\\', '/')
                        # Navegar até a pasta pai
                        if '/' in path:
                            parent = path.rsplit('/', 1)[0]
                            self.current_path = parent
                            self.history = [p for p in parent.split('/') if p]
                        else:
                            self.current_path = ""
                            self.history = []
                        self.search_results = None
                else:
                    if 1 <= idx <= len(items):
                        item = items[idx-1]
                        if item.is_folder:
                            self.enter_folder(item.name)
                        else:
                            self.show_file_info(item)
            elif cmd.startswith('s '):
                try:
                    idx = int(cmd.split()[1])
                    if self.search_results is not None:
                        if 1 <= idx <= len(self.search_results):
                            result = self.search_results[idx-1]
                            path = result['path_rel']
                            if path in self.selected_paths:
                                self.selected_paths.remove(path)
                            else:
                                self.selected_paths.add(path)
                    else:
                        if 1 <= idx <= len(items):
                            item = items[idx-1]
                            self.toggle_selection(item)
                except: pass
            elif cmd == 'sa':
                if self.search_results is not None:
                    for r in self.search_results:
                        self.selected_paths.add(r['path_rel'])
                else:
                    for it in items:
                        self.select_item(it)
            elif cmd == 'da':
                self.selected_paths.clear()
            elif cmd == 'x':
                self.extract_selected()
            elif cmd.startswith('/'):
                # Busca
                pattern = cmd[1:].strip()
                if pattern:
                    self.do_search(pattern)
                else:
                    print("Uso: /<padrão>  (ex: /%config% ou /%.txt)")
                    input("Enter...")
                    
    def do_search(self, pattern: str):
        """Realiza busca por padrão."""
        # Adicionar % automaticamente se não tiver
        if '%' not in pattern:
            pattern = f"%{pattern}%"
        
        if not isinstance(self.backend, SQLiteBackend):
            print("Busca só disponível para índices SQLite (GPU_IDX3)")
            input("Enter...")
            return
        
        results = self.backend.search(pattern, limit=200)
        
        if not results:
            print(f"Nenhum resultado para: {pattern}")
            input("Enter...")
        else:
            self.search_results = results
                    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        print("=" * 70)
        print(f"📦 EXPLORER ({self.backend.get_info()})")
        print("=" * 70)
        if self.search_results is not None:
            print(f"🔍 Resultados da busca: {len(self.search_results)} encontrados")
        else:
            print(f"📁 Caminho: /{self.current_path}")
        print(f"📊 Selecionados: {len(self.selected_paths)} itens")
        print("-" * 70)
        
    def print_contents(self, items: List[Union[FileItem, FolderItem]]):
        if not items:
            print("  (pasta vazia)")
            return
            
        print(f"{'#':>3}  {'Sel':>3}  {'Tipo':>4}  {'Tamanho':>10}  Nome")
        print("-" * 70)
        
        for i, item in enumerate(items, 1):
            # Check selection
            full_path = (self.current_path + "/" + item.name).strip("/")
            is_sel = full_path in self.selected_paths
            
            sel_mark = "[X]" if is_sel else "[ ]"
            type_mark = "📁" if item.is_folder else "📄"
            size_str = format_bytes(item.size)
            
            name = item.name
            if len(name) > 40: name = name[:37] + "..."
            
            print(f"{i:>3}  {sel_mark}  {type_mark}  {size_str:>10}  {name}")
            
    def print_search_results(self) -> List[Dict]:
        """Imprime resultados de busca e retorna a lista."""
        print(f"{'#':>3}  {'Sel':>3}  {'Tamanho':>10}  Caminho")
        print("-" * 70)
        
        for i, result in enumerate(self.search_results[:50], 1):  # Mostrar máximo 50
            path = result['path_rel']
            size = result.get('size', 0)
            is_sel = path in self.selected_paths
            
            sel_mark = "[X]" if is_sel else "[ ]"
            size_str = format_bytes(size)
            
            # Truncar caminho se muito longo
            display_path = path
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            
            print(f"{i:>3}  {sel_mark}  {size_str:>10}  {display_path}")
        
        if len(self.search_results) > 50:
            print(f"  ... e mais {len(self.search_results) - 50} resultados")
        
        return self.search_results
            
    def print_commands(self):
        print("-" * 70)
        print(" [N] Entrar/Navegar  |  [Enter] Voltar  |  s [N] Sel/Desel")
        print(" sa/da Sel/Desel Tudo  |  x Extrair  |  /<termo> Buscar  |  q Sair")
            
    def enter_folder(self, name: str):
        self.history.append(self.current_path)
        if self.current_path:
            self.current_path += "/" + name
        else:
            self.current_path = name
            
    def go_up(self):
        if self.history:
            self.current_path = self.history.pop()
        else:
            self.current_path = ""
            
    def toggle_selection(self, item):
        full_path = (self.current_path + "/" + item.name).strip("/")
        if full_path in self.selected_paths:
            self.selected_paths.remove(full_path)
        else:
            self.selected_paths.add(full_path)

    def select_item(self, item):
        full_path = (self.current_path + "/" + item.name).strip("/")
        self.selected_paths.add(full_path)

    def show_file_info(self, item: FileItem):
        """Mostra informações detalhadas do arquivo."""
        full_path = (self.current_path + "/" + item.name).strip("/")
        
        print("\n" + "=" * 60)
        print(f"📄 {item.name}")
        print("=" * 60)
        
        # Obter detalhes completos se backend suportar
        if isinstance(self.backend, SQLiteBackend):
            details = self.backend.get_file_details(full_path)
            
            if details:
                print(f"Tamanho: {format_bytes(details['size'])}")
                
                # Status de duplicata
                if details['is_duplicate']:
                    print(f"Tipo: É DUPLICATA de '{details.get('original_path', '?')}'")
                else:
                    # Mostrar Frame e Volume
                    frame_str = str(details['frame_id']) if details['frame_id'] is not None else "N/A"
                    vol_str = details['volume_name'] if details['volume_name'] else "N/A"
                    print(f"Frame: {frame_str}")
                    print(f"Volume: {vol_str}")
                
                # Verificar se tem duplicatas
                if details['has_duplicates']:
                    print(f"DUP: Sim ({details['duplicate_count']} cópias)")
                else:
                    print("DUP: Não")
                
                # Informações de ACLs
                if details['sddl']:
                    # Mostrar apenas um resumo do SDDL
                    sddl_preview = details['sddl'][:50] + "..." if len(details['sddl']) > 50 else details['sddl']
                    print(f"ACLS: {sddl_preview}")
                else:
                    print("ACLS: (não disponível)")
                
                # Atributos Windows se disponíveis
                if details['attrs']:
                    attrs_hex = f"0x{details['attrs']:08X}"
                    print(f"Attrs: {attrs_hex}")
                
                # Timestamps
                if details['mtime']:
                    import datetime
                    mtime_dt = datetime.datetime.fromtimestamp(details['mtime'])
                    print(f"Modificado: {mtime_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"Tamanho: {format_bytes(item.size)}")
                print("(Detalhes não disponíveis)")
        else:
            print(f"Tamanho: {format_bytes(item.size)}")
            if item.original_entry:
                entry = item.original_entry
                if entry.get('is_duplicate'):
                    print(f"Tipo: Duplicata de '{entry.get('original_path', '?')}'")
        
        print("=" * 60)
        input("Pressione Enter para continuar...")

    def extract_selected(self):
        """Extrai os itens selecionados."""
        if not self.selected_paths:
            print("Nenhum item selecionado para extração.")
            input("Enter...")
            return
        
        # Solicitar pasta de destino
        print("\n" + "=" * 50)
        print("EXTRAÇÃO DE ARQUIVOS")
        print("=" * 50)
        print(f"Itens selecionados: {len(self.selected_paths)}")
        
        output_path = input("Pasta de destino (Enter=./extracted): ").strip()
        if not output_path:
            output_path = "./extracted"
        
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar se é backend SQLite
        if not isinstance(self.backend, SQLiteBackend):
            print("Extração suportada apenas para índices SQLite (GPU_IDX3)")
            input("Enter...")
            return
        
        # Coletar todos os arquivos a extrair
        files_to_extract = []
        
        for sel_path in self.selected_paths:
            # Verificar se é arquivo ou pasta
            entry = self.backend.get_file_entry(sel_path)
            if entry:
                # É um arquivo
                files_to_extract.append(entry)
            else:
                # É uma pasta - pegar todos os arquivos dentro
                folder_files = self.backend.get_all_files_in_folder(sel_path)
                files_to_extract.extend(folder_files)
        
        if not files_to_extract:
            print("Nenhum arquivo encontrado para extrair.")
            input("Enter...")
            return
        
        print(f"\nExtraindo {len(files_to_extract)} arquivos para {output_dir}...")
        
        # Chamar extrator
        try:
            self.do_extraction(files_to_extract, output_dir)
            print(f"\n✅ Extração concluída! {len(files_to_extract)} arquivos extraídos.")
        except Exception as e:
            print(f"\n❌ Erro durante extração: {e}")
        
        input("Enter...")

    def do_extraction(self, files_to_extract: List[Dict], output_dir: Path):
        """
        Executa a extração seletiva dos arquivos.
        
        Algoritmo:
        1. Carrega todos os arquivos do índice para calcular posições nos frames
        2. Identifica quais frames contêm os arquivos desejados
        3. Descomprime apenas esses frames
        4. Extrai apenas os bytes correspondentes
        
        Args:
            files_to_extract: Lista de dicts com informações dos arquivos
            output_dir: Pasta de destino
        """
        from decompressor_lz4_ext3 import decompress_lz4_ext3
        
        backend = self.backend
        
        # Carregar informações de frames
        frames_list = list(backend.get_frames().values())
        frames_list.sort(key=lambda x: x['frame_id'])
        
        if not frames_list:
            print("Nenhum frame encontrado no arquivo.")
            return
        
        # Carregar TODOS os arquivos do índice para calcular posições
        c = backend.conn.cursor()
        c.execute("""
            SELECT path_rel, size, is_duplicate, original_path
            FROM files
            ORDER BY id
        """)
        
        all_files = []
        for r in c.fetchall():
            all_files.append({
                "path_rel": r[0],
                "size": r[1] if r[1] is not None else 0,
                "is_duplicate": bool(r[2]),
                "original": r[3]
            })
        
        # Criar set de caminhos a extrair para lookup rápido
        paths_to_extract = set(f['path_rel'].replace('\\', '/') for f in files_to_extract)
        
        
        # Calcular posição de cada arquivo nos frames
        # e identificar quais frames precisamos
        file_positions = []  # Lista de (file_info, start_offset, end_offset)
        global_offset = 0
        
        duplicates = []  # Duplicatas para processar depois
        
        for f in all_files:
            path_normalized = f['path_rel'].replace('\\', '/')
            size = f['size']
            
            if f['is_duplicate']:
                # Duplicatas não ocupam espaço nos frames
                if path_normalized in paths_to_extract:
                    duplicates.append(f)
                continue
            
            if path_normalized in paths_to_extract:
                file_positions.append({
                    "path_rel": f['path_rel'],
                    "size": size,
                    "start_offset": global_offset,
                    "end_offset": global_offset + size
                })
            
            global_offset += size
        
        if not file_positions and not duplicates:
            print("Nenhum arquivo encontrado para extrair.")
            return
        
        # Calcular quais frames são necessários
        frame_size = frames_list[0]['uncompressed_size'] if frames_list else (16 * 1024 * 1024)
        
        # Frame boundaries: cada frame tem uncompressed_size bytes
        # Precisamos calcular quais frames cobrem cada arquivo
        needed_frames = set()
        frame_cumulative_offset = 0
        frame_ranges = []  # (frame_id, start_global_offset, end_global_offset)
        
        for fr in frames_list:
            fr_start = frame_cumulative_offset
            fr_end = frame_cumulative_offset + fr['uncompressed_size']
            frame_ranges.append((fr['frame_id'], fr_start, fr_end))
            frame_cumulative_offset = fr_end
        
        # Identificar frames necessários
        for fp in file_positions:
            file_start = fp['start_offset']
            file_end = fp['end_offset']
            
            for frame_id, fr_start, fr_end in frame_ranges:
                # Frame intersecta com o arquivo?
                if fr_end > file_start and fr_start < file_end:
                    needed_frames.add(frame_id)
        
        print(f"Precisando descomprimir {len(needed_frames)} de {len(frames_list)} frames...")
        
        # Descomprimir frames necessários
        vol_handles = {}
        frame_data_cache = {}  # frame_id -> decompressed bytes
        
        try:
            for frame_id in sorted(needed_frames):
                fr = next(f for f in frames_list if f['frame_id'] == frame_id)
                vol_name = fr['volume_name']
                offset = fr['offset']
                c_size = fr['compressed_size']
                u_size = fr['uncompressed_size']
                
                # Abrir volume
                vol_path = backend.parent_dir / vol_name
                if vol_name not in vol_handles:
                    vol_handles[vol_name] = open(vol_path, 'rb')
                
                f = vol_handles[vol_name]
                f.seek(offset)
                compressed_data = f.read(c_size)
                
                # Descomprimir
                try:
                    decompressed = decompress_lz4_ext3(compressed_data, u_size)
                    frame_data_cache[frame_id] = decompressed
                except Exception as e:
                    # Fallback: tentar como RAW
                    if len(compressed_data) == u_size:
                        frame_data_cache[frame_id] = compressed_data
                    else:
                        print(f"[AVISO] Erro ao descomprimir frame {frame_id}: {e}")
                        frame_data_cache[frame_id] = b'\x00' * u_size
                
                print(f"  Frame {frame_id}/{max(needed_frames)}: descomprimido", end='\r')
            
            print()
            
            # Extrair arquivos
            extracted_count = 0
            
            for fp in file_positions:
                path_rel = fp['path_rel'].replace('/', os.sep)
                
                # CORREÇÃO: Remover prefixo de drive/anchor se presente
                # Ex: "C:\HCFMRP\arquivo.ico" -> "HCFMRP\arquivo.ico"
                path_obj = Path(path_rel)
                if path_obj.is_absolute():
                    # Remover prefixo (ex: C:\ ou /)
                    path_rel = str(path_obj.relative_to(path_obj.anchor))
                
                dest_path = output_dir / path_rel
                
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_start = fp['start_offset']
                file_size = fp['size']
                file_end = file_start + file_size
                
                # Montar bytes do arquivo a partir dos frames
                file_bytes = bytearray()
                remaining = file_size
                current_offset = file_start
                
                for frame_id, fr_start, fr_end in frame_ranges:
                    if frame_id not in frame_data_cache:
                        continue
                    
                    # Intersecção?
                    if fr_end <= current_offset:
                        continue
                    if fr_start >= file_end:
                        break
                    
                    frame_data = frame_data_cache[frame_id]
                    
                    # Calcular slice dentro do frame
                    slice_start = max(0, current_offset - fr_start)
                    slice_end = min(len(frame_data), file_end - fr_start)
                    
                    chunk = frame_data[slice_start:slice_end]
                    file_bytes.extend(chunk)
                    current_offset += len(chunk)
                    remaining -= len(chunk)
                    
                    if remaining <= 0:
                        break
                
                # Escrever arquivo
                with open(dest_path, 'wb') as out:
                    out.write(bytes(file_bytes[:file_size]))
                
                extracted_count += 1
                if extracted_count % 50 == 0:
                    print(f"  Extraídos: {extracted_count}/{len(file_positions)} arquivos...", end='\r')
            
            print(f"\n  Arquivos normais: {extracted_count} extraídos")
            
            # Processar duplicatas
            if duplicates:
                print(f"  Processando {len(duplicates)} duplicatas...")
                
                # Detectar prefixo raiz
                root_prefix = None
                if file_positions:
                    first_path = file_positions[0]['path_rel'].replace('\\', '/').split('/')
                    if len(first_path) > 1:
                        root_prefix = first_path[0]
                
                dup_ok = 0
                for dup in duplicates:
                    dup_rel = dup['path_rel'].replace('/', os.sep)
                    orig_rel = dup.get('original', '').replace('/', os.sep)
                    
                    if not orig_rel:
                        continue
                    
                    # CORREÇÃO: Remover prefixo de drive/anchor se presente
                    dup_path_obj = Path(dup_rel)
                    if dup_path_obj.is_absolute():
                        dup_rel = str(dup_path_obj.relative_to(dup_path_obj.anchor))
                    
                    orig_path_obj = Path(orig_rel)
                    if orig_path_obj.is_absolute():
                        orig_rel = str(orig_path_obj.relative_to(orig_path_obj.anchor))
                    
                    dest_path = output_dir / dup_rel
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Tentar encontrar original
                    candidates = [
                        output_dir / orig_rel,
                    ]
                    if root_prefix:
                        candidates.insert(0, output_dir / root_prefix / orig_rel)
                    
                    for src_path in candidates:
                        if src_path.exists():
                            shutil.copy2(src_path, dest_path)
                            dup_ok += 1
                            break
                
                print(f"  Duplicatas: {dup_ok}/{len(duplicates)} recriadas")
                extracted_count += dup_ok
            
        finally:
            # Fechar handles
            for f in vol_handles.values():
                f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Arquivo do archive (.001)")
    args = parser.parse_args()
    
    if not args.archive.exists():
        print("Arquivo não encontrado.")
        exit(1)
        
    backend = load_backend(args.archive)
    if backend:
        try:
            app = TerminalExplorer(backend)
            app.run()
        finally:
            backend.close()
            # Cleanup temp db if sqlite
            if isinstance(backend, SQLiteBackend):
                try:
                    if backend.db_path.exists():
                        backend.db_path.unlink()
                except: pass
