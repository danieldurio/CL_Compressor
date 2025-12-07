"""
Explorer - Navegador de Arquivos para Archives Compactados

Permite navegar pela estrutura de arquivos/pastas de um archive compactado
no terminal e extrair seletivamente arquivos ou pastas específicas.

Uso:
    python explorer.py arquivo.001
    
Comandos:
    - Números: Navegar para item
    - Enter sem número: Voltar para pasta pai
    - 's N': Selecionar/deselecionar item N
    - 'sa': Selecionar todos na pasta atual
    - 'da': Deselecionar todos
    - 'x': Extrair selecionados
    - 'q': Sair
"""

from __future__ import annotations
import struct
import zlib
import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple


def format_bytes(size: int) -> str:
    """Formata bytes para formato legível."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def read_archive_index(archive_path: Path) -> Optional[Dict[str, Any]]:
    """
    Lê o índice do arquivo compactado a partir do footer.
    
    Args:
        archive_path: Caminho para qualquer volume do archive (.001, .002, etc)
    
    Returns:
        Dicionário com o índice ou None se falhar
    """
    # Encontrar o último volume
    parent_dir = archive_path.parent
    name_stem = archive_path.name
    
    match = re.match(r"(.*)\.(\d{3})$", name_stem)
    if match:
        prefix = match.group(1)
    else:
        prefix = name_stem
    
    # Listar volumes
    volumes_files = sorted(parent_dir.glob(f"{prefix}.*"))
    volumes_files = [v for v in volumes_files if re.match(r".*\.\d{3}$", v.name)]
    
    if not volumes_files:
        print(f"Erro: Nenhum volume encontrado com prefixo '{prefix}'")
        return None
    
    last_vol = volumes_files[-1]
    
    try:
        with open(last_vol, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            
            if file_size < 24:
                print("Erro: Arquivo muito pequeno para conter footer")
                return None
            
            f.seek(-24, 2)
            footer = f.read(24)
            
            offset, size, magic = struct.unpack('<QQ8s', footer)
            
            offset, size, magic = struct.unpack('<QQ8s', footer)
            
            if magic == b'GPU_IDX1':
                f.seek(offset)
                compressed_index = f.read(size)
                index_bytes = zlib.decompress(compressed_index)
                index = json.loads(index_bytes.decode('utf-8'))
                
            elif magic == b'GPU_IDX2':
                # Streaming Read (GPU_IDX2)
                import gzip
                import io
                
                f.seek(offset)
                compressed_bytes = f.read(size)
                
                with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes), mode='rb') as gz:
                    # 1. Header
                    line = gz.readline()
                    header = json.loads(line.decode('utf-8'))
                    params = header["params"]
                    count_files = header.get("count_files", 0)
                    count_frames = header.get("count_frames", 0)
                    
                    files = []
                    # 2. Files Stream
                    for _ in range(count_files):
                        line = gz.readline()
                        if not line: break
                        files.append(json.loads(line.decode('utf-8')))
                        
                    frames = []
                    # 3. Frames Stream
                    for _ in range(count_frames):
                        line = gz.readline()
                        if not line: break
                        frames.append(json.loads(line.decode('utf-8')))
                        
                index = {
                    "files": files,
                    "frames": frames,
                    "params": params,
                    "dictionary": header.get("dictionary")
                }
            else:
                print(f"Erro: Assinatura inválida: {magic}")
                return None
            
        # Adicionar metadados úteis
        index['_archive_path'] = str(archive_path)
        index['_volumes'] = [str(v) for v in volumes_files]
        index['_parent_dir'] = str(parent_dir)
        
        return index
        
    except Exception as e:
        print(f"Erro ao ler índice: {e}")
        import traceback
        traceback.print_exc()
        return None


class TreeNode:
    """Nó da árvore de arquivos/pastas."""
    
    def __init__(self, name: str, is_folder: bool = True, full_path: str = ""):
        self.name = name
        self.is_folder = is_folder
        self.full_path = full_path
        self.size = 0
        self.file_count = 0
        self.children: Dict[str, TreeNode] = {}
        self.parent: Optional[TreeNode] = None
        self.file_entry: Optional[Dict] = None  # Para arquivos, referência ao entry original
    
    def add_path(self, path_parts: List[str], file_size: int, file_entry: Dict):
        """Adiciona um caminho à árvore."""
        if not path_parts:
            return
        
        self.size += file_size
        self.file_count += 1
        
        name = path_parts[0]
        
        if len(path_parts) == 1:
            # É um arquivo
            if name not in self.children:
                child = TreeNode(name, is_folder=False, full_path=file_entry['path_rel'])
                child.parent = self
                child.size = file_size
                child.file_count = 1
                child.file_entry = file_entry
                self.children[name] = child
        else:
            # É uma pasta
            if name not in self.children:
                current_path = "/".join(path_parts[:1]) if self.full_path == "" else f"{self.full_path}/{name}"
                child = TreeNode(name, is_folder=True, full_path=current_path)
                child.parent = self
                self.children[name] = child
            
            self.children[name].add_path(path_parts[1:], file_size, file_entry)
    
    def get_sorted_children(self) -> List['TreeNode']:
        """Retorna filhos ordenados: pastas primeiro, depois arquivos, por nome."""
        folders = sorted([c for c in self.children.values() if c.is_folder], key=lambda x: x.name.lower())
        files = sorted([c for c in self.children.values() if not c.is_folder], key=lambda x: x.name.lower())
        return folders + files
    
    def get_all_files(self) -> List[Dict]:
        """Retorna todos os arquivos (recursivamente) abaixo deste nó."""
        files = []
        if not self.is_folder and self.file_entry:
            files.append(self.file_entry)
        for child in self.children.values():
            files.extend(child.get_all_files())
        return files


def build_file_tree(files: List[Dict]) -> TreeNode:
    """Constrói árvore de arquivos a partir da lista do índice."""
    root = TreeNode("ROOT", is_folder=True, full_path="")
    
    for file_entry in files:
        if file_entry.get('is_duplicate'):
            continue
        
        path = file_entry['path_rel'].replace('\\', '/')
        parts = [p for p in path.split('/') if p]
        
        if parts:
            root.add_path(parts, file_entry['size'], file_entry)
    
    return root


class TerminalExplorer:
    """Explorador de arquivos interativo no terminal."""
    
    def __init__(self, index: Dict[str, Any]):
        self.index = index
        self.root = build_file_tree(index.get('files', []))
        self.current_node = self.root
        self.selected_paths: Set[str] = set()
        self.history: List[TreeNode] = []
    
    def clear_screen(self):
        """Limpa a tela do terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Imprime cabeçalho."""
        print("=" * 70)
        print("📦 EXPLORER - Navegador de Archive Compactado")
        print("=" * 70)
        
        # Mostrar caminho atual
        path = self.current_node.full_path or "/"
        print(f"📁 Caminho: {path}")
        print(f"📊 Selecionados: {len(self.selected_paths)} itens")
        print("-" * 70)
    
    def print_contents(self):
        """Imprime conteúdo da pasta atual."""
        children = self.current_node.get_sorted_children()
        
        if not children:
            print("  (pasta vazia)")
            return
        
        # Cabeçalho da lista
        print(f"{'#':>3}  {'Sel':>3}  {'Tipo':>4}  {'Tamanho':>10}  {'Itens':>8}  Nome")
        print("-" * 70)
        
        for i, child in enumerate(children, 1):
            # Verificar se está selecionado
            is_selected = child.full_path in self.selected_paths
            sel_mark = "[X]" if is_selected else "[ ]"
            
            # Tipo
            type_mark = "📁" if child.is_folder else "📄"
            
            # Tamanho
            size_str = format_bytes(child.size)
            
            # Contagem de itens
            items_str = f"{child.file_count}" if child.is_folder else "-"
            
            # Nome (truncar se muito longo)
            name = child.name
            if len(name) > 40:
                name = name[:37] + "..."
            
            print(f"{i:>3}  {sel_mark}  {type_mark}  {size_str:>10}  {items_str:>8}  {name}")
    
    def print_commands(self):
        """Imprime comandos disponíveis."""
        print("-" * 70)
        print("Comandos:")
        print("  [N]      - Entrar na pasta/ver arquivo N")
        print("  [Enter]  - Voltar para pasta pai")
        print("  s N      - Selecionar/deselecionar item N")
        print("  sa       - Selecionar todos na pasta")
        print("  da       - Deselecionar todos")
        print("  x        - EXTRAIR selecionados")
        print("  q        - Sair")
        print("-" * 70)
    
    def navigate_to(self, index: int):
        """Navega para o item no índice especificado."""
        children = self.current_node.get_sorted_children()
        
        if index < 1 or index > len(children):
            print(f"Índice inválido: {index}")
            return
        
        target = children[index - 1]
        
        if target.is_folder:
            self.history.append(self.current_node)
            self.current_node = target
        else:
            # Mostrar info do arquivo
            print(f"\n📄 Arquivo: {target.name}")
            print(f"   Caminho: {target.full_path}")
            print(f"   Tamanho: {format_bytes(target.size)}")
            if target.file_entry:
                print(f"   Duplicado: {'Sim' if target.file_entry.get('is_duplicate') else 'Não'}")
            input("\nPressione Enter para continuar...")
    
    def go_back(self):
        """Volta para pasta pai."""
        if self.history:
            self.current_node = self.history.pop()
        elif self.current_node.parent:
            self.current_node = self.current_node.parent
    
    def toggle_selection(self, index: int):
        """Alterna seleção do item no índice."""
        children = self.current_node.get_sorted_children()
        
        if index < 1 or index > len(children):
            print(f"Índice inválido: {index}")
            return
        
        target = children[index - 1]
        
        if target.full_path in self.selected_paths:
            self.selected_paths.discard(target.full_path)
            print(f"Deselecionado: {target.name}")
        else:
            self.selected_paths.add(target.full_path)
            print(f"Selecionado: {target.name}")
    
    def select_all(self):
        """Seleciona todos os itens na pasta atual."""
        for child in self.current_node.get_sorted_children():
            self.selected_paths.add(child.full_path)
        print("Todos os itens selecionados")
    
    def deselect_all(self):
        """Remove todas as seleções."""
        self.selected_paths.clear()
        print("Todas as seleções removidas")
    
    def get_files_to_extract(self) -> List[Dict]:
        """Retorna lista de arquivos a extrair baseado na seleção."""
        files_to_extract = []
        
        # Função recursiva para encontrar arquivos
        def find_files(node: TreeNode, prefix: str = ""):
            full_path = node.full_path
            
            # Verificar se este nó ou algum ancestral está selecionado
            is_selected = full_path in self.selected_paths
            
            # Checar ancestrais
            if not is_selected:
                for sel_path in self.selected_paths:
                    if full_path.startswith(sel_path + "/") or sel_path.startswith(full_path + "/"):
                        is_selected = True
                        break
            
            if node.is_folder:
                for child in node.children.values():
                    find_files(child)
            else:
                # É um arquivo - verificar se está selecionado ou se algum ancestral está
                should_extract = full_path in self.selected_paths
                
                if not should_extract:
                    # Verificar se alguma pasta ancestral está selecionada
                    for sel_path in self.selected_paths:
                        if full_path.startswith(sel_path + "/"):
                            should_extract = True
                            break
                
                if should_extract and node.file_entry:
                    files_to_extract.append(node.file_entry)
        
        find_files(self.root)
        return files_to_extract
    
    def extract_selected(self):
        """Inicia extração dos itens selecionados."""
        if not self.selected_paths:
            print("Nenhum item selecionado!")
            input("Pressione Enter para continuar...")
            return
        
        files = self.get_files_to_extract()
        
        print(f"\n📦 Itens a extrair: {len(files)} arquivos")
        total_size = sum(f['size'] for f in files)
        print(f"📊 Tamanho total: {format_bytes(total_size)}")
        
        # Pedir destino
        dest = input("\nDigite o caminho de destino (ou Enter para cancelar): ").strip()
        
        if not dest:
            print("Extração cancelada.")
            input("Pressione Enter para continuar...")
            return
        
        dest_path = Path(dest).resolve()
        
        # Confirmar
        confirm = input(f"\nExtrair para '{dest_path}'? (s/n): ").strip().lower()
        
        if confirm not in ['s', 'sim', 'y', 'yes']:
            print("Extração cancelada.")
            input("Pressione Enter para continuar...")
            return
        
        # Realizar extração
        print(f"\n🚀 Iniciando extração...")
        
        try:
            self._do_extraction(files, dest_path)
            print(f"\n✅ Extração concluída!")
        except Exception as e:
            print(f"\n❌ Erro na extração: {e}")
        
        input("Pressione Enter para continuar...")
    
    def _do_extraction(self, files: List[Dict], dest_path: Path):
        """Executa a extração dos arquivos usando GPU quando disponível."""
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Tentar importar GPU decompressor
        GPU_LZ4_Decompressor = None
        try:
            from gpu_lz4_decompressor import GPU_LZ4_Decompressor as GPU_Decomp
            GPU_LZ4_Decompressor = GPU_Decomp
        except ImportError:
            pass
        
        from decompressor_lz4_ext3 import decompress_lz4_ext3
        import lz4.block
        
        # GPU selector para seleção interativa
        try:
            from gpu_selector import prompt_gpu_selection
            GPU_SELECTOR_AVAILABLE = True
        except ImportError:
            GPU_SELECTOR_AVAILABLE = False
        
        # Carregar metadados do índice
        frames = self.index.get('frames', [])
        params = self.index.get('params', {})
        frame_modes = {int(k): v for k, v in params.get('frame_modes', {}).items()}
        frame_size = params.get('frame_size', 16 * 1024 * 1024)
        
        # Mapear frame_id -> frame_meta para busca rápida
        frames_map = {f['frame_id']: f for f in frames}
        
        # Mapear arquivos por posição no stream
        file_positions = {}
        current_pos = 0
        
        all_files = self.index.get('files', [])
        for f in all_files:
            if not f.get('is_duplicate'):
                file_positions[f['path_rel']] = {
                    'start': current_pos,
                    'end': current_pos + f['size'],
                    'size': f['size']
                }
                current_pos += f['size']
        
        # Calcular frames necessários para a extração
        frames_needed = set()
        for file_entry in files:
            path_rel = file_entry['path_rel']
            if path_rel not in file_positions:
                continue
            
            pos_info = file_positions[path_rel]
            file_start = pos_info['start']
            file_end = pos_info['end']
            
            start_frame = file_start // frame_size
            end_frame = (file_end - 1) // frame_size if file_end > 0 else start_frame
            
            for fid in range(start_frame, end_frame + 1):
                frames_needed.add(fid)
        
        frames_needed = sorted(frames_needed)
        print(f"🔧 Frames a descomprimir: {len(frames_needed)}")
        
        # Inicializar GPU decompressors
        gpu_decompressors = []
        
        if GPU_LZ4_Decompressor is not None:
            try:
                import pyopencl as cl
                
                platforms = cl.get_platforms()
                if platforms:
                    all_gpus = []
                    gpu_indices_map = []
                    global_idx = 0
                    
                    for p in platforms:
                        try:
                            platform_gpus = p.get_devices(device_type=cl.device_type.GPU)
                            for gpu in platform_gpus:
                                all_gpus.append(gpu)
                                gpu_indices_map.append(global_idx)
                                global_idx += 1
                        except:
                            pass
                    
                    if all_gpus:
                        # Prompt para GPU selection se disponível
                        excluded_indices = []
                        if GPU_SELECTOR_AVAILABLE and len(all_gpus) > 1:
                            excluded_indices = prompt_gpu_selection()
                        
                        # Inicializar decompressors
                        for idx, gpu_global_idx in enumerate(gpu_indices_map):
                            if gpu_global_idx in excluded_indices:
                                continue
                            
                            try:
                                decompressor = GPU_LZ4_Decompressor(device_index=gpu_global_idx)
                                if decompressor.enabled:
                                    gpu_decompressors.append(decompressor)
                                    print(f"  ✓ GPU {gpu_global_idx} inicializada")
                            except Exception as e:
                                print(f"  ✗ GPU {gpu_global_idx} falhou: {e}")
                
                if gpu_decompressors:
                    print(f"🎮 GPUs ativas: {len(gpu_decompressors)}")
                else:
                    print("⚠️ Nenhuma GPU disponível, usando CPU")
                    
            except ImportError:
                print("⚠️ PyOpenCL não encontrado, usando CPU")
            except Exception as e:
                print(f"⚠️ Erro GPU: {e}, usando CPU")
        else:
            print("⚠️ GPU Decompressor não disponível, usando CPU")
        
        # Abrir volumes
        parent_dir = Path(self.index['_parent_dir'])
        vol_handles = {}
        vol_locks = {}
        
        def get_vol_handle(vol_name):
            if vol_name not in vol_handles:
                vol_handles[vol_name] = open(parent_dir / vol_name, 'rb')
                vol_locks[vol_name] = threading.Lock()
            return vol_handles[vol_name], vol_locks[vol_name]
        
        # Calcular batch size automático baseado em GPU capabilities
        BATCH_SIZE = 24  # Padrão
        try:
            from gpu_capabilities import get_recommended_batch_size
            BATCH_SIZE = get_recommended_batch_size(frame_size_mb=16)
            print(f"📊 Batch Size automático: {BATCH_SIZE} frames")
        except Exception as e:
            print(f"📊 Batch Size padrão: {BATCH_SIZE} frames")
        
        decompressed_frames = {}  # frame_id -> bytes
        
        start_time = time.time()
        frames_processed = 0
        
        def decompress_frame_cpu(frame_id):
            """Descomprime um frame usando CPU."""
            frame_meta = frames_map.get(frame_id)
            if not frame_meta:
                return None
            
            vol_name = frame_meta['volume_name']
            vol_f, lock = get_vol_handle(vol_name)
            
            with lock:
                vol_f.seek(frame_meta['offset'])
                compressed_data = vol_f.read(frame_meta['compressed_size'])
            
            mode = frame_modes.get(frame_id, 'lz_ext3_gpu')
            
            if mode == 'lz_ext3_gpu':
                return decompress_lz4_ext3(compressed_data, frame_meta['uncompressed_size'])
            elif mode in ['lz4', 'lz4_gpu']:
                return lz4.block.decompress(compressed_data, uncompressed_size=frame_meta['uncompressed_size'])
            else:
                return compressed_data
        
        def decompress_batch_gpu(batch_frame_ids, gpu_decompressor):
            """Descomprime um batch de frames usando GPU."""
            results = {}
            compressed_data_list = []
            uncompressed_sizes = []
            valid_frame_ids = []
            
            for fid in batch_frame_ids:
                frame_meta = frames_map.get(fid)
                if not frame_meta:
                    continue
                
                mode = frame_modes.get(fid, 'lz_ext3_gpu')
                if mode != 'lz_ext3_gpu':
                    # Fallback CPU para modos não-GPU
                    results[fid] = decompress_frame_cpu(fid)
                    continue
                
                vol_name = frame_meta['volume_name']
                vol_f, lock = get_vol_handle(vol_name)
                
                with lock:
                    vol_f.seek(frame_meta['offset'])
                    compressed_data = vol_f.read(frame_meta['compressed_size'])
                
                compressed_data_list.append(compressed_data)
                uncompressed_sizes.append(frame_meta['uncompressed_size'])
                valid_frame_ids.append(fid)
            
            if compressed_data_list:
                try:
                    gpu_results = gpu_decompressor.decompress_batch(compressed_data_list, uncompressed_sizes)
                    
                    for i, fid in enumerate(valid_frame_ids):
                        if gpu_results[i] is not None:
                            results[fid] = gpu_results[i]
                        else:
                            # Fallback CPU
                            results[fid] = decompress_frame_cpu(fid)
                            
                except Exception as e:
                    print(f"⚠️ GPU batch falhou: {e}, fallback CPU")
                    for fid in valid_frame_ids:
                        results[fid] = decompress_frame_cpu(fid)
            
            return results
        
        try:
            # Processar frames em batches
            if gpu_decompressors:
                # Usar GPU com round-robin entre GPUs
                batches = [frames_needed[i:i+BATCH_SIZE] for i in range(0, len(frames_needed), BATCH_SIZE)]
                
                with ThreadPoolExecutor(max_workers=len(gpu_decompressors)) as executor:
                    futures = []
                    
                    for batch_idx, batch in enumerate(batches):
                        gpu_idx = batch_idx % len(gpu_decompressors)
                        future = executor.submit(decompress_batch_gpu, batch, gpu_decompressors[gpu_idx])
                        futures.append(future)
                    
                    for future in futures:
                        batch_results = future.result()
                        decompressed_frames.update(batch_results)
                        frames_processed += len(batch_results)
                        
                        # Progress
                        pct = (frames_processed / len(frames_needed)) * 100
                        print(f"  Descomprimindo: {frames_processed}/{len(frames_needed)} frames ({pct:.0f}%)", end='\r')
            else:
                # Fallback CPU paralelo
                import multiprocessing
                cpu_count = min(multiprocessing.cpu_count(), 8)
                
                with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                    futures = {executor.submit(decompress_frame_cpu, fid): fid for fid in frames_needed}
                    
                    for future in futures:
                        fid = futures[future]
                        try:
                            result = future.result()
                            if result:
                                decompressed_frames[fid] = result
                        except Exception as e:
                            print(f"⚠️ Frame {fid} falhou: {e}")
                        
                        frames_processed += 1
                        if frames_processed % 10 == 0:
                            pct = (frames_processed / len(frames_needed)) * 100
                            print(f"  Descomprimindo: {frames_processed}/{len(frames_needed)} frames ({pct:.0f}%)", end='\r')
            
            print()  # Nova linha após progress
            
            decomp_time = time.time() - start_time
            print(f"⏱️ Descompressão: {decomp_time:.1f}s ({len(frames_needed) / decomp_time:.1f} frames/s)")
            
            # Agora escrever arquivos
            print(f"\n📝 Escrevendo arquivos...")
            files_extracted = 0
            
            for file_entry in files:
                path_rel = file_entry['path_rel']
                
                if path_rel not in file_positions:
                    continue
                
                pos_info = file_positions[path_rel]
                file_start = pos_info['start']
                file_end = pos_info['end']
                file_size = pos_info['size']
                
                start_frame = file_start // frame_size
                end_frame = (file_end - 1) // frame_size if file_end > 0 else start_frame
                
                # Criar arquivo de destino
                dest_file = dest_path / path_rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(dest_file, 'wb') as out_f:
                    for frame_id in range(start_frame, end_frame + 1):
                        uncompressed = decompressed_frames.get(frame_id)
                        if uncompressed is None:
                            continue
                        
                        # Calcular offset dentro do frame
                        frame_stream_start = frame_id * frame_size
                        
                        # Calcular região do arquivo dentro deste frame
                        read_start = max(0, file_start - frame_stream_start)
                        read_end = min(len(uncompressed), file_end - frame_stream_start)
                        
                        if read_start < read_end:
                            out_f.write(uncompressed[read_start:read_end])
                
                files_extracted += 1
                if files_extracted % 100 == 0 or files_extracted == len(files):
                    print(f"  Escritos: {files_extracted}/{len(files)} arquivos", end='\r')
            
            print()
            
            total_time = time.time() - start_time
            print(f"\n📁 {files_extracted} arquivos extraídos em {total_time:.1f}s")
            
        finally:
            for f in vol_handles.values():
                f.close()
            
            # Liberar memória
            decompressed_frames.clear()
    
    def run(self):
        """Loop principal do explorador."""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_contents()
            self.print_commands()
            
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'q':
                    print("Saindo...")
                    break
                
                elif cmd == '':
                    self.go_back()
                
                elif cmd == 'sa':
                    self.select_all()
                    input("Pressione Enter para continuar...")
                
                elif cmd == 'da':
                    self.deselect_all()
                    input("Pressione Enter para continuar...")
                
                elif cmd == 'x':
                    self.extract_selected()
                
                elif cmd.startswith('s '):
                    try:
                        idx = int(cmd[2:])
                        self.toggle_selection(idx)
                        input("Pressione Enter para continuar...")
                    except ValueError:
                        print("Índice inválido")
                        input("Pressione Enter para continuar...")
                
                elif cmd.isdigit():
                    self.navigate_to(int(cmd))
                
                else:
                    print(f"Comando desconhecido: {cmd}")
                    input("Pressione Enter para continuar...")
                    
            except KeyboardInterrupt:
                print("\nSaindo...")
                break
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Explorer - Navegador de Archive Compactado"
    )
    parser.add_argument(
        'archive',
        type=str,
        help='Caminho para qualquer volume do archive (.001, .002, etc)'
    )
    
    args = parser.parse_args()
    
    archive_path = Path(args.archive).resolve()
    
    if not archive_path.exists():
        print(f"Erro: Arquivo não encontrado: {archive_path}")
        return 1
    
    print("📦 Carregando índice do archive...")
    index = read_archive_index(archive_path)
    
    if not index:
        return 1
    
    files = index.get('files', [])
    unique_files = [f for f in files if not f.get('is_duplicate')]
    
    print(f"✓ {len(unique_files)} arquivos encontrados")
    print(f"✓ {len(files) - len(unique_files)} duplicatas")
    
    input("\nPressione Enter para iniciar o explorador...")
    
    explorer = TerminalExplorer(index)
    explorer.run()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
