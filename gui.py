# gui.py - Interface Gráfica Funcional para Compressor/Descompressor LZ4
"""
GUI para o Compressor/Descompressor LZ4 com Deduplicação.

Funcionalidades:
- Explorar arquivos dentro de archives (.001)
- Extrair arquivos selecionados
- Visualizar estatísticas (Tree Map)
- Comprimir diretórios
- Configurar parâmetros do config.txt

Requisitos:
    pip install PySide6
"""

import sys
import os
import tempfile
import webbrowser
import threading
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QTreeView, QSplitter,
    QMessageBox, QLabel, QProgressBar, QTabWidget, QSpinBox,
    QFormLayout, QGroupBox, QTextEdit, QStatusBar, QHeaderView,
    QStyle, QCheckBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal, QObject, QThread
from PySide6.QtGui import QIcon, QFont

# Importar as classes reais do projeto
from explorer import (
    FileItem, FolderItem, ArchiveBackend, SQLiteBackend,
    InMemoryBackend, load_backend, format_bytes
)
from generate_tree_map import read_footer, build_tree, calculate_compression_stats, generate_html
import config_loader


# --- Sinais para comunicação entre threads ---
class WorkerSignals(QObject):
    """Sinais para comunicação entre worker threads e a GUI."""
    progress = Signal(int, str)  # (percentual, mensagem)
    finished = Signal(bool, str)  # (sucesso, mensagem)
    error = Signal(str)
    log = Signal(str)  # Para log detalhado


# --- Worker para Extração em Background ---
class ExtractionWorker(QThread):
    """Thread worker para extrair arquivos em background."""
    
    def __init__(self, backend: ArchiveBackend, selected_paths: List[str], 
                 output_dir: Path, archive_path: Path):
        super().__init__()
        self.backend = backend
        self.selected_paths = selected_paths
        self.output_dir = output_dir
        self.archive_path = archive_path
        self.signals = WorkerSignals()
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            import subprocess
            import tempfile
            import os
            
            # Preparar comando base
            cmd = [
                sys.executable, 
                str(Path(__file__).parent / "decompressor_lz4.py"),
                str(self.archive_path),
                "-o", str(self.output_dir)
            ]
            
            # Verificar se é extração seletiva
            temp_list_path = None
            is_selective = False
            
            if self.selected_paths and not (len(self.selected_paths) == 1 and "" in self.selected_paths):
                 is_selective = True

            if is_selective:
                fd, temp_list_path = tempfile.mkstemp(suffix=".txt", text=True)
                # Ensure we close the handle
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    for p in self.selected_paths:
                        f.write(f"{p}\n")
                
                cmd.extend(["--files-list", temp_list_path])
                self.signals.log.emit(f"Modo Seletivo: Lista de arquivos gerada em {temp_list_path}")
            else:
                self.signals.log.emit("Modo Completo: Extraindo todos os arquivos.")
            
            self.signals.progress.emit(10, "Iniciando descompressão...")
            self.signals.log.emit(f"Comando: {' '.join(cmd)}")
            
            # Executar com captura de output (Unbuffered force)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                encoding='utf-8', errors='replace',
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    self.signals.log.emit(line)
                    # Parse progress
                    import re
                    match = re.search(r"Progresso: (\d+)/(\d+) frames", line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        if total > 0:
                            pct = (current / total) * 100
                            self.signals.progress.emit(int(pct), line)
                    elif "[Meta]" in line:
                         self.signals.progress.emit(99, line)

            process.wait()
            
            # Cleanup temp file
            if temp_list_path and os.path.exists(temp_list_path):
                try:
                    os.unlink(temp_list_path)
                except: pass
            
            if process.returncode == 0:
                self.signals.finished.emit(True, f"Extração concluída em: {self.output_dir}")
            else:
                self.signals.finished.emit(False, f"Erro na extração (código {process.returncode})")
                    
        except Exception as e:
            import traceback
            self.signals.error.emit(f"{str(e)}\n{traceback.format_exc()}")


# --- Worker para Compressão em Background ---
class CompressionWorker(QThread):
    """Thread worker para comprimir diretórios em background."""
    
    def __init__(self, source_dir: Path, output_base: Path, 
                 volume_size_mb: int, use_acls: bool,
                 use_cpu: bool = False):
        super().__init__()
        self.source_dir = source_dir
        self.output_base = output_base
        self.volume_size_mb = volume_size_mb
        self.use_acls = use_acls
        self.use_cpu = use_cpu
        self.signals = WorkerSignals()
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            import subprocess
            
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "compressor_lz4_dedup.py"),
                str(self.source_dir),
                "-o", str(self.output_base),
                "--volume-size-mb", str(self.volume_size_mb)
            ]
            
            if self.use_acls:
                cmd.append("--acls")
            
            if self.use_cpu:
                cmd.append("--cpu")
            
            self.signals.progress.emit(5, "Iniciando compressão...")
            self.signals.log.emit(f"Comando: {' '.join(cmd)}")
            
            # Ambiente com output não-bufferizado
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Executar compressor como subprocesso com captura completa
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            # Ler output linha por linha
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                # Enviar para log
                self.signals.log.emit(line)
                
                # Tentar extrair progresso
                if "Progresso atual" in line or "%" in line:
                    try:
                        match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                        if match:
                            pct = float(match.group(1))
                            self.signals.progress.emit(int(pct), line)
                    except:
                        pass
                elif "[Compressor]" in line or "[Fase" in line or "[Pipeline]" in line:
                    self.signals.progress.emit(-1, line)
            
            process.wait()
            
            if process.returncode == 0:
                self.signals.finished.emit(True, f"Compressão concluída: {self.output_base}.001")
            else:
                self.signals.finished.emit(False, f"Erro na compressão (código {process.returncode}). Veja o log para detalhes.")
                
        except Exception as e:
            import traceback
            self.signals.error.emit(f"{str(e)}\n{traceback.format_exc()}")


# --- Nó interno para o modelo de árvore ---
class TreeNode:
    """Nó interno do modelo de árvore com suporte a filhos lazy-loaded."""
    
    def __init__(self, item, parent=None):
        self.item = item  # FileItem ou FolderItem do backend
        self.parent_node = parent
        self.children_nodes: List[TreeNode] = []
        self.children_loaded = False
        self.path = ""  # Caminho completo
    
    @property
    def name(self):
        return self.item.name
    
    @property
    def is_folder(self):
        return self.item.is_folder
    
    @property
    def size(self):
        return getattr(self.item, 'size', 0)
    
    def child_count(self):
        return len(self.children_nodes)
    
    def child(self, row: int):
        if 0 <= row < len(self.children_nodes):
            return self.children_nodes[row]
        return None
    
    def row(self):
        if self.parent_node:
            try:
                return self.parent_node.children_nodes.index(self)
            except ValueError:
                return 0
        return 0


# --- MODELO DE DADOS PARA QTreeView ---
class ArchiveTreeModel(QAbstractItemModel):
    """Modelo de árvore para exibir conteúdo do archive."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend: Optional[ArchiveBackend] = None
        
        root_item = FolderItem("")
        self.root_node = TreeNode(root_item)
        self.root_node.path = ""
        
        self.checked_paths = set()

    def get_node_path_or_rel(self, index):
        """Helper para obter ID único do nó (path)."""
        if not index.isValid(): return ""
        node = index.internalPointer()
        item = getattr(node, 'item', None)
        entry = getattr(item, 'original_entry', {}) if item else {}
        if entry and 'path_rel' in entry:
            return entry['path_rel']
        return node.path

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        
        # Flags base para todos os itens
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            
        return flags

    def setData(self, index, value, role):
        if role == Qt.CheckStateRole and index.column() == 0:
            path = self.get_node_path_or_rel(index)
            
            if value == Qt.Checked or int(value) == 2:
                self.checked_paths.add(path)
            else:
                self.checked_paths.discard(path)
            
            self.dataChanged.emit(index, index, [role])
            return True
        return False


    def set_backend(self, backend: ArchiveBackend):
        """Define o backend e recarrega o modelo."""
        self.beginResetModel()
        
        if self.backend:
            try:
                self.backend.close()
            except:
                pass
        
        self.backend = backend
        
        # Recriar nó raiz
        root_item = FolderItem("")
        self.root_node = TreeNode(root_item)
        self.root_node.path = ""
        self.root_node.children_loaded = False
        
        self.endResetModel()
        
        # Carregar conteúdo raiz
        self._load_children(self.root_node)

    def _load_children(self, parent_node: TreeNode):
        """Carrega filhos de um nó de pasta."""
        if not self.backend or parent_node.children_loaded:
            return
        
        try:
            items = self.backend.list_dir(parent_node.path)
            
            for item in items:
                child = TreeNode(item, parent_node)
                if parent_node.path:
                    child.path = f"{parent_node.path}/{item.name}"
                else:
                    child.path = item.name
                parent_node.children_nodes.append(child)
            
            parent_node.children_loaded = True
            
        except Exception as e:
            print(f"Erro ao carregar diretório {parent_node.path}: {e}")

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        
        if not parent.isValid():
            parent_node = self.root_node
        else:
            parent_node = parent.internalPointer()
        
        child = parent_node.child(row)
        if child:
            return self.createIndex(row, column, child)
        
        return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        
        child_node = index.internalPointer()
        parent_node = child_node.parent_node
        
        if parent_node is None or parent_node is self.root_node:
            return QModelIndex()
        
        return self.createIndex(parent_node.row(), 0, parent_node)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if not self.backend:
            return 0
        
        if not parent.isValid():
            parent_node = self.root_node
        else:
            parent_node = parent.internalPointer()
        
        # Carregar filhos se ainda não carregados
        if parent_node.is_folder and not parent_node.children_loaded:
            self._load_children(parent_node)
        
        return parent_node.child_count()

    def columnCount(self, parent=QModelIndex()):
        return 8  # Nome, Tamanho, Comp., Data, Volume, Atributos, Frame, Dup

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        
        node = index.internalPointer()
        
        if role == Qt.DisplayRole:
            col = index.column()
            if col == 0:
                return node.name
            
            if node.is_folder:
                return ""
            
            # Access via item wrapper (TreeNode -> FileItem -> original_entry)
            item = getattr(node, 'item', None)
            entry = getattr(item, 'original_entry', {}) if item else {}
            if not entry:
                 entry = getattr(node, 'original_entry', {}) # Fallback
            
            if col == 1: # Tamanho Original
                return format_bytes(node.size)
            
            elif col == 2: # Tamanho Comprimido
                comp_size = entry.get('compressed_size', 0)
                if comp_size:
                    return format_bytes(comp_size)
                return "-"
                
            elif col == 3: # Data Modificação
                mtime = entry.get('mtime', 0)
                if mtime:
                    import datetime
                    return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                return "-"
                
            elif col == 4: # Volume
                vol = entry.get('volume_name')
                if not vol:
                    vol = entry.get('frame_vol', "-")
                return vol
                
            elif col == 5: # Atributos
                attrs = entry.get('attrs', 0)
                return f"0x{attrs:X}" if attrs else "-"
                
            elif col == 6: # Frame
                fid = entry.get('sframe_id')
                return str(fid) if fid is not None else "-"
                
            elif col == 7: # Dup
                is_dup = entry.get('is_duplicate', False)
                return "Sim" if is_dup else "Não"
        
        elif role == Qt.DecorationRole and index.column() == 0:
            # Ícones do sistema
            style = QApplication.style()
            if node.is_folder:
                return style.standardIcon(QStyle.SP_DirIcon)
            else:
                return style.standardIcon(QStyle.SP_FileIcon)
        
        elif role == Qt.CheckStateRole and index.column() == 0:
            path = self.get_node_path_or_rel(index)
            return Qt.Checked if path in self.checked_paths else Qt.Unchecked

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            headers = [
                "Nome", "Tamanho", "Tamanho Comp.", "Data Modificação", "Volume", "Atributos", "Frame", "Dup"
            ]
            if section < len(headers):
                return headers[section]
        return None

    def hasChildren(self, parent: QModelIndex = QModelIndex()) -> bool:
        if not self.backend:
            return False
        
        if not parent.isValid():
            return True
        
        node = parent.internalPointer()
        return node.is_folder

    def canFetchMore(self, parent: QModelIndex) -> bool:
        if not self.backend:
            return False
        
        if not parent.isValid():
            node = self.root_node
        else:
            node = parent.internalPointer()
        
        return node.is_folder and not node.children_loaded

    def fetchMore(self, parent: QModelIndex):
        if not self.backend:
            return
        
        if not parent.isValid():
            node = self.root_node
        else:
            node = parent.internalPointer()
        
        if node.is_folder and not node.children_loaded:
            self.beginInsertRows(parent, 0, 0)
            self._load_children(node)
            self.endInsertRows()

    def get_node_path(self, index: QModelIndex) -> str:
        """Retorna o caminho completo de um nó."""
        if not index.isValid():
            return ""
        node = index.internalPointer()
        return node.path


# --- JANELA PRINCIPAL ---
class MainWindow(QMainWindow):
    """Janela principal da aplicação."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🗜️ LZ4 GPU Compressor - Explorer")
        self.setGeometry(100, 100, 1100, 750)
        
        self.backend: Optional[ArchiveBackend] = None
        self.archive_path: Optional[Path] = None
        self.extraction_worker = None
        self.compression_worker = None
        
        # Widgets de configuração (para referência)
        self.config_widgets: Dict[str, QWidget] = {}
        
        self._setup_ui()
        self._connect_signals()
        self._update_ui_state(False)

    def _setup_ui(self):
        """Configura a interface do usuário."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # === TAB 1: Explorer ===
        self._setup_explorer_tab()
        
        # === TAB 2: Compressor ===
        self._setup_compressor_tab()
        
        # === TAB 3: Configurações ===
        self._setup_settings_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Pronto")

    def _setup_explorer_tab(self):
        """Configura a aba do Explorer."""
        explorer_tab = QWidget()
        explorer_layout = QVBoxLayout(explorer_tab)
        
        # Barra de abertura de arquivo
        file_group = QGroupBox("Arquivo do Archive")
        file_layout = QHBoxLayout(file_group)
        
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Caminho para o arquivo .001")
        self.browse_button = QPushButton("📂 Procurar")
        self.open_button = QPushButton("📦 Abrir Archive")
        
        file_layout.addWidget(self.path_input)
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.open_button)
        explorer_layout.addWidget(file_group)
        
        # Informações e Ações
        actions_layout = QHBoxLayout()
        self.info_label = QLabel("Status: Nenhum arquivo aberto")
        self.extract_button = QPushButton("📤 Extrair Tudo")
        self.stats_button = QPushButton("📊 Ver Estatísticas")
        
        actions_layout.addWidget(self.info_label)
        actions_layout.addStretch(1)
        self.extract_selected_button = QPushButton("✅ Extrair Selecionados")
        self.extract_all_button = QPushButton("📤 Extrair TUDO")
        
        actions_layout.addWidget(self.extract_selected_button)
        actions_layout.addWidget(self.extract_all_button)
        actions_layout.addWidget(self.stats_button)
        explorer_layout.addLayout(actions_layout)

        # Barra de Busca
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar arquivos (ex: %.txt)")
        self.search_button = QPushButton("🔍 Buscar")
        self.reset_search_button = QPushButton("❌ Limpar")
        
        self.select_all_button = QPushButton("☑️ Selecionar Todos")
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.select_all_button)
        search_layout.addWidget(self.reset_search_button)
        explorer_layout.addLayout(search_layout)
        
        # Explorador de Arquivos (QTreeView)
        self.tree_view = QTreeView()
        self.tree_model = ArchiveTreeModel()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.setAlternatingRowColors(True)
        
        # Ajustar colunas
        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        explorer_layout.addWidget(self.tree_view)
        
        # Barra de progresso e log
        self.explorer_progress = QProgressBar()
        self.explorer_progress.setVisible(False)
        explorer_layout.addWidget(self.explorer_progress)
        
        self.explorer_log = QTextEdit()
        self.explorer_log.setReadOnly(True)
        self.explorer_log.setMaximumHeight(100)
        self.explorer_log.setVisible(False)
        explorer_layout.addWidget(self.explorer_log)
        
        self.tabs.addTab(explorer_tab, "📁 Explorer")

    def _setup_compressor_tab(self):
        """Configura a aba do Compressor."""
        compress_tab = QWidget()
        compress_layout = QVBoxLayout(compress_tab)
        
        # Configurações de compressão
        config_group = QGroupBox("Configurações de Compressão")
        config_form = QFormLayout(config_group)
        
        # Pasta de origem
        source_layout = QHBoxLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Pasta de origem para comprimir")
        self.source_browse = QPushButton("📂")
        source_layout.addWidget(self.source_input)
        source_layout.addWidget(self.source_browse)
        config_form.addRow("Origem:", source_layout)
        
        # Arquivo de saída
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Caminho base para o arquivo de saída")
        self.output_browse = QPushButton("📂")
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(self.output_browse)
        config_form.addRow("Saída:", output_layout)
        
        # Parâmetros
        self.volume_size_spin = QSpinBox()
        self.volume_size_spin.setRange(1, 4096)
        self.volume_size_spin.setValue(98)
        self.volume_size_spin.setSuffix(" MB")
        config_form.addRow("Tamanho do Volume:", self.volume_size_spin)
        
        # Opções
        options_layout = QHBoxLayout()
        self.acls_check = QCheckBox("Incluir ACLs/Metadados")
        self.cpu_check = QCheckBox("Forçar modo CPU")
        # Ler modo CPU das configurações
        self.cpu_check.setChecked(config_loader.is_force_cpu_mode())
        options_layout.addWidget(self.acls_check)
        options_layout.addWidget(self.cpu_check)
        options_layout.addStretch(1)
        config_form.addRow("Opções:", options_layout)
        
        compress_layout.addWidget(config_group)
        
        # Botão de compressão
        compress_actions = QHBoxLayout()
        compress_actions.addStretch(1)
        self.compress_button = QPushButton("🗜️ Iniciar Compressão")
        self.compress_button.setMinimumHeight(40)
        compress_actions.addWidget(self.compress_button)
        compress_actions.addStretch(1)
        compress_layout.addLayout(compress_actions)
        
        # Barra de progresso de compressão
        self.compress_progress = QProgressBar()
        self.compress_progress.setVisible(False)
        compress_layout.addWidget(self.compress_progress)
        
        # Log de compressão
        log_group = QGroupBox("Log de Compressão")
        log_layout = QVBoxLayout(log_group)
        self.compress_log = QTextEdit()
        self.compress_log.setReadOnly(True)
        self.compress_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.compress_log)
        compress_layout.addWidget(log_group)
        
        self.tabs.addTab(compress_tab, "🗜️ Compressor")

    def _setup_settings_tab(self):
        """Configura a aba de Configurações."""
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Scroll area para configurações
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Carregar configurações atuais
        config = config_loader.load_config(force_reload=True)
        
        # === Compressor GPU ===
        comp_group = QGroupBox("🖥️ Compressor LZ4 GPU")
        comp_form = QFormLayout(comp_group)
        
        self.cfg_force_cpu = QCheckBox()
        self.cfg_force_cpu.setChecked(config.get('FORCE_CPU_MODE', False))
        self.cfg_force_cpu.setToolTip("Forçar uso exclusivo de CPU, ignorando GPU")
        comp_form.addRow("Forçar Modo CPU:", self.cfg_force_cpu)
        self.config_widgets['FORCE_CPU_MODE'] = self.cfg_force_cpu
        
        self.cfg_comp_batch = QSpinBox()
        self.cfg_comp_batch.setRange(1, 256)
        self.cfg_comp_batch.setValue(config.get('COMPRESSOR_BATCH_SIZE', 50) or 50)
        self.cfg_comp_batch.setToolTip("Frames por batch na GPU (mais = mais VRAM)")
        comp_form.addRow("Batch Size Compressor:", self.cfg_comp_batch)
        self.config_widgets['COMPRESSOR_BATCH_SIZE'] = self.cfg_comp_batch
        
        self.cfg_read_buffer = QSpinBox()
        self.cfg_read_buffer.setRange(0, 16)
        self.cfg_read_buffer.setValue(config.get('READ_BUFFER_BATCHES', 1))
        self.cfg_read_buffer.setToolTip("Batches em buffer de leitura (0=infinito)")
        comp_form.addRow("Read Buffer Batches:", self.cfg_read_buffer)
        self.config_widgets['READ_BUFFER_BATCHES'] = self.cfg_read_buffer
        
        self.cfg_write_buffer = QSpinBox()
        self.cfg_write_buffer.setRange(0, 16)
        self.cfg_write_buffer.setValue(config.get('WRITE_BUFFER_BATCHES', 1))
        self.cfg_write_buffer.setToolTip("Batches em buffer de escrita (0=infinito)")
        comp_form.addRow("Write Buffer Batches:", self.cfg_write_buffer)
        self.config_widgets['WRITE_BUFFER_BATCHES'] = self.cfg_write_buffer
        
        scroll_layout.addWidget(comp_group)
        
        # === Decompressor GPU ===
        decomp_group = QGroupBox("📤 Decompressor LZ4 GPU")
        decomp_form = QFormLayout(decomp_group)
        
        self.cfg_decomp_batch = QSpinBox()
        self.cfg_decomp_batch.setRange(1, 256)
        self.cfg_decomp_batch.setValue(config.get('DECOMPRESSOR_BATCH_SIZE', 40) or 40)
        self.cfg_decomp_batch.setToolTip("Frames por batch na descompressão")
        decomp_form.addRow("Batch Size Decomp:", self.cfg_decomp_batch)
        self.config_widgets['DECOMPRESSOR_BATCH_SIZE'] = self.cfg_decomp_batch
        
        self.cfg_max_threads = QSpinBox()
        self.cfg_max_threads.setRange(1, 32)
        self.cfg_max_threads.setValue(config.get('MAX_WORKER_THREADS', 2))
        self.cfg_max_threads.setToolTip("Threads paralelas para descompressão")
        decomp_form.addRow("Max Worker Threads:", self.cfg_max_threads)
        self.config_widgets['MAX_WORKER_THREADS'] = self.cfg_max_threads
        
        self.cfg_gpu_fallback = QCheckBox()
        self.cfg_gpu_fallback.setChecked(config.get('GPU_FALLBACK_ENABLED', True))
        self.cfg_gpu_fallback.setToolTip("Se GPU falhar, usar CPU automaticamente")
        decomp_form.addRow("GPU Fallback:", self.cfg_gpu_fallback)
        self.config_widgets['GPU_FALLBACK_ENABLED'] = self.cfg_gpu_fallback
        
        scroll_layout.addWidget(decomp_group)
        
        # === Scanner / IO ===
        io_group = QGroupBox("📂 Scanner / IO")
        io_form = QFormLayout(io_group)
        
        self.cfg_scan_workers = QSpinBox()
        self.cfg_scan_workers.setRange(1, 32)
        self.cfg_scan_workers.setValue(config.get('NUM_SCAN_WORKERS', 4))
        self.cfg_scan_workers.setToolTip("Workers paralelos para scan de diretórios")
        io_form.addRow("Scan Workers:", self.cfg_scan_workers)
        self.config_widgets['NUM_SCAN_WORKERS'] = self.cfg_scan_workers
        
        scroll_layout.addWidget(io_group)
        
        # === Deduplicator ===
        dedup_group = QGroupBox("🔄 Deduplicator")
        dedup_form = QFormLayout(dedup_group)
        
        self.cfg_io_workers = QSpinBox()
        self.cfg_io_workers.setRange(1, 32)
        self.cfg_io_workers.setValue(config.get('NUM_IO_WORKERS', 4))
        self.cfg_io_workers.setToolTip("Workers I/O para leitura paralela")
        dedup_form.addRow("IO Workers:", self.cfg_io_workers)
        self.config_widgets['NUM_IO_WORKERS'] = self.cfg_io_workers
        
        self.cfg_num_readers = QSpinBox()
        self.cfg_num_readers.setRange(1, 32)
        self.cfg_num_readers.setValue(config.get('NUM_READERS', 4))
        self.cfg_num_readers.setToolTip("Threads de leitura para hash GPU")
        dedup_form.addRow("Num Readers:", self.cfg_num_readers)
        self.config_widgets['NUM_READERS'] = self.cfg_num_readers
        
        self.cfg_buffer_size = QSpinBox()
        self.cfg_buffer_size.setRange(32, 1024)
        self.cfg_buffer_size.setValue(config.get('BUFFER_SIZE', 256))
        self.cfg_buffer_size.setToolTip("Tamanho do buffer entre readers e GPU")
        dedup_form.addRow("Buffer Size:", self.cfg_buffer_size)
        self.config_widgets['BUFFER_SIZE'] = self.cfg_buffer_size
        
        scroll_layout.addWidget(dedup_group)
        
        # === GPU Kernel (Avançado) ===
        kernel_group = QGroupBox("⚙️ GPU Kernel (Avançado)")
        kernel_form = QFormLayout(kernel_group)
        
        self.cfg_hash_log = QSpinBox()
        self.cfg_hash_log.setRange(14, 24)
        self.cfg_hash_log.setValue(config.get('HASH_LOG', 20))
        self.cfg_hash_log.setToolTip("Log2 das entradas na tabela hash")
        kernel_form.addRow("Hash Log:", self.cfg_hash_log)
        self.config_widgets['HASH_LOG'] = self.cfg_hash_log
        
        self.cfg_hash_candidates = QSpinBox()
        self.cfg_hash_candidates.setRange(2, 16)
        self.cfg_hash_candidates.setValue(config.get('HASH_CANDIDATES', 8))
        self.cfg_hash_candidates.setToolTip("Candidatos por entrada (Top-K)")
        kernel_form.addRow("Hash Candidates:", self.cfg_hash_candidates)
        self.config_widgets['HASH_CANDIDATES'] = self.cfg_hash_candidates
        
        self.cfg_good_match = QSpinBox()
        self.cfg_good_match.setRange(32, 512)
        self.cfg_good_match.setValue(config.get('GOOD_ENOUGH_MATCH', 256))
        self.cfg_good_match.setToolTip("Match 'bom o suficiente' (para busca)")
        kernel_form.addRow("Good Enough Match:", self.cfg_good_match)
        self.config_widgets['GOOD_ENOUGH_MATCH'] = self.cfg_good_match
        
        scroll_layout.addWidget(kernel_group)
        
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        settings_layout.addWidget(scroll)
        
        # Botões de ação
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        
        self.reload_config_btn = QPushButton("🔄 Recarregar")
        self.reload_config_btn.setToolTip("Recarregar configurações do arquivo")
        buttons_layout.addWidget(self.reload_config_btn)
        
        self.save_config_btn = QPushButton("💾 Salvar Configurações")
        self.save_config_btn.setToolTip("Salvar alterações no config.txt")
        buttons_layout.addWidget(self.save_config_btn)
        
        buttons_layout.addStretch(1)
        settings_layout.addLayout(buttons_layout)
        
        self.tabs.addTab(settings_tab, "⚙️ Configurações")

    def _connect_signals(self):
        """Conecta os sinais aos slots."""
        # Explorer tab
        self.browse_button.clicked.connect(self._browse_archive)
        self.open_button.clicked.connect(self._open_archive)
        self.extract_selected_button.clicked.connect(self._extract_selected)
        self.extract_all_button.clicked.connect(self._extract_all)
        self.stats_button.clicked.connect(self._show_stats)
        self.tree_view.expanded.connect(self._handle_expansion)
        self.search_button.clicked.connect(self._search_files)
        self.reset_search_button.clicked.connect(self._reset_search)
        self.search_input.returnPressed.connect(self._search_files)
        self.select_all_button.clicked.connect(self._select_all_search)

        # Compressor tab
        self.source_browse.clicked.connect(self._browse_source)
        self.output_browse.clicked.connect(self._browse_output)
        self.compress_button.clicked.connect(self._start_compression)
        
        # Settings tab
        self.reload_config_btn.clicked.connect(self._reload_config)
        self.save_config_btn.clicked.connect(self._save_config)

    def _update_ui_state(self, is_open: bool):
        """Atualiza o estado da UI baseado em se há um archive aberto."""
        self.extract_selected_button.setEnabled(is_open)
        self.extract_all_button.setEnabled(is_open)
        self.stats_button.setEnabled(is_open)
        self.tree_view.setEnabled(is_open)
        
        if is_open and self.backend:
            self.info_label.setText(f"✅ {self.backend.get_info()}")
        else:
            self.info_label.setText("Status: Nenhum arquivo aberto")

    # === Explorer Methods ===
    
    def _browse_archive(self):
        """Abre diálogo para selecionar arquivo do archive."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Arquivo do Archive",
            str(Path.home()),
            "Arquivos de Archive (*.001 *.002 *.003);;Todos os Arquivos (*)"
        )
        if file_name:
            self.path_input.setText(file_name)

    def _open_archive(self):
        """Carrega o backend do archive."""
        path_str = self.path_input.text()
        if not path_str:
            QMessageBox.warning(self, "Aviso", "Por favor, insira o caminho do arquivo.")
            return
        
        archive_path = Path(path_str)
        if not archive_path.exists():
            QMessageBox.critical(self, "Erro", f"Arquivo não encontrado: {archive_path}")
            return
        
        try:
            self.status_bar.showMessage("Carregando archive...")
            QApplication.processEvents()
            
            new_backend = load_backend(archive_path)
            
            if new_backend:
                self.archive_path = archive_path
                self.backend = new_backend
                self.tree_model.set_backend(new_backend)
                self._update_ui_state(True)
                self.status_bar.showMessage(f"Archive aberto: {archive_path.name}")
            else:
                QMessageBox.critical(
                    self, "Erro", 
                    "Não foi possível carregar o arquivo.\n"
                    "Verifique se é um arquivo de archive válido (.001)."
                )
                self._update_ui_state(False)
                
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Erro", f"Erro ao abrir archive:\n{e}\n\n{traceback.format_exc()}")
            self._update_ui_state(False)

    def _handle_expansion(self, index: QModelIndex):
        """Carrega conteúdo ao expandir uma pasta."""
        if self.tree_model.canFetchMore(index):
            self.tree_model.fetchMore(index)

    def _extract_selected(self):
        self._extract_files(extract_mode='selected')

    def _extract_all(self):
        self._extract_files(extract_mode='all')
        
    def _select_all_search(self):
        """Seleciona todos os itens visíveis no nível raiz (útil para busca)."""
        if not self.backend: return
        
        root = self.tree_model.root_node
        # Só faz sentido se tiver filhos carregados (busca ou pasta raiz)
        if not root.children_nodes:
            return
            
        added_count = 0
        for node in root.children_nodes:
            item = getattr(node, 'item', None)
            entry = getattr(item, 'original_entry', {}) if item else {}
            path = entry.get('path_rel') or node.path
            
            if path:
                self.tree_model.checked_paths.add(path)
                added_count += 1
        
        if added_count > 0:
            # Emitir sinal para atualizar UI
            tl = self.tree_model.index(0, 0, QModelIndex())
            br = self.tree_model.index(self.tree_model.rowCount()-1, 0, QModelIndex())
            self.tree_model.dataChanged.emit(tl, br, [Qt.CheckStateRole])

    def _extract_files(self, extract_mode="auto"):
        """Extrai arquivos. mode='auto' (selecionado ou tudo), 'selected', 'all'."""
        if not self.backend or not self.archive_path:
            return
        
        selected_paths = set(self.tree_model.checked_paths)
        
        # Se for All, ignorar seleção
        if extract_mode == 'all':
            selected_paths = set()
        
        # Se seleção vazia e não é All, tentar seleção da UI
        if not selected_paths and extract_mode != 'all':
            selected_indexes = self.tree_view.selectionModel().selectedIndexes()
            for index in selected_indexes:
                if index.column() == 0:
                    node = index.internalPointer() # TreeNode
                    item = getattr(node, 'item', None)
                    entry = getattr(item, 'original_entry', {}) if item else {}
                    path = entry.get('path_rel') or self.tree_model.get_node_path(index)
                    if path:
                        selected_paths.add(path)
        
        # Lógica de Decisão
        if extract_mode == 'all':
             reply = QMessageBox.question(
                self, "Extrair Tudo",
                "Deseja extrair TODOS os arquivos do archive?",
                QMessageBox.Yes | QMessageBox.No
             )
             if reply != QMessageBox.Yes: return
             selected_paths = {""} # Raiz
             
        elif not selected_paths:
             # Nada selecionado
             if extract_mode == 'selected':
                 QMessageBox.information(self, "Aviso", "Nenhum arquivo selecionado.")
                 return
             else: # auto
                reply = QMessageBox.question(
                    self, "Extrair Tudo",
                    "Nenhum item selecionado. Deseja extrair todos os arquivos?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes: return
                selected_paths = {""}
        
        # Selecionar pasta de destino
        output_dir = QFileDialog.getExistingDirectory(
            self, "Selecionar Pasta de Destino",
            str(Path.home())
        )
        
        if not output_dir:
            return
        
        # Iniciar extração em background
        self.explorer_progress.setVisible(True)
        self.explorer_progress.setRange(0, 100)
        self.explorer_progress.setValue(0)
        self.explorer_log.setVisible(True)
        self.explorer_log.clear()
        
        self.extract_selected_button.setEnabled(False)
        self.extract_all_button.setEnabled(False)
        
        self.extraction_worker = ExtractionWorker(
            self.backend,
            list(selected_paths),
            Path(output_dir),
            self.archive_path
        )
        
        self.extraction_worker.signals.progress.connect(self._on_extraction_progress)
        self.extraction_worker.signals.finished.connect(self._on_extraction_finished)
        self.extraction_worker.signals.error.connect(self._on_extraction_error)
        self.extraction_worker.signals.log.connect(self._on_extraction_log)
        
        self.extraction_worker.start()

    def _on_extraction_progress(self, percent: int, message: str):
        """Atualiza progresso da extração."""
        if percent >= 0:
            self.explorer_progress.setValue(percent)
        self.status_bar.showMessage(message)

    def _on_extraction_log(self, message: str):
        """Adiciona mensagem ao log."""
        self.explorer_log.append(message)

    def _on_extraction_finished(self, success: bool, message: str):
        """Finaliza extração."""
        self.explorer_progress.setVisible(False)
        self.extract_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Sucesso", message)
        else:
            QMessageBox.warning(self, "Aviso", message)
        
        self.status_bar.showMessage("Pronto")

    def _on_extraction_error(self, error: str):
        """Trata erro na extração."""
        self.explorer_progress.setVisible(False)
        self.extract_button.setEnabled(True)
        self.explorer_log.append(f"❌ ERRO: {error}")
        QMessageBox.critical(self, "Erro", f"Erro durante extração:\n{error}")
        self.status_bar.showMessage("Erro na extração")

    def _show_stats(self):
        """Mostra estatísticas do archive (Tree Map)."""
        if not self.archive_path:
            return
        
        try:
            self.status_bar.showMessage("Gerando estatísticas...")
            QApplication.processEvents()
            
            # Encontrar o último volume (onde o índice está)
            parent_dir = self.archive_path.parent
            name_stem = self.archive_path.name
            
            # Resolver prefixo (ex: data.001 -> data)
            import re
            match = re.match(r"(.*)\.(\d{3})$", name_stem)
            prefix = match.group(1) if match else name_stem
            
            # Encontrar todos os volumes
            volumes = sorted([v for v in parent_dir.glob(f"{prefix}.*") if re.match(r".*\.\d{3}$", v.name)])
            if not volumes:
                QMessageBox.warning(self, "Aviso", "Nenhum volume encontrado.")
                return
            
            last_vol = volumes[-1]
            self.status_bar.showMessage(f"Lendo índice de {last_vol.name}...")
            QApplication.processEvents()
            
            # Ler footer e gerar dados
            index = read_footer(last_vol)
            
            if not index:
                QMessageBox.warning(self, "Aviso", f"Não foi possível ler o índice do arquivo {last_vol.name}.")
                return
            
            # Construir árvore e calcular estatísticas
            tree = build_tree(index.get('files', []))
            stats = calculate_compression_stats(index)
            
            # Gerar HTML em arquivo temporário
            temp_dir = Path(tempfile.gettempdir())
            html_path = temp_dir / f"treemap_{self.archive_path.stem}.html"
            
            generate_html(tree.to_dict(), stats, html_path)
            
            # Abrir no navegador
            webbrowser.open(str(html_path))
            
            self.status_bar.showMessage(f"Estatísticas abertas em: {html_path}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Erro", f"Erro ao gerar estatísticas:\n{e}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("Erro ao gerar estatísticas")

    def _search_files(self):
        """Realiza busca de arquivos no backend."""
        if not self.backend:
            return
            
        pattern = self.search_input.text().strip()
        if not pattern:
            return
            
        # Adicionar wildcards automaticamente se o usuário não forneceu
        if '%' not in pattern:
            pattern = f"%{pattern}%"
            
        try:
            self.status_bar.showMessage(f"Buscando: {pattern}...")
            # Usar o método de busca do backend (se suportado)
            if hasattr(self.backend, 'search'):
                results = self.backend.search(pattern)
            else:
                # Fallback simples para InMemoryBackend se não tiver search específico
                results = [] 
                
            if not results:
                QMessageBox.information(self, "Busca", "Nenhum arquivo encontrado.")
                self.status_bar.showMessage("Nenhum arquivo encontrado.")
                return

            # Criar uma árvore temporária plana com os resultados
            # Usar ArchiveTreeModel com um root customizado
            from explorer import FolderItem, FileItem # Garante imports
            
            search_root_item = FolderItem(f"Resultados: {pattern}")
            search_root_node = TreeNode(search_root_item)
            search_root_node.children_loaded = True 
            search_root_node.path = f"Search:{pattern}"
            
            for entry in results:
                # Entry é um dict. Precisamos converter para FileItem
                name = entry.get('path_rel', '??').rsplit('/', 1)[-1]
                size = entry.get('size', 0)
                item = FileItem(name, size, entry)
                
                # Criar nó wrapper
                node = TreeNode(item, parent=search_root_node)
                search_root_node.children_nodes.append(node)
            
            # Atualizar modelo (Hack: substituir root node)
            self.tree_model.beginResetModel()
            self.tree_model.root_node = search_root_node
            self.tree_model.path_cache = {} 
            self.tree_model.endResetModel()
            
            self.tree_view.expandAll()
            self.status_bar.showMessage(f"Encontrados {len(results)} arquivos.")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na busca: {e}")

    def _reset_search(self):
        """Limpa a busca e restaura a visualização original."""
        if not self.backend:
            return
        
        self.search_input.clear()
        # Restaurar root original recarregando backend no model
        # Simplesmente chamando set_backend com o mesmo backend
        self.tree_model.set_backend(self.backend)
        self.status_bar.showMessage("Visualização restaurada.")


    # === Compressor Methods ===
    
    def _browse_source(self):
        """Abre diálogo para selecionar pasta de origem."""
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar Pasta de Origem",
            str(Path.home())
        )
        if folder:
            self.source_input.setText(folder)
            # Sugerir nome de saída
            if not self.output_input.text():
                source_name = Path(folder).name
                self.output_input.setText(str(Path(folder).parent / f"{source_name}_compressed"))

    def _browse_output(self):
        """Abre diálogo para selecionar arquivo de saída."""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Selecionar Arquivo de Saída",
            str(Path.home()),
            "Archive (*.001);;Todos os Arquivos (*)"
        )
        if file_name:
            # Remover extensão se presente
            output_base = Path(file_name).with_suffix('')
            self.output_input.setText(str(output_base))

    def _start_compression(self):
        """Inicia o processo de compressão."""
        source_str = self.source_input.text()
        output_str = self.output_input.text()
        
        if not source_str:
            QMessageBox.warning(self, "Aviso", "Por favor, selecione a pasta de origem.")
            return
        
        if not output_str:
            QMessageBox.warning(self, "Aviso", "Por favor, selecione o arquivo de saída.")
            return
        
        source_dir = Path(source_str)
        if not source_dir.is_dir():
            QMessageBox.critical(self, "Erro", f"Pasta não encontrada: {source_dir}")
            return
        
        output_base = Path(output_str)
        
        # Confirmar
        reply = QMessageBox.question(
            self, "Confirmar Compressão",
            f"Comprimir:\n{source_dir}\n\nPara:\n{output_base}.001\n\n"
            f"Volume Size: {self.volume_size_spin.value()} MB\n"
            f"Modo CPU: {'Sim' if self.cpu_check.isChecked() else 'Não'}\n"
            f"ACLs: {'Sim' if self.acls_check.isChecked() else 'Não'}\n\n"
            "Continuar?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Iniciar compressão
        self.compress_log.clear()
        self.compress_progress.setVisible(True)
        self.compress_progress.setRange(0, 100)
        self.compress_progress.setValue(0)
        self.compress_button.setEnabled(False)
        
        self.compression_worker = CompressionWorker(
            source_dir,
            output_base,
            self.volume_size_spin.value(),
            use_acls=self.acls_check.isChecked(),
            use_cpu=self.cpu_check.isChecked()
        )
        
        self.compression_worker.signals.progress.connect(self._on_compression_progress)
        self.compression_worker.signals.finished.connect(self._on_compression_finished)
        self.compression_worker.signals.error.connect(self._on_compression_error)
        self.compression_worker.signals.log.connect(self._on_compression_log)
        
        self.compression_worker.start()

    def _on_compression_progress(self, percent: int, message: str):
        """Atualiza progresso da compressão."""
        if percent >= 0:
            self.compress_progress.setValue(percent)
        self.status_bar.showMessage(message)

    def _on_compression_log(self, message: str):
        """Adiciona ao log de compressão."""
        self.compress_log.append(message)
        # Scroll para baixo
        scrollbar = self.compress_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_compression_finished(self, success: bool, message: str):
        """Finaliza compressão."""
        self.compress_progress.setVisible(False)
        self.compress_button.setEnabled(True)
        
        if success:
            self.compress_log.append(f"\n✅ {message}")
            QMessageBox.information(self, "Sucesso", message)
        else:
            self.compress_log.append(f"\n❌ {message}")
            QMessageBox.warning(self, "Aviso", message)
        
        self.status_bar.showMessage("Pronto")

    def _on_compression_error(self, error: str):
        """Trata erro na compressão."""
        self.compress_progress.setVisible(False)
        self.compress_button.setEnabled(True)
        self.compress_log.append(f"\n❌ ERRO: {error}")
        QMessageBox.critical(self, "Erro", f"Erro durante compressão:\n{error}")
        self.status_bar.showMessage("Erro na compressão")

    # === Settings Methods ===
    
    def _reload_config(self):
        """Recarrega configurações do arquivo."""
        config = config_loader.load_config(force_reload=True)
        
        # Atualizar widgets
        self.cfg_force_cpu.setChecked(config.get('FORCE_CPU_MODE', False))
        self.cfg_comp_batch.setValue(config.get('COMPRESSOR_BATCH_SIZE', 50) or 50)
        self.cfg_read_buffer.setValue(config.get('READ_BUFFER_BATCHES', 1))
        self.cfg_write_buffer.setValue(config.get('WRITE_BUFFER_BATCHES', 1))
        self.cfg_decomp_batch.setValue(config.get('DECOMPRESSOR_BATCH_SIZE', 40) or 40)
        self.cfg_max_threads.setValue(config.get('MAX_WORKER_THREADS', 2))
        self.cfg_gpu_fallback.setChecked(config.get('GPU_FALLBACK_ENABLED', True))
        self.cfg_scan_workers.setValue(config.get('NUM_SCAN_WORKERS', 4))
        self.cfg_io_workers.setValue(config.get('NUM_IO_WORKERS', 4))
        self.cfg_num_readers.setValue(config.get('NUM_READERS', 4))
        self.cfg_buffer_size.setValue(config.get('BUFFER_SIZE', 256))
        self.cfg_hash_log.setValue(config.get('HASH_LOG', 20))
        self.cfg_hash_candidates.setValue(config.get('HASH_CANDIDATES', 8))
        self.cfg_good_match.setValue(config.get('GOOD_ENOUGH_MATCH', 256))
        
        self.status_bar.showMessage("Configurações recarregadas")
        QMessageBox.information(self, "Configurações", "Configurações recarregadas do arquivo.")

    def _save_config(self):
        """Salva configurações no arquivo config.txt."""
        config_path = Path(__file__).parent / "config.txt"
        
        try:
            # Ler arquivo atual
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Mapeamento de valores
            values = {
                'FORCE_CPU_MODE': str(self.cfg_force_cpu.isChecked()),
                'COMPRESSOR_BATCH_SIZE': str(self.cfg_comp_batch.value()),
                'READ_BUFFER_BATCHES': str(self.cfg_read_buffer.value()),
                'WRITE_BUFFER_BATCHES': str(self.cfg_write_buffer.value()),
                'DECOMPRESSOR_BATCH_SIZE': str(self.cfg_decomp_batch.value()),
                'MAX_WORKER_THREADS': str(self.cfg_max_threads.value()),
                'GPU_FALLBACK_ENABLED': str(self.cfg_gpu_fallback.isChecked()),
                'NUM_SCAN_WORKERS': str(self.cfg_scan_workers.value()),
                'NUM_IO_WORKERS': str(self.cfg_io_workers.value()),
                'NUM_READERS': str(self.cfg_num_readers.value()),
                'BUFFER_SIZE': str(self.cfg_buffer_size.value()),
                'HASH_LOG': str(self.cfg_hash_log.value()),
                'HASH_CANDIDATES': str(self.cfg_hash_candidates.value()),
                'GOOD_ENOUGH_MATCH': str(self.cfg_good_match.value()),
            }
            
            # Atualizar linhas
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    key = stripped.split('=')[0].strip()
                    if key in values:
                        # Preservar indentação e comentários inline
                        parts = line.split('=', 1)
                        prefix = parts[0]
                        # Verificar se há comentário inline
                        rest = parts[1] if len(parts) > 1 else ''
                        if '#' in rest:
                            comment = '#' + rest.split('#', 1)[1]
                            new_lines.append(f"{prefix}= {values[key]}  {comment}")
                        else:
                            new_lines.append(f"{prefix}= {values[key]}\n")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # Escrever arquivo
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # Forçar reload do cache
            config_loader.load_config(force_reload=True)
            
            self.status_bar.showMessage("Configurações salvas")
            QMessageBox.information(self, "Configurações", f"Configurações salvas em:\n{config_path}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Erro", f"Erro ao salvar configurações:\n{e}\n\n{traceback.format_exc()}")

    def closeEvent(self, event):
        """Limpa recursos ao fechar."""
        if self.backend:
            try:
                self.backend.close()
            except:
                pass
        
        if self.extraction_worker and self.extraction_worker.isRunning():
            self.extraction_worker.cancel()
            self.extraction_worker.wait(2000)
        
        if self.compression_worker and self.compression_worker.isRunning():
            self.compression_worker.cancel()
            self.compression_worker.wait(2000)
        
        event.accept()


def main():
    """Ponto de entrada principal."""
    app = QApplication(sys.argv)
    
    # Aplicar estilo escuro (opcional)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
