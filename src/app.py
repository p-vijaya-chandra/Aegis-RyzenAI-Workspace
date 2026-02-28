import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from llm_engine import AMDHybridLLM
from rag_pipeline import LocalRAG

class AegisWorkspaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aegis Workspace - AMD Ryzen AI Secured")
        self.setGeometry(100, 100, 800, 600)

        # Initialize AI Backends
        self.rag = LocalRAG()
        self.llm = None # Initialized lazily to save startup time

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready (Air-Gapped Mode)")
        layout.addWidget(self.status_label)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about your local files...")
        layout.addWidget(self.query_input)

        self.submit_btn = QPushButton("Ask Aegis")
        self.submit_btn.clicked.connect(self.process_query)
        layout.addWidget(self.submit_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def process_query(self):
        user_query = self.query_input.text()
        if not user_query: return

        self.chat_history.append(f"<b>You:</b> {user_query}")
        self.query_input.clear()
        QApplication.processEvents() # Force UI update

        # Lazy load LLM
        if not self.llm:
            self.status_label.setText("Status: Loading AMD Ryzen AI Models...")
            QApplication.processEvents()
            self.llm = AMDHybridLLM()

        self.status_label.setText("Status: NPU Retrieving Context...")
        QApplication.processEvents()
        context = self.rag.retrieve_context(user_query)

        prompt = f"Using ONLY the following local context, answer the question.\nContext: {context}\nQuestion: {user_query}"

        self.status_label.setText("Status: iGPU Generating Response...")
        QApplication.processEvents()
        
        # Generate Answer
        response = self.llm.generate_response(prompt)
        
        self.chat_history.append(f"<b>Aegis (AMD Local AI):</b> {response}<br>")
        self.status_label.setText("Status: Ready (Air-Gapped Mode)")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QWidget
    app = QApplication(sys.argv)
    window = AegisWorkspaceApp()
    window.show()
    sys.exit(app.exec_())
