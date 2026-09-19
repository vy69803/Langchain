import unittest
import os
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


class TestProjectSmoke(unittest.TestCase):
    def test_import_langchain_rag(self):
        import langchain_rag
        self.assertIsNotNone(langchain_rag.__all__)
        self.assertIn("create_rag_pipeline", langchain_rag.__all__)
        self.assertIn("create_hybrid_search_engine", langchain_rag.__all__)
        self.assertIn("create_production_chunker", langchain_rag.__all__)

    def test_import_production_api(self):
        from production_api.config import get_settings
        from production_api.main import app

        settings = get_settings()
        self.assertIsNotNone(settings.app_name)
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
