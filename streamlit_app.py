# Entry point for `streamlit run streamlit_app.py`
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ui.dashboard import *   # noqa – dashboard registers all Streamlit pages
