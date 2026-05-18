@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python convert_compressed_parnets_to_images.py
pause