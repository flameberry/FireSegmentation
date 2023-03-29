@echo off

pushd %~dp0\..\

set DIRECTORY=%~dp0\..\venv

if not exist venv\ (
    python -m venv %DIRECTORY%
)
call %DIRECTORY%\Scripts\activate.bat

pip install numpy tensorflow pillow scikit-image opencv-python

popd
PAUSE