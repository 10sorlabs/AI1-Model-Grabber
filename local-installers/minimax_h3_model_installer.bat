@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "TITLE=MiniMax H3"
set "ROOT=%CD%"
set "FAILURES=0"

call :preflight
if errorlevel 1 goto :fatal
call :find_python
if errorlevel 1 goto :fatal
call :update_comfyui
if errorlevel 1 goto :fatal

echo.
echo ============================================================
echo  Installing %TITLE% models - approximately 63.4 GB
echo ============================================================

call :download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" "models\diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors" "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" "models\vae\minimax_h3_video_vae_fp16.safetensors" "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" "models\vae\minimax_h3_audio_vae_fp32.safetensors" "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
if errorlevel 1 set /a FAILURES+=1

echo.
echo ============================================================
echo  Installing %TITLE% custom nodes
echo ============================================================

call :install_node "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git" "8692bc8ef8beaaeee80fd52ba80477dc9e61547b"
if errorlevel 1 set /a FAILURES+=1
call :install_node "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git" "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" "4ee72c065db22c9d96c2427954dc69e7b908444b"
if errorlevel 1 set /a FAILURES+=1

if not "%FAILURES%"=="0" goto :partial_failure

echo.
echo ============================================================
echo  Installation complete
echo ============================================================
echo ComfyUI was updated. Restart it before loading the workflow.
echo.
pause
exit /b 0

:preflight
if not exist "models" (
    echo ERROR: Place this file directly inside the main ComfyUI directory.
    echo The models folder was not found.
    exit /b 1
)
if not exist "custom_nodes" (echo ERROR: The custom_nodes folder was not found.& exit /b 1)
if not exist ".git" (echo ERROR: This ComfyUI directory is not a Git checkout and cannot be updated.& exit /b 1)
where curl.exe >nul 2>nul || (echo ERROR: curl.exe was not found.& exit /b 1)
where git.exe >nul 2>nul || (echo ERROR: Git for Windows was not found.& exit /b 1)
where powershell.exe >nul 2>nul || (echo ERROR: Windows PowerShell was not found.& exit /b 1)
exit /b 0

:find_python
set "PYTHON="
if exist "%ROOT%\.venv\Scripts\python.exe" set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%\venv\Scripts\python.exe" set "PYTHON=%ROOT%\venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%\python_embeded\python.exe" set "PYTHON=%ROOT%\python_embeded\python.exe"
if not defined PYTHON if exist "%ROOT%\python_embedded\python.exe" set "PYTHON=%ROOT%\python_embedded\python.exe"
if not defined PYTHON if exist "%ROOT%\..\python_embeded\python.exe" set "PYTHON=%ROOT%\..\python_embeded\python.exe"
if not defined PYTHON if exist "%ROOT%\..\python_embedded\python.exe" set "PYTHON=%ROOT%\..\python_embedded\python.exe"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 (echo ERROR: The Python environment used by ComfyUI was not found.& exit /b 1)
exit /b 0

:update_comfyui
echo.
echo ============================================================
echo  Updating ComfyUI to the newest official master
echo ============================================================
git.exe -C "%ROOT%" remote set-url origin "https://github.com/Comfy-Org/ComfyUI.git"
if errorlevel 1 (echo [ERROR] Could not configure the ComfyUI remote.& exit /b 1)
git.exe -C "%ROOT%" fetch --prune origin master
if errorlevel 1 (echo [ERROR] Could not download the current ComfyUI version.& exit /b 1)
git.exe -C "%ROOT%" reset --hard origin/master
if errorlevel 1 (echo [ERROR] Could not install the current ComfyUI version.& exit /b 1)
if not exist "%ROOT%\requirements.txt" (echo [ERROR] ComfyUI requirements.txt was not found.& exit /b 1)
"%PYTHON%" -m pip install --disable-pip-version-check -r "%ROOT%\requirements.txt"
if errorlevel 1 (echo [ERROR] ComfyUI requirements failed.& exit /b 1)
echo [OK] ComfyUI is current.
exit /b 0

:download
set "DL_URL=%~1"
set "DL_DEST=%ROOT%\%~2"
set "DL_SHA=%~3"
for %%D in ("%DL_DEST%") do if not exist "%%~dpD" mkdir "%%~dpD"
if exist "%DL_DEST%.part" del /q "%DL_DEST%.part" >nul 2>nul
if exist "%DL_DEST%" (
    call :verify "%DL_DEST%" "%DL_SHA%"
    if not errorlevel 1 (echo [SKIP] Verified: %DL_DEST%& exit /b 0)
    echo [RETRY] Existing file failed verification and will be replaced.
    del /q "%DL_DEST%" >nul 2>nul
)
echo [DOWNLOAD] %DL_DEST%
curl.exe --location --fail --retry 5 --retry-delay 5 --retry-max-time 300 --connect-timeout 30 --output "%DL_DEST%.part" "%DL_URL%"
if errorlevel 1 (
    echo [ERROR] Download failed: %DL_URL%
    if exist "%DL_DEST%.part" del /q "%DL_DEST%.part" >nul 2>nul
    exit /b 1
)
call :verify "%DL_DEST%.part" "%DL_SHA%"
if errorlevel 1 (echo [ERROR] Checksum failed: %DL_DEST%& del /q "%DL_DEST%.part" >nul 2>nul& exit /b 1)
move /y "%DL_DEST%.part" "%DL_DEST%" >nul
if errorlevel 1 (echo [ERROR] Could not finalize: %DL_DEST%& exit /b 1)
echo [OK] %DL_DEST%
exit /b 0

:verify
set "VERIFY_FILE=%~1"
set "VERIFY_SHA=%~2"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $env:VERIFY_FILE).Hash.ToLowerInvariant(); if($actual -ne $env:VERIFY_SHA){exit 1}"
set "VERIFY_RESULT=%ERRORLEVEL%"
set "VERIFY_FILE="
set "VERIFY_SHA="
exit /b %VERIFY_RESULT%

:install_node
set "NODE_NAME=%~1"
set "NODE_URL=%~2"
set "NODE_REF=%~3"
set "NODE_DEST=%ROOT%\custom_nodes\%~1"
if exist "%NODE_DEST%" if not exist "%NODE_DEST%\.git" (echo [ERROR] %NODE_DEST% is not a Git repository.& exit /b 1)
if not exist "%NODE_DEST%\.git" (
    echo [CLONE] %NODE_NAME%
    git.exe clone --filter=blob:none "%NODE_URL%" "%NODE_DEST%"
    if errorlevel 1 (echo [ERROR] Clone failed: %NODE_NAME%& exit /b 1)
)
git.exe -C "%NODE_DEST%" remote set-url origin "%NODE_URL%"
if errorlevel 1 (echo [ERROR] Could not configure: %NODE_NAME%& exit /b 1)
git.exe -C "%NODE_DEST%" cat-file -e "%NODE_REF%^^{commit}" >nul 2>nul
if errorlevel 1 git.exe -C "%NODE_DEST%" fetch --no-tags --filter=blob:none origin "%NODE_REF%"
if errorlevel 1 (echo [ERROR] Fetch failed: %NODE_NAME%& exit /b 1)
git.exe -C "%NODE_DEST%" checkout --detach "%NODE_REF%"
if errorlevel 1 (echo [ERROR] Checkout failed: %NODE_NAME%& exit /b 1)
if exist "%NODE_DEST%\requirements.txt" (
    echo [REQUIREMENTS] %NODE_NAME%
    "%PYTHON%" -m pip install --disable-pip-version-check -r "%NODE_DEST%\requirements.txt"
    if errorlevel 1 (echo [ERROR] Requirements failed: %NODE_NAME%& exit /b 1)
)
echo [OK] %NODE_NAME%
exit /b 0

:partial_failure
echo.
echo ============================================================
echo  Installation finished with %FAILURES% failed item(s)
echo ============================================================
echo Successful items were retained. Fix the errors and run this file again.
echo.
pause
exit /b 1

:fatal
echo.
echo Installation could not start.
echo.
pause
exit /b 1
