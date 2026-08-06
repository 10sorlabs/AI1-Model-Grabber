@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "TITLE=Krea 2"
set "ROOT=%CD%"
set "FAILURES=0"

call :preflight
if errorlevel 1 goto :fatal
call :find_python
if errorlevel 1 goto :fatal

echo.
echo ============================================================
echo  Installing %TITLE% models
echo ============================================================

call :download "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" "models\diffusion_models\krea2_turbo_fp8_scaled.safetensors" "eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" "models\text_encoders\qwen3vl_4b_fp8_scaled.safetensors" "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors" "models\vae\qwen_image_vae.safetensors" "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/MysticXXX_KREA2_v3.safetensors" "models\loras\MysticXXX_KREA2_v3.safetensors" "3437fb03a86c0a3be180b3da67f6a042a13d928d79d46b8835c3d04be6ec4382" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/pawg_krea2.safetensors" "models\loras\pawg_krea2.safetensors" "6df1ae992e4ac7ae2e5576b01074f31cc6ba20c442711d9b87e920996c24c30d" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/RealisticSnapshotKrea2.safetensors" "models\loras\RealisticSnapshotKrea2.safetensors" "dfd67eb881bec4caeb9409e5b7c127adcb9fb724e6007e1e77165af51e6f908d" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/4xNMKDSuperscale_4xNMKDSuperscale.pt" "models\upscale_models\4xNMKDSuperscale_4xNMKDSuperscale.pt" "1d1b0078fe71446e0469d8d4df59e96baa80d83cda600d68237d655830821bcc" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" "models\ultralytics\bbox\face_yolov8m.pt" "717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth" "models\sams\sam_vit_b_01ec64.pth" "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912" "none"
if errorlevel 1 set /a FAILURES+=1

echo.
echo ============================================================
echo  Installing %TITLE% custom nodes
echo ============================================================

call :install_node "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git" "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git" "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI-Impact-Subpack" "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git" "50c7b71a6a224734cc9b21963c6d1926816a97f1"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git" "8692bc8ef8beaaeee80fd52ba80477dc9e61547b"
if errorlevel 1 set /a FAILURES+=1
call :install_node "RES4LYF" "https://github.com/ClownsharkBatwing/RES4LYF.git" "e716cd1cb2c5cff90131bf4914b75b75a0489d48"
if errorlevel 1 set /a FAILURES+=1

if not "%FAILURES%"=="0" goto :partial_failure

echo.
echo ============================================================
echo  Installation complete
echo ============================================================
echo Restart ComfyUI before loading the workflow.
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
