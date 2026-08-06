@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "TITLE=Dataset Generator"
set "ROOT=%CD%"
set "FAILURES=0"

call :preflight
if errorlevel 1 goto :fatal
call :find_python
if errorlevel 1 goto :fatal

if not defined HF_TOKEN (
    echo.
    echo This preset contains gated Hugging Face models.
    echo Accept their licenses first, then enter a read token beginning with hf_.
    set /p "HF_TOKEN=Hugging Face token: "
)
if not defined HF_TOKEN (
    echo ERROR: A Hugging Face token is required.
    goto :fatal
)

echo.
echo ============================================================
echo  Installing %TITLE% models
echo ============================================================

call :download "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" "models\text_encoders\qwen_3_8b_fp8mixed.safetensors" "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6" "huggingface"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" "models\diffusion_models\flux-2-klein-9b-fp8.safetensors" "865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee" "huggingface"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" "models\vae\flux2-vae.safetensors" "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/QWEN2512_Bigsloppytits_v1_copy_000003000.safetensors" "models\loras\QWEN2512_Bigsloppytits_v1_copy_000003000.safetensors" "21ea0686c37054f85a63870f1a6633f64154e49a202ed6c2bb793486ae5f55f0" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors" "models\loras\bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors" "1315a08947e5d6d7c53ea4fc59f272e0d54efd7f2da35999b7d873a7cf4fa89b" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" "models\checkpoints\Qwen-Rapid-AIO-NSFW-v23.safetensors" "fdb919fc81bea63f13759967fc92c9118142e5c70d4e6795199233a35eefa233" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/gravedigga/loras/resolve/main/zit_upscaler.safetensors" "models\upscale_models\zit_upscaler.safetensors" "009671cec5a384db31052b52e344e5989b0c51a5ad4d25a8c2c629f658754d13" "none"
if errorlevel 1 set /a FAILURES+=1
call :download "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth" "models\sams\sam_vit_b_01ec64.pth" "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912" "none"
if errorlevel 1 set /a FAILURES+=1

echo.
echo ============================================================
echo  Installing %TITLE% custom nodes
echo ============================================================

call :install_node "ComfyUI-Impact-Subpack" "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git" "50c7b71a6a224734cc9b21963c6d1926816a97f1"
if errorlevel 1 set /a FAILURES+=1
call :install_node "RES4LYF" "https://github.com/ClownsharkBatwing/RES4LYF.git" "e716cd1cb2c5cff90131bf4914b75b75a0489d48"
if errorlevel 1 set /a FAILURES+=1
call :install_node "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git" "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git" "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
if errorlevel 1 set /a FAILURES+=1
call :install_node "seedvr2_videoupscaler" "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git" "4490bd1f482e026674543386bb2a4d176da245b9"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI_Comfyroll_CustomNodes" "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git" "d78b780ae43fcf8c6b7c6505e6ffb4584281ceca"
if errorlevel 1 set /a FAILURES+=1
call :install_node "ComfyUI_FaceAnalysis" "https://github.com/cubiq/ComfyUI_FaceAnalysis.git" "8846653446a6b13582da11793faf950325a398e0"
if errorlevel 1 set /a FAILURES+=1

set "HF_TOKEN="
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
if not exist "custom_nodes" (
    echo ERROR: The custom_nodes folder was not found.
    exit /b 1
)
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
if errorlevel 1 (
    echo ERROR: The Python environment used by ComfyUI was not found.
    exit /b 1
)
exit /b 0

:download
set "DL_URL=%~1"
set "DL_DEST=%ROOT%\%~2"
set "DL_SHA=%~3"
set "DL_AUTH=%~4"
for %%D in ("%DL_DEST%") do if not exist "%%~dpD" mkdir "%%~dpD"
if exist "%DL_DEST%.part" del /q "%DL_DEST%.part" >nul 2>nul
if exist "%DL_DEST%" (
    call :verify "%DL_DEST%" "%DL_SHA%"
    if not errorlevel 1 (
        echo [SKIP] Verified: %DL_DEST%
        exit /b 0
    )
    echo [RETRY] Existing file failed verification and will be replaced.
    del /q "%DL_DEST%" >nul 2>nul
)
echo [DOWNLOAD] %DL_DEST%
if /i "%DL_AUTH%"=="huggingface" (
    curl.exe --location --fail --retry 5 --retry-delay 5 --retry-max-time 300 --connect-timeout 30 -H "Authorization: Bearer %HF_TOKEN%" --output "%DL_DEST%.part" "%DL_URL%"
) else (
    curl.exe --location --fail --retry 5 --retry-delay 5 --retry-max-time 300 --connect-timeout 30 --output "%DL_DEST%.part" "%DL_URL%"
)
if errorlevel 1 (
    echo [ERROR] Download failed: %DL_URL%
    if exist "%DL_DEST%.part" del /q "%DL_DEST%.part" >nul 2>nul
    exit /b 1
)
call :verify "%DL_DEST%.part" "%DL_SHA%"
if errorlevel 1 (
    echo [ERROR] Checksum failed: %DL_DEST%
    del /q "%DL_DEST%.part" >nul 2>nul
    exit /b 1
)
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
if exist "%NODE_DEST%" if not exist "%NODE_DEST%\.git" (
    echo [ERROR] %NODE_DEST% exists but is not a Git repository.
    exit /b 1
)
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
set "HF_TOKEN="
echo.
echo Installation could not start.
echo.
pause
exit /b 1
