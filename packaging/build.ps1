# 一键构建 dist\vidsum.exe。默认用 testenv 的 Python（装了全部可选依赖）。
param(
    [string]$Python = "E:\anaconda\envs\testenv\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    Write-Error "找不到 Python：$Python（用 -Python 指定解释器）"
    exit 1
}

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "该环境没装 pyinstaller，先执行：$Python -m pip install pyinstaller"
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
$spec = Join-Path $root "packaging\vidsum.spec"

Write-Host "构建 $root\dist\vidsum\ ..."
& $Python -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $root "dist") `
    --workpath (Join-Path $root "build") `
    $spec
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller 构建失败"
    exit $LASTEXITCODE
}

Write-Host "完成：$(Join-Path $root 'dist\vidsum\vidsum.exe')（onedir，整目录一起拷走）"
Write-Host "把 models\whisper 放到 vidsum.exe 旁边即可离线转写；输出默认落在 exe 旁的 vidsum-out\。"
Write-Host "分发：把 dist\vidsum 整个目录压成 zip，解压后双击 vidsum.exe 就是 Web 控制台。"
