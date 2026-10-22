# 把项目根的 models\whisper 镜像到 dist\vidsum\models\whisper。
#
# 为什么需要它：huggingface_hub 的缓存扫描不认 junction/符号链接，
# 挂 junction 会被当成“本地没有缓存”转而去连 Hub（离线机器上直接
# 报 LocalEntryNotFoundError）。所以 exe 旁边的 models 必须是真目录。
# 同卷用硬链接：秒完成、不占第二份空间；跨卷自动退回复制。
#
# 另外 PyInstaller 每次构建都会清空 dist\vidsum，重建 exe 后跑一次
# 本脚本即可恢复模型目录，不必手动拷 1.5GB。
param(
    [string]$Source,
    [string]$Dest
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = Join-Path $root "models\whisper" }
if (-not $Dest) { $Dest = Join-Path $root "dist\vidsum\models\whisper" }

if (-not (Test-Path $Source)) {
    Write-Error "找不到模型目录：$Source"
    exit 1
}

$srcRoot = (Resolve-Path $Source).Path
$sameVolume = ((Split-Path -Qualifier $srcRoot) -eq (Split-Path -Qualifier $Dest))

$count = 0
Get-ChildItem $srcRoot -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($srcRoot.Length).TrimStart("\")
    $target = Join-Path $Dest $rel
    $dir = Split-Path -Parent $target
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    if (Test-Path $target) { return }
    if ($sameVolume) {
        # 硬链接失败（文件系统不支持等）不中断，下面退回复制
        New-Item -ItemType HardLink -Path $target -Target $_.FullName -ErrorAction SilentlyContinue | Out-Null
    }
    if (-not (Test-Path $target)) {
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
    $count++
}

if ($sameVolume) {
    Write-Host "已镜像 $count 个文件到 $Dest（硬链接）"
} else {
    Write-Host "已镜像 $count 个文件到 $Dest（复制）"
}
