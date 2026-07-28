<#
.SYNOPSIS
  Reads nvidia-smi around the whole life of the Steno process.

.DESCRIPTION
  The acceptance criterion is that used video memory before launch and after
  quit are the same figure. This measures exactly that, against the real binary
  rather than a harness.

  Quitting is done by posting WM_CLOSE to Steno's own window. That is a message
  to our own process, not synthetic input into the desktop session, and unlike
  taskkill it runs the normal exit sequence — which is the thing under test:
  TerminateProcess would skip `RunEvent::Exit`, and with it the model unload and
  the shutdown of a server Steno started.

  What this does NOT cover: showing the window, dictating and running a cleanup.
  Those need the global shortcut held down by a human. `examples/vram.rs both`
  covers the same load-and-release path through the same code.

.PARAMETER Exe
  The Steno binary to measure.
#>
[CmdletBinding()]
param(
    [string] $Exe = "F:\dev\steno\src-tauri\target-cuda\release\steno.exe",
    [int] $SettleSeconds = 8
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win {
  public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr w, IntPtr l);
}
"@

function Used {
    (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
}

function Report($label, $value) {
    "{0,-34} {1,6} MiB" -f $label, $value | Write-Host
}

$baseline = Used
Report 'baseline, Steno not running' $baseline

$process = Start-Process -FilePath $Exe -PassThru
Start-Sleep -Seconds $SettleSeconds
$running = Used
Report 'launched, window hidden' $running

# Every top-level window belonging to the process, hidden ones included:
# MainWindowHandle is zero while the window is not visible.
$handles = New-Object System.Collections.ArrayList
# Not $pid: that is a read-only automatic variable in PowerShell, and assigning
# to it throws once per enumerated window.
$callback = [Win+EnumProc] {
    param($hWnd, $lParam)
    $owner = [uint32]0
    [void][Win]::GetWindowThreadProcessId($hWnd, [ref]$owner)
    if ($owner -eq $process.Id) { [void]$handles.Add($hWnd) }
    return $true
}
[void][Win]::EnumWindows($callback, [IntPtr]::Zero)

if ($handles.Count -eq 0) {
    Write-Warning "no window found for pid $($process.Id); cannot close it cleanly"
    exit 1
}

# WM_CLOSE
foreach ($h in $handles) { [void][Win]::PostMessage($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) }

if (-not $process.WaitForExit(30000)) {
    Write-Warning "Steno did not exit within 30 s; the clean-quit path is not working"
    exit 1
}

# The driver does not release instantly once the process is gone.
Start-Sleep -Seconds 3
$after = Used
Report 'after quit' $after

Write-Host ''
Report 'held while hidden' ([int]$running - [int]$baseline)
$residue = [int]$after - [int]$baseline
Report 'residue after quit' $residue
$verdict = if ([math]::Abs($residue) -le 64) { 'clean' } else { 'MEMORY NOT RETURNED' }
"{0,-34} {1}" -f 'verdict', $verdict | Write-Host
