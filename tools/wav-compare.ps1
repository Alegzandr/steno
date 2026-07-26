# Compares two mono 16-bit PCM WAVs on the numbers that matter for dictation
# quality: level, and how much of the signal a noise gate has zeroed out.
#
# Exactly-zero samples are the tell. A real microphone never delivers a perfect
# zero; a run of them means something upstream (a vendor APO, Windows noise
# suppression, NVIDIA Broadcast) decided that stretch was silence and cut it.
# Whisper hears those cuts.
#
#   .\tools\wav-compare.ps1 broadcast.wav focusrite.wav

param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B
)

$ErrorActionPreference = 'Stop'

function Read-Wav {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 12 -or [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF') {
        throw "$Path is not a RIFF file"
    }

    # Walk the chunks rather than assuming a 44-byte header: a WAV written by
    # another tool may carry LIST or fact chunks before the data.
    $pos = 12
    $fmt = $null
    $dataOffset = -1
    $dataLen = 0
    while ($pos + 8 -le $bytes.Length) {
        $id = [System.Text.Encoding]::ASCII.GetString($bytes, $pos, 4)
        $len = [BitConverter]::ToUInt32($bytes, $pos + 4)
        if ($id -eq 'fmt ') {
            $fmt = [pscustomobject]@{
                Channels = [BitConverter]::ToUInt16($bytes, $pos + 10)
                Rate     = [BitConverter]::ToUInt32($bytes, $pos + 12)
                Bits     = [BitConverter]::ToUInt16($bytes, $pos + 22)
            }
        }
        if ($id -eq 'data') { $dataOffset = $pos + 8; $dataLen = $len; break }
        $pos += 8 + $len + ($len % 2)
    }

    if ($null -eq $fmt) { throw "$Path has no fmt chunk" }
    if ($dataOffset -lt 0) { throw "$Path has no data chunk" }
    if ($fmt.Bits -ne 16) { throw "$Path is $($fmt.Bits)-bit; this script reads 16-bit PCM" }

    # Guard against a truncated file claiming more data than it holds.
    $dataLen = [math]::Min([int64]$dataLen, [int64]($bytes.Length - $dataOffset))
    $frames = [int]($dataLen / (2 * $fmt.Channels))

    # Downmix so a stereo capture stays comparable with a mono one.
    $samples = New-Object 'double[]' $frames
    for ($i = 0; $i -lt $frames; $i++) {
        $sum = 0.0
        for ($c = 0; $c -lt $fmt.Channels; $c++) {
            $sum += [BitConverter]::ToInt16($bytes, $dataOffset + ($i * $fmt.Channels + $c) * 2)
        }
        $samples[$i] = $sum / $fmt.Channels
    }

    [pscustomobject]@{
        Name    = Split-Path $Path -Leaf
        Rate    = $fmt.Rate
        Ch      = $fmt.Channels
        Frames  = $frames
        Samples = $samples
    }
}

function Measure-Wav {
    param($Wav)

    $n = $Wav.Frames
    if ($n -eq 0) { throw "$($Wav.Name) holds no samples" }

    $peak = 0.0; $sumSq = 0.0; $zeros = 0; $run = 0; $longestRun = 0
    foreach ($v in $Wav.Samples) {
        $a = [math]::Abs($v)
        if ($a -gt $peak) { $peak = $a }
        $sumSq += $v * $v
        if ($v -eq 0) {
            $zeros++; $run++
            if ($run -gt $longestRun) { $longestRun = $run }
        } else {
            $run = 0
        }
    }

    $rms = [math]::Sqrt($sumSq / $n)
    $dbfs = { param($x) if ($x -gt 0) { [math]::Round(20 * [math]::Log10($x / 32768.0), 1) } else { [double]::NegativeInfinity } }

    # Envelope at 100 ms, so two recordings of the same sentence line up
    # visually whatever their sample rate.
    $block = [int]($Wav.Rate / 10)
    $envelope = ""
    for ($b = 0; $b * $block -lt $n; $b++) {
        $end = [math]::Min(($b + 1) * $block, $n) - 1
        $p = 0.0
        for ($i = $b * $block; $i -le $end; $i++) {
            $a = [math]::Abs($Wav.Samples[$i]); if ($a -gt $p) { $p = $a }
        }
        $r = $p / 32768.0
        $envelope += if ($r -eq 0) { '_' }
            elseif ($r -lt 0.002) { '.' }
            elseif ($r -lt 0.02) { ':' }
            elseif ($r -lt 0.1) { '=' }
            elseif ($r -lt 0.5) { '#' }
            else { '@' }
    }

    [pscustomobject]@{
        Name       = $Wav.Name
        Format     = "$($Wav.Rate) Hz, $($Wav.Ch) ch"
        Duration   = [math]::Round($n / $Wav.Rate, 2)
        Peak       = [int]$peak
        PeakDbfs   = & $dbfs $peak
        Rms        = [math]::Round($rms, 1)
        RmsDbfs    = & $dbfs $rms
        ZeroPct    = [math]::Round(100 * $zeros / $n, 2)
        LongestGap = [math]::Round(1000 * $longestRun / $Wav.Rate, 0)
        Envelope   = $envelope
    }
}

$left = Measure-Wav (Read-Wav $A)
$right = Measure-Wav (Read-Wav $B)

$rows = @(
    @{ Label = 'file';                A = $left.Name;                    B = $right.Name }
    @{ Label = 'format';              A = $left.Format;                  B = $right.Format }
    @{ Label = 'duration (s)';        A = $left.Duration;                B = $right.Duration }
    @{ Label = 'peak';                A = "$($left.Peak) ($($left.PeakDbfs) dBFS)";  B = "$($right.Peak) ($($right.PeakDbfs) dBFS)" }
    @{ Label = 'rms';                 A = "$($left.Rms) ($($left.RmsDbfs) dBFS)";    B = "$($right.Rms) ($($right.RmsDbfs) dBFS)" }
    @{ Label = 'exactly zero (%)';    A = $left.ZeroPct;                 B = $right.ZeroPct }
    @{ Label = 'longest zero run (ms)'; A = $left.LongestGap;            B = $right.LongestGap }
)

$w = ($rows | ForEach-Object { $_.A.ToString().Length } | Measure-Object -Maximum).Maximum
$w = [math]::Max($w, 20)

""
"{0,-22} {1,-$w}  {2}" -f '', 'A', 'B'
"{0,-22} {1,-$w}  {2}" -f ('-' * 22), ('-' * $w), ('-' * $w)
foreach ($r in $rows) {
    "{0,-22} {1,-$w}  {2}" -f $r.Label, $r.A, $r.B
}

""
"envelope, 100 ms per character  (_ zero  . very low  : low  = mid  # strong  @ saturated)"
"A  $($left.Envelope)"
"B  $($right.Envelope)"
""

$deltaPeak = [math]::Round($left.PeakDbfs - $right.PeakDbfs, 1)
$deltaRms = [math]::Round($left.RmsDbfs - $right.RmsDbfs, 1)
"A is $deltaPeak dB from B on peak, $deltaRms dB on rms."
if ($left.ZeroPct -gt 1 -or $right.ZeroPct -gt 1) {
    "Exactly-zero samples above 1% mean a gate is cutting, not a quiet room."
}
