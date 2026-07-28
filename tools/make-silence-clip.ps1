<#
.SYNOPSIS
  Writes a 16 kHz mono WAV of low-level noise, for testing the hallucination
  guards.

.DESCRIPTION
  Whisper does not return nothing for silence, it returns subtitle boilerplate.
  Reproducing that needs audio that is quiet but not digitally silent — a real
  gated microphone delivers a noise floor, and it is that noise the decoder
  hallucinates over.

  The level is the interesting parameter. Below the RMS floor in settings.json
  the clip never reaches Whisper at all, so to exercise the two output-side
  guards the level has to be set deliberately above it.

.PARAMETER Path
  Where to write the WAV.

.PARAMETER Dbfs
  RMS level of the generated noise. -55 sits under the default floor, -40 sits
  above it.

.PARAMETER Seconds
  Clip length.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Path,
    [double] $Dbfs = -40,
    [double] $Seconds = 10
)

$rate = 16000
$count = [int]($rate * $Seconds)

# RMS of uniform noise on [-a, a] is a / sqrt(3); solve for the peak that lands
# on the requested level.
$rms = [math]::Pow(10, $Dbfs / 20)
$peak = $rms * [math]::Sqrt(3) * 32767

$random = New-Object System.Random 20260727
$samples = New-Object byte[] ($count * 2)

for ($i = 0; $i -lt $count; $i++) {
    $value = [int](($random.NextDouble() * 2 - 1) * $peak)
    if ($value -lt -32768) { $value = -32768 }
    if ($value -gt 32767) { $value = 32767 }
    $bytes = [BitConverter]::GetBytes([int16]$value)
    $samples[$i * 2] = $bytes[0]
    $samples[$i * 2 + 1] = $bytes[1]
}

$directory = Split-Path -Parent $Path
if ($directory -and -not (Test-Path $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}

$stream = [System.IO.File]::Create($Path)
$writer = New-Object System.IO.BinaryWriter($stream)

$dataBytes = $samples.Length
$writer.Write([char[]]'RIFF')
$writer.Write([uint32](36 + $dataBytes))
$writer.Write([char[]]'WAVE')
$writer.Write([char[]]'fmt ')
$writer.Write([uint32]16)          # PCM chunk size
$writer.Write([uint16]1)           # PCM
$writer.Write([uint16]1)           # mono
$writer.Write([uint32]$rate)
$writer.Write([uint32]($rate * 2)) # byte rate
$writer.Write([uint16]2)           # block align
$writer.Write([uint16]16)          # bits per sample
$writer.Write([char[]]'data')
$writer.Write([uint32]$dataBytes)
$writer.Write($samples)

$writer.Dispose()
$stream.Dispose()

Write-Host ("wrote   {0} ({1:N1} s of noise at {2} dBFS)" -f $Path, $Seconds, $Dbfs)
