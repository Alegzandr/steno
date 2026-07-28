<#
.SYNOPSIS
  Renders a fixed French passage to a 16 kHz mono WAV using the offline Windows
  speech synthesiser.

.DESCRIPTION
  Produces a reproducible clip for timing transcription, so the realtime factor
  can be compared between backends without a human dictating the same words
  twice at the same speed. Entirely offline: System.Speech drives the SAPI
  voices already installed with Windows.

  The passage is deliberately the kind of thing Steno is for — technical French
  with identifiers, CLI flags and library names — because that is what the
  custom vocabulary and the decoder actually have to cope with.

.PARAMETER Path
  Where to write the WAV.

.PARAMETER Voice
  SAPI voice name. Defaults to the first French voice installed.

.PARAMETER Passage
  Name of the passage file in this directory. `carry-passage.fr.txt` is the
  ninety-second one whose technical terms all fall in the final thirty seconds,
  used to measure whether the vocabulary bias survives past the first window.
#>
[CmdletBinding()]
param(
    [string] $Path = "$PSScriptRoot\..\fixtures\rtf-fr-30s.wav",
    [string] $Voice,
    [string] $Passage = 'rtf-passage.fr.txt'
)

Add-Type -AssemblyName System.Speech

# The passage lives in its own file, read with an explicit encoding, and not in
# a here-string. Windows PowerShell 5.1 decodes a .ps1 with no byte order mark
# as the system ANSI code page, so a UTF-8 script silently turns "périphérique"
# into "pÃ©riphÃ©rique" — and the synthesiser dutifully pronounces the mojibake,
# giving you a clip of French that is not French. Tuned to land near thirty
# seconds with Hortense at her default rate; changing it changes every number
# measured against it.
$here = $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
$passage = (Get-Content -Path (Join-Path $here $Passage) -Encoding UTF8 -Raw)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

if ($Voice) {
    $synth.SelectVoice($Voice)
} else {
    $french = $synth.GetInstalledVoices() |
        Where-Object { $_.VoiceInfo.Culture.Name -like 'fr-*' } |
        Select-Object -First 1
    if (-not $french) {
        throw "No French SAPI voice is installed. Add one under Settings > Time & language > Speech."
    }
    $synth.SelectVoice($french.VoiceInfo.Name)
}

Write-Host "voice   $($synth.Voice.Name) ($($synth.Voice.Culture.Name))"

$directory = Split-Path -Parent $Path
if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }

# 16 kHz mono 16-bit: exactly what the capture pipeline produces, so the clip
# needs no resampling and measures only the decoder.
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)

$synth.SetOutputToWaveFile($Path, $format)
$synth.Speak($passage)
$synth.SetOutputToNull()
$synth.Dispose()

$bytes = (Get-Item $Path).Length
# 44-byte RIFF header, then 2 bytes per sample at 16 kHz.
$seconds = ($bytes - 44) / 32000.0
Write-Host ("wrote   {0} ({1:N1} s, {2:N0} bytes)" -f $Path, $seconds, $bytes)
