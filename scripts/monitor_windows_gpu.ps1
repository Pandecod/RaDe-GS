# Run on the Windows HOST of the training server, not another PC.
param([string]$OutputDirectory = [Environment]::GetFolderPath('Desktop'))
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    throw "Output directory does not exist: $OutputDirectory"
}
$runTag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$csvPath = Join-Path $OutputDirectory ("radegs_windows_gpu_" + $runTag + '.csv')
Write-Host "GPU memory log: $csvPath"
Write-Host 'Start training in WSL. Press Ctrl+C here after training finishes.'
try {
    while ($true) {
        $timestamp = [DateTime]::UtcNow.ToString('o')
        try {
            $adapters = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory)
            if ($adapters.Count -eq 0) { throw 'No GPU adapter counters returned' }
            $rows = foreach ($adapter in $adapters) {
                [pscustomobject]@{
                    utc = $timestamp
                    adapter = $adapter.Name
                    dedicated_bytes = $adapter.DedicatedUsage
                    shared_bytes = $adapter.SharedUsage
                    total_committed_bytes = $adapter.TotalCommitted
                    status = 'available'
                    error = ''
                }
            }
        } catch {
            $rows = @([pscustomobject]@{
                utc = $timestamp; adapter = ''; dedicated_bytes = ''
                shared_bytes = ''; total_committed_bytes = ''
                status = 'unavailable'; error = $_.Exception.Message
            })
            Write-Warning $_.Exception.Message
        }
        $rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Append -Encoding UTF8
        Start-Sleep -Seconds 5
    }
} finally {
    Write-Host "Saved GPU memory log: $csvPath"
}
