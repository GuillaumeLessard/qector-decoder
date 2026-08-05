$root = 'C:\Users\Clinque du Batiment\Downloads\qector-decoder-clone'
$files = Get-Content "$root\file_list.txt"
$results = foreach ($f in $files) {
    if (Test-Path -LiteralPath $f) {
        $lines = (Get-Content -LiteralPath $f -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        [PSCustomObject]@{ Path = $f; Lines = $lines }
    }
}
$results | Sort-Object Lines -Descending | ForEach-Object { "$($_.Lines)`t$($_.Path)" } | Out-File -FilePath "$root\line_counts.txt" -Encoding utf8
$total = ($results | Measure-Object -Property Lines -Sum).Sum
"TOTAL_LINES:$total" | Out-File -FilePath "$root\line_counts.txt" -Encoding utf8 -Append
Write-Host "DONE total=$total files=$($results.Count)"
