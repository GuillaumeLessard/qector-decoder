$root = 'C:\Users\Clinque du Batiment\Downloads\qector-decoder-clone'
$excludePattern = '\\(\.venv|target|__pycache__|\.git|\.hypothesis|\.ruff_cache|\.pytest_cache)\\'
Get-ChildItem -Path $root -Recurse -Include *.rs,*.py -File |
    Where-Object { $_.FullName -notmatch $excludePattern } |
    Select-Object -ExpandProperty FullName |
    Sort-Object |
    Out-File -FilePath "$root\file_list.txt" -Encoding utf8
Write-Host "DONE"
