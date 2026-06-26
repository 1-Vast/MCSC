$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")

python main.py mcsc --stage full
python main.py deepbaseline
python main.py graphbaseline
python main.py moltransbaseline
python main.py sotaevidence
python main.py check
python main.py verifygate
python -m compileall -q main.py model scripts
