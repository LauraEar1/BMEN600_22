# Importing CAPS Sleep Database to OneDrive
  
## Using AWS Command Line Interface

```powershell
aws login

aws s3 sync --no-sign-request s3://physionet-open/capslpdb/1.0.0/  "C:\Users\username\OneDrive - University of Calgary\capslpdb" --exclude "*" --include "data?.fileformat" --include "data??.format"
```
