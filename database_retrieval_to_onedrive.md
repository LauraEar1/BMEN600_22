# Importing CAPS Sleep Database to OneDrive
  
## Using AWS Command Line Interface

### Create AWS account and IAM credentials

### Download and import selected files to OneDrive using AWS CLI

aws login

aws s3 sync --no-sign-request s3://physionet-open/capslpdb/1.0.0/  "C:\Users\jawad\OneDrive - University of Calgary\capslpdb" --exlude "*" --include "data?.fileformat" --include "data??.format"
