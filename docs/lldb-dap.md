# lldb-dap
1. Install llvm-17
```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg

sudo ./llvm.sh 17
```

2. Install VSCode LLDB-DAP extension

3. Configure path to lldb-dap
Edit `.vscode/settings.json`:
```json
{
  // ...
  "lldb-dap.executable-path": "/usr/lib/llvm-17/bin/lldb-vscode"
}
```
