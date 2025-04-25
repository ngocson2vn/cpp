# submodule
```Bash
git submodule add -f https://github.com/tensorflow/tensorflow.git third_party/tensorflow
git config -f .gitmodules submodule.third_party/tensorflow.shallow true
git submodule update --init --recursive
```

# shows modified files
```Bash
git diff --name-only
```

# git auto completion
```Bash
sudo apt-get install git-core bash-completion
source /usr/share/bash-completion/completions/git
```

# git fetch origin
```Bash
git init
git remote add origin https://github.com/tensorflow/tensorflow.git
git fetch origin --depth 1 6c5b4ba75d45899bb002c9ad6c8d91663b02f206
git checkout FETCH_HEAD
git checkout -b v1.15.3
```

fetch a remote branch:
```Bash
git fetch origin <remote_branch>:<local_branch>
```

# git merge multiple commits
```Bash
git reset --soft xxxxx
git commit --amend
```

# git delete untracked files
git ls-files --others --exclude-standard | xargs rm -rf

# git search by commit message
git log --all --grep="message"

# Remove submodule
```Bash
# Remove the submodule entry from .git/config
git submodule deinit -f path/to/submodule

# Remove the submodule directory from the superproject's .git/modules directory
rm -rf .git/modules/path/to/submodule

# Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
git rm -f path/to/submodule

# Remove cache
git rm --cached cuda-samples
```

# Create a completely new branch (orphan branch)
```Bash
git checkout --orphan new_branch
```