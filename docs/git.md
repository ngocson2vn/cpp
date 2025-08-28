# submodule
```Bash
git submodule add -f https://github.com/tensorflow/tensorflow.git third_party/tensorflow
git config -f .gitmodules submodule.third_party/tensorflow.shallow true
git submodule update --init --recursive

## Manual way
# 1. Create .gitmodules
[submodule "llvm-project"]
	path = llvm-project
	url = https://github.com/llvm/llvm-project.git
	shallow = true

# 2. Update index
git update-index --add --cacheinfo 160000 570885128351868c1308bb22e8ca351d318bc4a1 llvm-project

# 3. Init
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

# Change commit author/user
```Bash
git commit --amend --author="Son Nguyen <ngocson2vn@gmail.com>"

# Per repo
cd repo
git config user.name "Son Nguyen"
git config user.email "ngocson2vn@gmail.com"
```

# Tags
```Bash
git fetch origin --tags --prune
git checkout -b v3.4.0 tags/v3.4.0
```
