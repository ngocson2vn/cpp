# Python Source Code
```Bash
ls -l ~/.pyenv/sources/3.11.2/Python-3.11.2
```

# Debug Frame
```C++
-exec p PyUnicode_AsUTF8(frame.f_code.co_filename)
-exec p PyUnicode_AsUTF8(frame.f_code.co_name)
```