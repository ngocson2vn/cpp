# undefined symbol: SSL_load_error_strings
**Root cause:** <br/>
The openssl API `SSL_load_error_strings()` is defined in an old openssl version. <br/>
(1) The loaded .so file was built from a collection of .cpp files. There is some .cpp file that calls `SSL_load_error_strings()`.
(2) The loaded .so file was linked with a static library which calls `SSL_load_error_strings()`.

**Solution:**
(1) Find the .cpp file and update it with new openssl API.
(2) Find the static library and upgrade it.
