# LIBRARY_PATH
```Bash
LIBRARY_PATH=/usr/local/cuda-12.4/lib64
```
Linker will search libraries in ${LIBRARY_PATH} directories.

# Internal vs External Linkage
`static bool __initialized = true;` (global scope, no namespace)
- Internal linkage, per-TU copies; name is not namespace-mangled (e.g., `__initialized`), but still local, so coexistence is fine.

In the final ELF:
- Static symbol table (`.symtab`): may contain two local entries with the same name, each referencing a different section index and value (address/offset). This is allowed.

# PLT and GOT
https://systemoverlord.com/2017/03/19/got-and-plt-for-pwning.html

[GOT and PLT for pwning](./GOT%20and%20PLT%20for%20pwning%20System%20Overlord.html)

## PLT
PLT stands for Procedure Linkage Table and is partly used by ELF binaries to facilitate dynamic linking.  To speed up program startup, ELF binaries use lazy binding of a procedure address, meaning the address of a procedure isn't known until it's called.

Every dynamically bound program has a PLT containing entries to nonlocal routines.

When a program is compiled, the call to routines that will be dynamically linked are updated to point to the PLT entry.

Initial calls to a PLT entry force the entry to call the runtime linker to resolve the routine's actual address and then jump to the actual address.  Subsequent calls to the now resolved routine are constant time O(1) lookups, only requiring an indirect jump.

## Example of finding the real library function
```Bash
0x00007fe1dc82d594 <+116>:	call   0x7fe1dd9370d0 <_ZN10tensorflow8internal15LogMessageFatalD1Ev@plt>
```
This is a call to a PLT entry at **0x7fe1dd9370d0**.

```Bash
(gdb) disassemble 0x7fe1dd9370d0
Dump of assembler code for function _ZN10tensorflow8internal15LogMessageFatalD1Ev@plt:
   0x00007fe1dd9370d0 <+0>:	jmp    *0xc104a(%rip)        # 0x7fe1dd9f8120 <_ZN10tensorflow8internal15LogMessageFatalD1Ev@got.plt>
   0x00007fe1dd9370d6 <+6>:	push   $0xa0
   0x00007fe1dd9370db <+11>:	jmp    0x7fe1dd9366c0
End of assembler dump.
```
0xc104a(%rip) = 0x00007fe1dd9370d6 + 0xc104a = 0x7fe1dd9f8120<br/>
Next, we need to dereference the address 0x7fe1dd9f8120:
```Bash
(gdb) x/xg 0x7fe1dd9f8120
0x7fe1dd9f8120 <_ZN10tensorflow8internal15LogMessageFatalD1Ev@got.plt>:	0x00007fe1dd7bb450
```

Disassemble 0x00007fe1dd7bb450, we will get the real library function code:
```Bash
(gdb) disassemble 0x00007fe1dd7bb450
Dump of assembler code for function _ZN10tensorflow8internal15LogMessageFatalD1Ev:
   0x00007fe1dd7bb450 <+0>:	mov    0x23c299(%rip),%rax        # 0x7fe1dd9f76f0
   0x00007fe1dd7bb457 <+7>:	push   %rbp
   0x00007fe1dd7bb458 <+8>:	lea    0x18(%rax),%rdx
   0x00007fe1dd7bb45c <+12>:	add    $0x40,%rax
   0x00007fe1dd7bb460 <+16>:	mov    %rdx,(%rdi)
   0x00007fe1dd7bb463 <+19>:	mov    %rsp,%rbp
   0x00007fe1dd7bb466 <+22>:	mov    %rax,0x80(%rdi)
   0x00007fe1dd7bb46d <+29>:	call   0x7fe1dd9829a0 <_ZN10tensorflow8internal10LogMessage18GenerateLogMessageEv@plt>
   0x00007fe1dd7bb472 <+34>:	cmpq   $0x0,0x286376(%rip)        # 0x7fe1dda417f0 <_ZN10tensorflow8internalL19custom_abort_handleE+16>
   0x00007fe1dd7bb47a <+42>:	je     0x7fe1dd7bb43c <_ZN10tensorflow8internal15LogMessageFatalD1Ev.cold.163>
   0x00007fe1dd7bb480 <+48>:	lea    0x286359(%rip),%rdi        # 0x7fe1dda417e0 <_ZN10tensorflow8internalL19custom_abort_handleE>
   0x00007fe1dd7bb487 <+55>:	call   *0x28636b(%rip)        # 0x7fe1dda417f8 <_ZN10tensorflow8internalL19custom_abort_handleE+24>
   0x00007fe1dd7bb48d <+61>:	jmp    0x7fe1dd7bb43c <_ZN10tensorflow8internal15LogMessageFatalD1Ev.cold.163>
End of assembler dump.
```
