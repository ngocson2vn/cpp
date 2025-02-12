# Linux x64 Calling Convention
[Linux x64 Calling Convention： Stack Frame ｜ Red Team Notes (1_22_2025 4：44：27 PM).html](./Linux%20x64%20Calling%20Convention：%20Stack%20Frame%20｜%20Red%20Team%20Notes%20(1_22_2025%204：44：27%20PM).html)

# %rip
The special %rip register stores the address of the next instruction to execute. If we jump to another instruction or call another function, %rip is updated. In particular, when we call another function, we must save the caller's next instruction to execute so that we can resume there when the callee finishes. The call instruction does this for us automatically by storing it on the stack, and the ret instruction pops the value off into %rip.

The %rip register on x86-64 is a special-purpose register that always holds the memory address of the next instruction to execute in the program's code segment. The processor increments %rip automatically after each instruction, and control flow instructions like branches set the value of %rip to change the next instruction.
Perhaps surprisingly, %rip also shows up when an assembly program refers to a global variable. See the sidebar under "Addressing modes" below to understand how %rip-relative addressing works.