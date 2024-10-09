

hello:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 ed 2f 00 00 	mov    0x2fed(%rip),%rax        # 403ff8 <__gmon_start__>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	callq  *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	retq   

Disassembly of section .plt:

0000000000401020 <.plt>:
  401020:	ff 35 e2 2f 00 00    	pushq  0x2fe2(%rip)        # 404008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 e4 2f 00 00    	jmpq   *0x2fe4(%rip)        # 404010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <puts@plt>:
  401030:	ff 25 e2 2f 00 00    	jmpq   *0x2fe2(%rip)        # 404018 <puts@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	pushq  $0x0
  40103b:	e9 e0 ff ff ff       	jmpq   401020 <.plt>

Disassembly of section .text:

0000000000401040 <_start>:
  401040:	31 ed                	xor    %ebp,%ebp
  401042:	49 89 d1             	mov    %rdx,%r9
  401045:	5e                   	pop    %rsi
  401046:	48 89 e2             	mov    %rsp,%rdx
  401049:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40104d:	50                   	push   %rax
  40104e:	54                   	push   %rsp
  40104f:	49 c7 c0 b0 11 40 00 	mov    $0x4011b0,%r8
  401056:	48 c7 c1 50 11 40 00 	mov    $0x401150,%rcx
  40105d:	48 c7 c7 22 11 40 00 	mov    $0x401122,%rdi
  401064:	ff 15 86 2f 00 00    	callq  *0x2f86(%rip)        # 403ff0 <__libc_start_main@GLIBC_2.2.5>
  40106a:	f4                   	hlt    
  40106b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401070 <_dl_relocate_static_pie>:
  401070:	c3                   	retq   
  401071:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401078:	00 00 00 
  40107b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401080 <deregister_tm_clones>:
  401080:	b8 30 40 40 00       	mov    $0x404030,%eax
  401085:	48 3d 30 40 40 00    	cmp    $0x404030,%rax
  40108b:	74 13                	je     4010a0 <deregister_tm_clones+0x20>
  40108d:	b8 00 00 00 00       	mov    $0x0,%eax
  401092:	48 85 c0             	test   %rax,%rax
  401095:	74 09                	je     4010a0 <deregister_tm_clones+0x20>
  401097:	bf 30 40 40 00       	mov    $0x404030,%edi
  40109c:	ff e0                	jmpq   *%rax
  40109e:	66 90                	xchg   %ax,%ax
  4010a0:	c3                   	retq   
  4010a1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4010a8:	00 00 00 00 
  4010ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004010b0 <register_tm_clones>:
  4010b0:	be 30 40 40 00       	mov    $0x404030,%esi
  4010b5:	48 81 ee 30 40 40 00 	sub    $0x404030,%rsi
  4010bc:	48 89 f0             	mov    %rsi,%rax
  4010bf:	48 c1 ee 3f          	shr    $0x3f,%rsi
  4010c3:	48 c1 f8 03          	sar    $0x3,%rax
  4010c7:	48 01 c6             	add    %rax,%rsi
  4010ca:	48 d1 fe             	sar    %rsi
  4010cd:	74 11                	je     4010e0 <register_tm_clones+0x30>
  4010cf:	b8 00 00 00 00       	mov    $0x0,%eax
  4010d4:	48 85 c0             	test   %rax,%rax
  4010d7:	74 07                	je     4010e0 <register_tm_clones+0x30>
  4010d9:	bf 30 40 40 00       	mov    $0x404030,%edi
  4010de:	ff e0                	jmpq   *%rax
  4010e0:	c3                   	retq   
  4010e1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4010e8:	00 00 00 00 
  4010ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004010f0 <__do_global_dtors_aux>:
  4010f0:	80 3d 39 2f 00 00 00 	cmpb   $0x0,0x2f39(%rip)        # 404030 <__TMC_END__>
  4010f7:	75 17                	jne    401110 <__do_global_dtors_aux+0x20>
  4010f9:	55                   	push   %rbp
  4010fa:	48 89 e5             	mov    %rsp,%rbp
  4010fd:	e8 7e ff ff ff       	callq  401080 <deregister_tm_clones>
  401102:	c6 05 27 2f 00 00 01 	movb   $0x1,0x2f27(%rip)        # 404030 <__TMC_END__>
  401109:	5d                   	pop    %rbp
  40110a:	c3                   	retq   
  40110b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401110:	c3                   	retq   
  401111:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  401118:	00 00 00 00 
  40111c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401120 <frame_dummy>:
  401120:	eb 8e                	jmp    4010b0 <register_tm_clones>

0000000000401122 <main>:
  401122:	55                   	push   %rbp
  401123:	48 89 e5             	mov    %rsp,%rbp
  401126:	48 83 ec 10          	sub    $0x10,%rsp
  40112a:	89 7d fc             	mov    %edi,-0x4(%rbp)
  40112d:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  401131:	bf 04 20 40 00       	mov    $0x402004,%edi
  401136:	e8 f5 fe ff ff       	callq  401030 <puts@plt>
  40113b:	b8 00 00 00 00       	mov    $0x0,%eax
  401140:	c9                   	leaveq 
  401141:	c3                   	retq   
  401142:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401149:	00 00 00 
  40114c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401150 <__libc_csu_init>:
  401150:	41 57                	push   %r15
  401152:	49 89 d7             	mov    %rdx,%r15
  401155:	41 56                	push   %r14
  401157:	49 89 f6             	mov    %rsi,%r14
  40115a:	41 55                	push   %r13
  40115c:	41 89 fd             	mov    %edi,%r13d
  40115f:	41 54                	push   %r12
  401161:	4c 8d 25 98 2c 00 00 	lea    0x2c98(%rip),%r12        # 403e00 <__frame_dummy_init_array_entry>
  401168:	55                   	push   %rbp
  401169:	48 8d 2d 98 2c 00 00 	lea    0x2c98(%rip),%rbp        # 403e08 <__init_array_end>
  401170:	53                   	push   %rbx
  401171:	4c 29 e5             	sub    %r12,%rbp
  401174:	48 83 ec 08          	sub    $0x8,%rsp
  401178:	e8 83 fe ff ff       	callq  401000 <_init>
  40117d:	48 c1 fd 03          	sar    $0x3,%rbp
  401181:	74 1b                	je     40119e <__libc_csu_init+0x4e>
  401183:	31 db                	xor    %ebx,%ebx
  401185:	0f 1f 00             	nopl   (%rax)
  401188:	4c 89 fa             	mov    %r15,%rdx
  40118b:	4c 89 f6             	mov    %r14,%rsi
  40118e:	44 89 ef             	mov    %r13d,%edi
  401191:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401195:	48 83 c3 01          	add    $0x1,%rbx
  401199:	48 39 dd             	cmp    %rbx,%rbp
  40119c:	75 ea                	jne    401188 <__libc_csu_init+0x38>
  40119e:	48 83 c4 08          	add    $0x8,%rsp
  4011a2:	5b                   	pop    %rbx
  4011a3:	5d                   	pop    %rbp
  4011a4:	41 5c                	pop    %r12
  4011a6:	41 5d                	pop    %r13
  4011a8:	41 5e                	pop    %r14
  4011aa:	41 5f                	pop    %r15
  4011ac:	c3                   	retq   
  4011ad:	0f 1f 00             	nopl   (%rax)

00000000004011b0 <__libc_csu_fini>:
  4011b0:	c3                   	retq   

Disassembly of section .fini:

00000000004011b4 <_fini>:
  4011b4:	48 83 ec 08          	sub    $0x8,%rsp
  4011b8:	48 83 c4 08          	add    $0x8,%rsp
  4011bc:	c3                   	retq   
