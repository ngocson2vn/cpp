# Keys: The primary differences are in threading model and in separate physical memories
- the smallest executable unit of parallelism on a CUDA device comprises 32 threads (termed a warp of threads)
- Modern NVIDIA GPUs can support up to 2048 active threads concurrently per multiprocessor (see Features and Specifications of the CUDA C++ Programming Guide) On GPUs with 80 multiprocessors, this leads to more than 160,000 concurrently active threads.
- Because separate registers are allocated to all active threads, no swapping of registers or other state need occur when switching among GPU threads. Resources stay allocated to each thread until it completes its execution.

# Best Practices
- To use CUDA, data values must be transferred from the host to the device. These transfers are costly in terms of performance and should be minimized.  
For example, transferring two matrices to the device to perform a matrix addition and then transferring the results back to the host will not realize much performance benefit. The issue here is the number of operations performed per data element transferred. For the preceding procedure, assuming matrices of size $N \times N$, there are $N^2$ operations (additions) and $3N^2$ elements transferred, so the ratio of operations to elements transferred is 1:3 or O(1). Performance benefits can be more readily achieved when this ratio is higher. For example, a matrix multiplication of the same matrices requires N3 operations (multiply-add), so the ratio of operations to elements transferred is O(N), in which case the larger the matrix the greater the performance benefit. The types of operations are an additional factor, as additions have different complexity profiles than, for example, trigonometric functions. It is important to include the overhead of transferring data to and from the device in determining whether operations should be performed on the host or on the device.

- All kernel launches are asynchronous, as are memory-copy functions with the Async suffix on their names.

- it is necessary to synchronize the CPU thread with the GPU by calling `cudaDeviceSynchronize()` immediately before starting and stopping the CPU timer. `cudaDeviceSynchronize()` blocks the calling CPU thread until all CUDA calls previously issued by the thread are completed.

# Memory Optimization
## Check PCIe specs
```bash
# lspci -vv

# Search for NVIDIA
af:00.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
	Subsystem: NVIDIA Corporation TU104GL [Tesla T4]
	Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx+
	Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
	Latency: 0
	Interrupt: pin A routed to IRQ 434
	NUMA node: 2
	Region 0: Memory at ed000000 (32-bit, non-prefetchable) [size=16M]
	Region 1: Memory at 39bfc0000000 (64-bit, prefetchable) [size=256M]
	Region 3: Memory at 39bff0000000 (64-bit, prefetchable) [size=32M]
	Capabilities: [60] Power Management version 3
		Flags: PMEClk- DSI- D1- D2- AuxCurrent=375mA PME(D0+,D1-,D2-,D3hot+,D3cold+)
		Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
	Capabilities: [68] #00 [0080]
	Capabilities: [78] Express (v2) Endpoint, MSI 00
		DevCap:	MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
			ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset- SlotPowerLimit 0.000W
		DevCtl:	Report errors: Correctable- Non-Fatal+ Fatal+ Unsupported-
			RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop-
			MaxPayload 256 bytes, MaxReadReq 512 bytes
		DevSta:	CorrErr- UncorrErr- FatalErr- UnsuppReq- AuxPwr+ TransPend-
		LnkCap:	Port #0, Speed 8GT/s, Width x16, ASPM not supported, Exit Latency L0s <1us, L1 <4us
			ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
		LnkCtl:	ASPM Disabled; RCB 64 bytes Disabled- CommClk+
			ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
		LnkSta:	Speed 8GT/s, Width x16, TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
		DevCap2: Completion Timeout: Range AB, TimeoutDis+, LTR-, OBFF Via message
		DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
		LnkCtl2: Target Link Speed: 8GT/s, EnterCompliance- SpeedDis-
			 Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
			 Compliance De-emphasis: -6dB
		LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+, EqualizationPhase1+
			 EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
	Capabilities: [c8] MSI-X: Enable+ Count=6 Masked-
		Vector table: BAR=0 offset=00b90000
		PBA: BAR=0 offset=00ba0000
	Capabilities: [100 v1] Virtual Channel
		Caps:	LPEVC=0 RefClk=100ns PATEntryBits=1
		Arb:	Fixed- WRR32- WRR64- WRR128-
		Ctrl:	ArbSelect=Fixed
		Status:	InProgress-
		VC0:	Caps:	PATOffset=00 MaxTimeSlots=1 RejSnoopTrans-
			Arb:	Fixed- WRR32- WRR64- WRR128- TWRR128- WRR256-
			Ctrl:	Enable+ ID=0 ArbSelect=Fixed TC/VC=01
			Status:	NegoPending- InProgress-
	Capabilities: [258 v1] L1 PM Substates
		L1SubCap: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1- L1_PM_Substates+
		L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-

		L1SubCtl2:
	Capabilities: [128 v1] Power Budgeting <?>
	Capabilities: [420 v2] Advanced Error Reporting
		UESta:	DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
		UEMsk:	DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
		UESvrt:	DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
		CESta:	RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
		CEMsk:	RxErr+ BadTLP+ BadDLLP+ Rollover+ Timeout+ NonFatalErr+
		AERCap:	First Error Pointer: 00, GenCap- CGenEn- ChkCap- ChkEn-
	Capabilities: [600 v1] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
	Capabilities: [900 v1] #19
	Capabilities: [bb0 v1] #15
	Capabilities: [bcc v1] Single Root I/O Virtualization (SR-IOV)
		IOVCap:	Migration-, Interrupt Message Number: 000
		IOVCtl:	Enable- Migration- Interrupt- MSE- ARIHierarchy+
		IOVSta:	Migration-
		Initial VFs: 16, Total VFs: 16, Number of VFs: 0, Function Dependency Link: 00
		VF offset: 4, stride: 1, Device ID: 0000
		Supported Page Size: 00000573, System Page Size: 00000001
		Region 0: Memory at ee000000 (32-bit, non-prefetchable)
		Region 1: Memory at 000039bec0000000 (64-bit, prefetchable)
		Region 3: Memory at 000039bfd0000000 (64-bit, prefetchable)
		VF Migration: offset: 00000000, BIR: 0
	Capabilities: [c14 v1] Alternative Routing-ID Interpretation (ARI)
		ARICap:	MFVC- ACS-, Next Function: 1
		ARICtl:	MFVC- ACS-, Function Group: 0
	Kernel driver in use: nvidia
	Kernel modules: nvidia_drm, nvidia
```
In the context of PCIe (Peripheral Component Interconnect Express), "Speed 8GT/s" refers to the data transfer rate or speed of the PCIe interface. Let's break down what this means:

PCIe: PCIe is a high-speed serial computer expansion bus standard used for connecting various hardware components like graphics cards, network cards, storage controllers, and more to the motherboard of a computer. It provides a high-speed data transfer pathway between these components and the CPU.

Speed: The speed of a PCIe interface is typically measured in gigatransfers per second (GT/s). One GT/s is equivalent to one billion data transfers per second. This metric represents how quickly data can be sent and received between the components connected through PCIe.

8GT/s: When you see "Speed 8GT/s," it means that the PCIe interface is operating at a speed of 8 gigatransfers per second. This is a common speed setting for PCIe slots and is often referred to as PCIe Gen3 (third generation). PCIe Gen3 has a maximum theoretical bandwidth of approximately 8 gigabits per second per lane (in each direction, so 16 gigabits per second for a full-duplex connection).

Keep in mind that PCIe comes in different generations, and each generation offers different speeds. For example:

PCIe Gen1: Speed of 2.5GT/s
PCIe Gen2: Speed of 5GT/s
PCIe Gen3: Speed of 8GT/s
PCIe Gen4: Speed of 16GT/s
PCIe Gen5: Speed of 32GT/s
The speed of the PCIe interface is a crucial factor when considering the performance of components connected to it, such as graphics cards, storage devices, and expansion cards. It determines the maximum data transfer rate that can be achieved between these components and the rest of the computer system.

## Convert PCIe speed to bandwidth
To convert the PCIe speed of 8GT/s to bandwidth, you need to consider the data transfer rate per lane and the number of lanes in the PCIe connection.

PCIe Gen3 (8GT/s): This generation of PCIe has a data transfer rate of 8 gigatransfers per second per lane.

Bandwidth calculation: To calculate the bandwidth in gigabits per second (Gbps) for a single lane of PCIe Gen3, you can simply use the speed value:

Bandwidth per lane = Speed (GT/s) * 1 (lane) = 8 Gbps

Now, if you have a PCIe connection with multiple lanes, such as x4 (4 lanes) or x16 (16 lanes), you can calculate the total bandwidth by multiplying the bandwidth per lane by the number of lanes.

For example:

PCIe Gen3 x4: 8 Gbps (per lane) * 4 (lanes) = 32 Gbps
PCIe Gen3 x16: 8 Gbps (per lane) * 16 (lanes) = 128 Gbps
So, the bandwidth for a PCIe Gen3 connection depends on the number of lanes in the specific PCIe slot or connection. In the case of 8GT/s, the bandwidth per lane is 8 Gbps, and the total bandwidth depends on how many lanes are available in the specific PCIe configuration.

## Pinned Memory
Higher bandwidth between the host and the device is achieved when using page-locked (or pinned) memory.

Page-locked or pinned memory transfers attain the highest bandwidth between the host and the device. On PCIe x16 Gen3 cards, for example, pinned memory can attain roughly 12 GB/s transfer
rates.

Pinned memory is allocated using the cudaHostAlloc() functions in the Runtime API. The bandwidthTest CUDA Sample shows how to use these functions as well as how to measure memory transfer performance.

### Why pin memory?
- Pagable memory is transferred using the host CPU
- Pinned memory is transferred using the DMA engines
  - Frees the CPU for asynchronous execution
  - Achieves a higher percent of peak bandwidth

### ALLOCATING PINNED MEMORY
- cudaMallocHost(...) / cudaHostAlloc(...)
  - Allocate/Free pinned memory on the host
  - Replaces malloc/free/new
- cudaFreeHost(...)
  - Frees memory allocated by cudaMallocHost or cudaHostAlloc
- cudaHostRegister(...) / cudaHostUnregister(...)
  - Pins/Unpins pagable memory (making it pinned memory)
  - Slow so don't do often

## Asynchronous and Overlapping Transfers with Computation
- Data transfers between the host and the device using `cudaMemcpy()` are blocking transfers; that is, control is returned to the host thread only after the data transfer is complete.

- The `cudaMemcpyAsync()` function is a non-blocking variant of `cudaMemcpy()` in which control is returned immediately
to the host thread. In contrast with `cudaMemcpy()`, the asynchronous transfer version requires pinned host memory (If you want truly asynchronous behavior (e.g. overlap of copy and compute) then the memory must be pinned. If it is not pinned, there won’t be any runtime errors, but the copy will not be asynchronous - it will be performed like an ordinary `cudaMemcpy()`.) (see Pinned Memory), and it contains an additional argument, a stream ID.