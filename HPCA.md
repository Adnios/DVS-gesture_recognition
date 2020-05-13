## NVDIMM-C: A Byte-Addressable Non-Volatile Memory Module for Compatibility with Standard DDR Memory Interfaces

>Samsung Electronics

### Knowledge

- 非易失性双列直插式内存模块（英语：non-volatile dual in-line memory module，缩写NVDIMM）是一种用于计算机的随机存取存储器。非易失性存储器是即使断电也能保留其内容的内存，这包括意外断电、系统崩溃或正常关机。双列直插式表示该内存使用DIMM封装。NVDIMM在某些情况下可以改善应用程序的性能、数据安全性和系统崩溃恢复时间。这增强了固态硬盘（SSD）的耐用性和可靠性。DIMM全称Dual-Inline-Memory-Modules，中文名叫双列直插式存储模块，是指奔腾CPU推出后出现的新型内存条，它提供了64位的数据通道。
- 2 representative NVDIMM: a proprietary Inter DDR-T and JEDEC NVDIMM-P
    - 采用新平台成本高昂,衡量其迁移到新版本的效率平台要复杂得多
    - 查找可以由所有现有系统支持的新存储设备
- NVM(非易失性存储器)
    - performance, density with dynamic random-access memory(DRAM)
    - byte-addressable(因此可通过加载/存储进行访问CPU的指令)
- inspired by backward compatibility, to build a non-volatile DIMM (NVDIMM) that supports not only existing systems but also the next-generation systems. (NVDIMM-C，一种用于兼容性的NVDIMM架构具有标准DDR4内存接口，无任何修改硬件和软件环境)
- NVDIMM-C: an NVDIMM architecture for the Compatibility with the standard DDR4 memory interfaces without any modification to the hardware and software environments
- proof-of-concept (PoC) 概念验证

### contributions

- 我们介绍了一种新的可字节寻址的存储器架构，该架构可满足维护向后的目标与现有系统的兼容性。
- 我们描述了一种允许多种主芯片（例如主机CPU和设备的媒体控制器）共享同一条内存总线，而无需修改DDR4接口和协议。
- 我们提出建议的详细实施包括硬件和软件堆栈的体系结构。
- 我们在真实的服务器系统上评估建议的设备无需对运行环境进行任何修改.我们发现NVDIMM-C架构可以提供低延迟NVM设备实现均衡的性能

### DRAM

动态随机存取存储器（Dynamic Random Access Memory，DRAM）是一种半导体存储器，主要的作用原理是利用电容内存储电荷的多寡来代表一个二进制比特（bit）是1还是0。由于在现实中晶体管会有漏电电流的现象，导致电容上所存储的电荷数量并不足以正确的判别数据，而导致数据毁损。因此对于DRAM来说，周期性地充电是一个无可避免的要件。由于这种需要定时刷新的特性，因此被称为“动态”存储器。相对来说，静态存储器（SRAM）只要存入数据后，纵使不刷新也不会丢失记忆。

- 直接访问（DAX）是NVM设备的一种机制
- 任何在刷新操作期间，对DRAM的请求无效。此受限时间段称为tRFC时间。
- tRFC和tREFI(平均刷新间隔)都可以通过OS内核编程

### Architecture

![2 design options](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200504225314.png "opt title")

- 首先，NVMC可以放置在如图1a所示的存储器模块前端;因此，它直接连接到DDR4内存总线。Obviously, this design option is available only if the latency of NVM media is competitive with DRAM or if both the iMC and NVMCs support an asynchronous handshake protocol (e.g., JEDEC NVDIMM-P and Intel DDR-T)
- 选择了b, 后端NVMC和NVM媒体是完全隐藏的，因此NVMC和NVM的操作和时间与同步域无关 。 因此，无论主机iMC的时间约束如何，任何类型的NVM技术都可以用作后端媒体。
-  使用NAND闪存作为NVM媒体，因为它的高密度和低成本的特点，以及它建立良好的内存技术。
-  主机iMC无法访问NVM媒体，因为NVMC和NVM媒体与DDR4内存总线物理分离。 直观地说，DRAM作为前端架构可以使用DRAM作为缓存或 NVM介质的缓冲区。

总线争用

1.  根据对DRAM缓存的访问是否被击中或丢失，NVMC可以访问DRAM缓存。 因此，NVMC除了NAND控制器外，还必须包括DDR4控制器。
2.  在计算机设计中，多个设备同时访问单个共享信道的情况大多导致总线争用。 两者共享的DRAM缓存的总线(即iMC和NVMC)是这项工作中最严重的挑战。
3.  图2a显示了出现总线争用的几种情况。 在情况1(C1)中，NVMC为DRAM行发出一个激活命令，该命令不同于先前由主机iMC激活的行。  当时，iMC可能向共享内存总线发出Read命令，从而导致内存命令的冲突。 这种命令冲突可能随时发生，即使NVMC可以 通过CA总线处理IMC行为。 对于C2，假设两个主程序正在访问DRAM的同一行。 NVMC试图在打开的行上执行突发读取. 然而，iMC可以拥有  通过发出预充电来关闭（停用）该行，因此，随后的Read命令将无效，并且还可能导致意外状态或关键内存错误。
4.  在本文中，我们提出了一种无碰撞机制的DRAM-作为前端NVDIMM体系结构  ，NVMC等待直到刷新命令由主机iMC发出。 当检测到时，NVMC首先等待默认的tR FC时间，然后在额外的tR FC时间内访问DRAM。


### POC

在本工作中，我们实现了所提出的DRAM前端体系结构作为概念证明(PoC)设备。 图3显示了我们的PoC设备的印刷电路板(PCB)布局。

## Writing

![pic alt](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200505211004.png "opt title")

- 一步一步展开

![pic alt](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200505211555-writing.png "opt title")

![pic alt](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200505211644-writing.png "opt title")

- Discussion

![pic alt](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200505213054-writing.png "opt title")

- Introduction 四段论


## Q-Zilla: A Scheduling Framework and Core Microarchitecture for Tail-tolerant Microservices

https://www.youtube.com/watch?v=pdC-SzwsaA8


