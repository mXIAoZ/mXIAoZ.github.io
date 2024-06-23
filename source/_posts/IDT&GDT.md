---
title: 中断设置和键鼠处理
date: 2024/06/23
categories:
- 30daysOS
- CS
tags: [CS,30daysOS]
mathjax: true
---

## VGA 设定

首先简单看待调色板设定的相关程序，具体可以参考下面的链接：

```c
void set_palette(int start, int end, unsigned char *rgb) {
  int eflags = io_load_eflags(); // 记录标志

  io_cli(); // 禁止中断

  io_out8(0x03c8, start); // vga 设定，参考：https://wiki.osdev.org/VGA_Hardware
  for (int i = start; i <= end; i++) {
    io_out8(0x03c9, rgb[0] / 4);
    io_out8(0x03c9, rgb[1] / 4);
    io_out8(0x03c9, rgb[2] / 4);
    rgb += 3;
  }

  io_store_eflags(eflags);
}
```

## GDT&IDT

### GDT

"global segment descriptor table"，全局段号记录表。将段号记录在内存的某个地方，然后将内存的起始地址和有效设定个数放在CPU的GDTR的（global segment descriptor table register）特殊寄存器中。段寄存器是16位，但是由于cpu设定原因，低3位不可用，所以真正可以使用的位数为13位，即可以分为8192个段。

<!--more-->

```c
  #define ADR_GDT 0x00270000
  #define LIMIT_GDT 0x0000ffff

  #define ADR_BOOTPACK 0x00280000
  #define LIMIT_BOOTPACK 0x0007ffff

  #define AR_DATA32_RW 0x4092
  #define AR_CODE32_ER 0x409a
  
  struct SegmentDescriptor *gdt = (struct SegmentDescriptor *)ADR_GDT;
  
  struct SegmentDescriptor {
    short limit_low, base_low;
    char base_mid, access_right;
    char limit_high, base_high;
	};
  

  for (int i = 0; i <= LIMIT_GDT / 8; i++) {
    set_segmdesc(gdt + i, 0, 0, 0);
  }

  set_segmdesc(gdt + 1, 0xffffffff, 0x00000000, AR_DATA32_RW);
  set_segmdesc(gdt + 2, LIMIT_BOOTPACK, ADR_BOOTPACK, AR_CODE32_ER);
  load_gdtr(LIMIT_GDT, ADR_GDT);
```

这里设定了两个段，第一个段从0x00000000到0xffffffff，设定了cpu能管理内存的大小。第二个段从0x00270000到0x0007ffff共512KB，

函数的最后一个变量代表段属性（access right），因为12位段属性的高4放在limit high中，所以程序里有意把ar当作如下16位处理：

“xxxx0000xxxxxxx”。

ar的高4位称为扩展访问权，由“GD00”构成，G值得是Gbit，D指段模式，为1表示32位模式，为0表示16位模式。

ar低8位：

00000000（0x00）：未使用的记录表

10010010（0x92）：系统专用，可读写，不可执行

10011010（0x9a）：系统专用，可执行，可读不可写

11110010（0xf2）：应用程序用，可读写，不可执行

11111010（0xfa）：应用程序用，可执行，可读不可写

CPU处于系统模式还是应用模式，取决于执行中的应用程序是位于访问权为0x9a的段，还是位于访问权为0xfa的段。

### IDT

"interrput descriptor table "，中断记录表。IDT记录了0-255的中断号码与调用函数的关系。

``` 
  #define ADR_IDT 0x0026f800
  #define LIMIT_IDT 0x000007ff
  
	#define AR_INTGATE32 0x008e
  
  struct GateDescriptor *idt = (struct GateDescriptor *)ADR_IDT;
  
  struct GateDescriptor {
    short offset_low, selector;
    char dw_count, access_right;
    short offset_high;
	};
  
  for (int i = 0; i <= LIMIT_IDT / 8; i++) {
    set_gatedesc(idt + i, 0, 0, 0);
  }
  load_idtr(LIMIT_IDT, ADR_IDT);

  set_gatedesc(idt + 0x21, (int)asm_int_handler21, 2 * 8, AR_INTGATE32);
  set_gatedesc(idt + 0x27, (int)asm_int_handler27, 2 * 8, AR_INTGATE32);
  set_gatedesc(idt + 0x2c, (int)asm_int_handler2c, 2 * 8, AR_INTGATE32);
```

### PIC

 "programmable interrupt controller"，可编程中断控制器。在设计上CPU只能处理一个中断，PIC是连接着CPU的芯片，它将8个中断信号（interrupt request）集合成一个中断信号。当有一个中断信号进来，就将唯一的输出管脚设置为ON，并且通知CPU。一般有两个PIC，从PIC通过2号IRQ与主PIC相连（硬件设计）。

#### PIC寄存器

PIC的寄存器都是8位的。

IMR："interrupt mask register"，中断屏蔽寄存器，如果某一位的值是1，则该位对应的IRQ信号被屏蔽。

ICW："initial control word"，初始化控制数据。ICW共有4个，即存储了4个字节的数据。ICW1和ICW4 和 PIC与主板的配线方式、中断信号的电气信号有关，因此这里的电脑设定程序是一个固定值。ICW3是有关主-从连接设定，因为硬件已经不可更改，所以软件设定也是一致的（00000100）。

* 我们可以编程的中断信号只有ICW2，ICW2决定了IRQ以哪一号中断通知CPU。

*注：中断发生之后，CPU命令PIC发送两个字节的数据，PIC利用信号线发送 “0xcd 0x？”的两个数据，从CPU看来，就是执行“INT 0x？”的指令。*

我们可以设定的中断为“0x20 - 0x2f”，16个中断信号：IRQ0-16，由上述PIC发送。其中“0x00 - 0x1f”是CPU内部中断，不可以和外部中断冲突。

```c
void init_pic(void) {
  // 禁止所有中断
  io_out8(PIC0_IMR, 0xff);
  io_out8(PIC1_IMR, 0xff);

  io_out8(PIC0_ICW1, 0x11); // 边缘触发模式
  io_out8(PIC0_ICW2, 0x20); // IRQ0-7由INT20-27接收
  io_out8(PIC0_ICW3, 1 << 2); // PIC1由IRQ2连接
  io_out8(PIC0_ICW4, 0x01); // 无缓冲区模式

  io_out8(PIC1_ICW1, 0x11); // 边缘触发模式
  io_out8(PIC1_ICW2, 0x28); // IRQ8-15由INT28-2f接收
  io_out8(PIC1_ICW3, 2); // PIC1由IRQ2连接
  io_out8(PIC1_ICW4, 0x01); // 无缓冲区模式

  io_out8(PIC0_IMR, 0xfb); // PIC1以外中断全部禁止
  io_out8(PIC1_IMR, 0xff); // 禁止全部中断
}
```

## 中断处理程序

鼠标处理为例，鼠标中断处理程序：

```c
void int_handler2c(int *esp) {
  io_out8(PIC1_OCW2, 0x64); // 通知PIC1 IRQ-12的受理已经完成
  io_out8(PIC0_OCW2, 0x62); // 通知PIC0 IRQ-02的受理已经完成
  unsigned char data = io_in8(PORT_KEYDAT);
  
  fifo8_put(&mousefifo, data);
}
```

IRQ-12是从PIC的4号，因为主/从PIC的协调不能自动完成，因此需要首先通知从PIC，再通知主PIC完成鼠标的中断处理。

这里使用一个队列来处理不同的中断，因为中断通知完成之后，会处理屏幕信息比较耗时，如果一直等待处理完成再打开中断的话会导致系统不连贯，以及不能从网上接受数据的情况。

中断处理完成之后必须使用IRETD指令，所以需要使用汇编：

```
asm_int_handler2c:
  PUSH    ES
  PUSH    DS
  PUSHAD
  MOV     EAX, ESP
  PUSH    EAX
  MOV     AX, SS
  MOV     DS, AX
  MOV     ES, AX
  CALL    int_handler2c
  POP     EAX
  POPAD
  POP     DS
  POP     ES
  IRETD
```

## 键/鼠初始化

### 键盘初始化：

```c
#define PORT_KEYDAT 0x0060
#define PORT_KEYSTA 0x0064
#define PORT_KEYCMD 0x0064

#define KEYSTA_SEND_NOTREADY 0x02
#define KEYCMD_WRITE_MODE 0x60
#define KBC_MODE 0x47

void wait_KBC_sendready(void) {
  for (;;) {
    if ((io_in8(PORT_KEYSTA) & KEYSTA_SEND_NOTREADY) == 0) {
      break;
    }
  }
}

void init_keyboard(void) {
  wait_KBC_sendready();
  io_out8(PORT_KEYCMD, KEYCMD_WRITE_MODE);
  wait_KBC_sendready();
  io_out8(PORT_KEYDAT, KBC_MODE);
}
```

在初始化程序中，一边确认是否可以往键盘控制电路发送信息，一边发送模式设定指令。模式设定的指令是0x60，利用鼠标模式的模式号码是0x47.

### 键盘初始化：

```
#define KEYCMD_SENDTO_MOUSE 0xd4
#define MOUSECMD_ENABLE 0xf4

void enable_mouse(struct FIFO32 *fifo, int data0, struct MouseDec *mdec) {
  mousefifo = fifo;
  mousedata0 = data0;

  wait_KBC_sendready();
  io_out8(PORT_KEYCMD, KEYCMD_SENDTO_MOUSE);
  wait_KBC_sendready();
  io_out8(PORT_KEYDAT, MOUSECMD_ENABLE);

  mdec->phase = 0;
}
```

发送鼠标激活指令，归根结底还是要向键盘控制器发送指令。

如果往键盘控制电路发送指令0xd4，下一个数据会自动发送给鼠标。

收到鼠标激活指令后，会给CPU发送0xfa中断，因此最后一行为了区分第一个鼠标中断信号和后续的鼠标中断，在接收到0xfa后，鼠标的phase就不会再是0。

