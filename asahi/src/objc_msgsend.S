// objc_msgSend for AAPCS64 (Asahi Linux). x0 = receiver, x1 = selector, the
// rest of the arguments in x2-x7 / v0-v7 / stack, x8 = indirect result.
// Looks up the IMP with every argument register saved, then tail-calls it so
// the method sees the caller's arguments and returns straight to the caller.
// A nil receiver returns zero in x0/x1 and v0-v3 (Objective-C's nil rule).
    .text
    .p2align 4
    .globl objc_msgSend
    .type objc_msgSend, %function
objc_msgSend:
    cbz     x0, 1f
    stp     x29, x30, [sp, #-224]!
    mov     x29, sp
    stp     x0, x1, [sp, #16]
    stp     x2, x3, [sp, #32]
    stp     x4, x5, [sp, #48]
    stp     x6, x7, [sp, #64]
    str     x8, [sp, #80]
    stp     q0, q1, [sp, #96]
    stp     q2, q3, [sp, #128]
    stp     q4, q5, [sp, #160]
    stp     q6, q7, [sp, #192]
    bl      metal_asahi_objc_lookup
    mov     x16, x0
    ldp     x0, x1, [sp, #16]
    ldp     x2, x3, [sp, #32]
    ldp     x4, x5, [sp, #48]
    ldp     x6, x7, [sp, #64]
    ldr     x8, [sp, #80]
    ldp     q0, q1, [sp, #96]
    ldp     q2, q3, [sp, #128]
    ldp     q4, q5, [sp, #160]
    ldp     q6, q7, [sp, #192]
    ldp     x29, x30, [sp], #224
    br      x16
1:  mov     x1, #0
    movi    d0, #0
    movi    d1, #0
    movi    d2, #0
    movi    d3, #0
    ret
    .size objc_msgSend, .-objc_msgSend

// On AAPCS64 there is no separate struct-return or float-return entry; these
// exist only so metal-cpp's x86 branches link. They behave as objc_msgSend.
    .globl objc_msgSend_stret
    .type objc_msgSend_stret, %function
objc_msgSend_stret:
    b       objc_msgSend
    .size objc_msgSend_stret, .-objc_msgSend_stret
    .globl objc_msgSend_fpret
    .type objc_msgSend_fpret, %function
objc_msgSend_fpret:
    b       objc_msgSend
    .size objc_msgSend_fpret, .-objc_msgSend_fpret
    .section .note.GNU-stack,"",%progbits
