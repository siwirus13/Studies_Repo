#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF
    
    ORG 0x0000
    goto START
    
    ORG 0x0005

START
    bcf STATUS, RP1
    bsf STATUS, RP0
    clrf TRISB
    
    bcf STATUS, RP0
    
    movlw 0xFF
    movwf 0x28
    movwf PORTB
    
    movlw d'1'
    movwf 0x29

MAIN_LOOP
    movf 0x29, W
    movwf 0x2A
    
WAIT_LOOP:
    call DELAY_1S
    decfsz 0x2A, F
    goto WAIT_LOOP
    
    bcf STATUS, C
    rlf 0x28, F
    
    movf 0x28, W
    movwf PORTB
    
    movf 0x28, F
    btfsc STATUS, Z
    goto END_PROG
    
    incf 0x29, F
    goto MAIN_LOOP

END_PROG:
    goto $

DELAY_1S:
    movlw d'6'
    movwf 0x25
DELAY_OUTER:
    movlw d'255'
    movwf 0x26
DELAY_INNER:
    movlw d'255'
    movwf 0x27
DELAY_CORE:
    decfsz 0x27, F
    goto DELAY_CORE
    decfsz 0x26, F
    goto DELAY_INNER
    decfsz 0x25, F
    goto DELAY_OUTER
    return
    
    end
