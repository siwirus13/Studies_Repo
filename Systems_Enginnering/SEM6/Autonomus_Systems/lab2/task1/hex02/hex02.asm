#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF
    
    ORG 0x0000
    goto START
    
    ORG 0x0005

START
    bcf STATUS, RP1
    bsf STATUS, RP0
    clrf TRISB
LEDON
    bcf STATUS, RP1
    bcf STATUS, RP0
    movlw 0xFF
    movwf PORTB
    
    bcf STATUS, RP0
    bcf STATUS, RP1
    movlw 0xFF
    movwf 0x25
LOOP1
    movlw 0xFF
    movwf 0x26
    decf 0x25, F
    btfsc STATUS,Z
    goto LEDOFF
LOOP2
    decf 0x26, F
    btfsc STATUS,Z
    goto LOOP1
    goto LOOP2

LEDOFF
    bcf STATUS, RP1
    bcf STATUS, RP0
    movlw PORTB
    movwf FSR
    bcf STATUS, IRP
    movlw 0x00
    movwf INDF
    goto $
    
    end
