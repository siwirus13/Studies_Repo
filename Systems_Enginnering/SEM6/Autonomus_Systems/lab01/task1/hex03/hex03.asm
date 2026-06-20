#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF
    
    ORG 0x0000
    goto START
    
    ORG 0x0005

START
    bcf STATUS, RP1
    bsf STATUS, RP0
    movlw 0x00
    movwf TRISA
    movlw 0xFF
    movwf TRISC
    bcf STATUS, RP0
    movlw 0x00
    movwf PORTA

    movlw 0x00
    movwf 0x20
LOOP
    btfsc PORTC, 0
    bsf 0x20, 0
    btfsc PORTC, 1
    bsf 0x20, 1
    btfsc PORTC, 2
    bsf 0x20, 2
    btfsc PORTC, 3
    bsf 0x20, 3
    movf 0x20, W
    movwf PORTA
    goto LOOP
    
    end
