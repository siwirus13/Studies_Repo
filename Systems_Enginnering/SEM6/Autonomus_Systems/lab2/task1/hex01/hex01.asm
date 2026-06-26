#include <p16f877a.inc>
    errorlevel -302
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
    goto $
    
    end
