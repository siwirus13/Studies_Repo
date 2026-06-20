#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF
    
    ORG 0x0000
    goto START
    
    ORG 0x0005

START
    bcf STATUS, RP1
    bsf STATUS, RP0

    movlw 0x06
    movwf ADCON1

    movlw 0x00
    movwf TRISA
    movlw 0xFF
    movwf TRISC
    
    bcf STATUS, RP0
    
    movlw 0x00
    movwf PORTA

LOOP
    btfss PORTC, 1
    goto CHECK_2
    movlw b'00000001'
    xorwf PORTA, f
    call DELAY
WAIT_1:
    btfsc PORTC, 1
    goto WAIT_1
    call DELAY

CHECK_2:
    btfss PORTC, 2
    goto CHECK_3
    movlw b'00000010'
    xorwf PORTA, f
    call DELAY
WAIT_2:
    btfsc PORTC, 2
    goto WAIT_2
    call DELAY

CHECK_3:
    btfss PORTC, 3
    goto CHECK_4
    movlw b'00000100'
    xorwf PORTA, f
    call DELAY
WAIT_3:
    btfsc PORTC, 3
    goto WAIT_3
    call DELAY

CHECK_4:
    btfss PORTC, 4
    goto CHECK_5
    movlw b'00001000'
    xorwf PORTA, f
    call DELAY
WAIT_4:
    btfsc PORTC, 4
    goto WAIT_4
    call DELAY

CHECK_5:
    btfss PORTC, 5
    goto CHECK_6
    movlw b'00010000'
    xorwf PORTA, f
    call DELAY
WAIT_5:
    btfsc PORTC, 5
    goto WAIT_5
    call DELAY

CHECK_6:
    btfss PORTC, 6
    goto END_LOOP
    movlw b'00100000'
    xorwf PORTA, f
    call DELAY
WAIT_6:
    btfsc PORTC, 6
    goto WAIT_6
    call DELAY

END_LOOP:
    goto LOOP

DELAY:
    movlw d'100'
    movwf 0x20
DELAY_OUTER:
    movlw d'200'
    movwf 0x21
DELAY_INNER:
    decfsz 0x21, f
    goto DELAY_INNER
    decfsz 0x20, f
    goto DELAY_OUTER
    return
    
    end
