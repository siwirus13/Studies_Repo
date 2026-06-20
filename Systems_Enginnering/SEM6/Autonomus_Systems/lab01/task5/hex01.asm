#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF
    
    ORG 0x0000
    goto START
    
    ORG 0x0005

START
    bcf STATUS, RP1
    bsf STATUS, RP0
    
    clrf TRISB
    movlw 0xFF
    movwf TRISC
    
    bcf STATUS, RP0
    
    clrf PORTB
    
    movlw d'50'
    movwf 0x30

MAIN_LOOP
    comf PORTB, F
    
    movf 0x30, W
    movwf 0x31

DELAY_CYCLE:
    call DELAY_20MS
    call CHECK_BUTTONS
    
    movf 0x31, W
    btfsc STATUS, Z
    goto MAIN_LOOP
    
    decfsz 0x31, F
    goto DELAY_CYCLE
    
    goto MAIN_LOOP

CHECK_BUTTONS:
CHECK_RC0:
    btfss PORTC, 6
    goto CHECK_RC1
    
    movlw d'250'
    subwf 0x30, W
    btfsc STATUS, C
    goto WAIT_RC0
    
    movlw d'5'
    addwf 0x30, F
    
WAIT_RC0:
    call DELAY_20MS
    btfsc PORTC, 6
    goto WAIT_RC0
    goto CHECK_END

CHECK_RC1:
    btfss PORTC, 7
    goto CHECK_END
    
    movlw d'55'
    subwf 0x30, W
    btfss STATUS, C
    goto WAIT_RC1
    
    movlw d'5'
    subwf 0x30, F
    
WAIT_RC1:
    call DELAY_20MS
    btfsc PORTC, 7
    goto WAIT_RC1

CHECK_END:
    return

DELAY_20MS:
    movlw d'100'
    movwf 0x25
DELAY_OUTER:
    movlw d'66'
    movwf 0x26
DELAY_INNER:
    decfsz 0x26, F
    goto DELAY_INNER
    decfsz 0x25, F
    goto DELAY_OUTER
    return
    
    end
