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
    
    clrf 0x30
    clrf 0x31
    clrf 0x32
    clrf 0x33
    clrf 0x34
    clrf 0x35
    clrf 0x36
    clrf 0x37

LOOP
    call DELAY_20MS

CHECK_0:
    btfss PORTC, 0
    goto DECREMENT_0
    movlw d'250'
    movwf 0x30
    bsf PORTB, 0
    goto CHECK_1

DECREMENT_0:
    movf 0x30, W
    btfsc STATUS, Z
    goto CHECK_1
    decfsz 0x30, F
    goto CHECK_1
    bcf PORTB, 0

CHECK_1:
    btfss PORTC, 1
    goto DECREMENT_1
    movlw d'250'
    movwf 0x31
    bsf PORTB, 1
    goto CHECK_2

DECREMENT_1:
    movf 0x31, W
    btfsc STATUS, Z
    goto CHECK_2
    decfsz 0x31, F
    goto CHECK_2
    bcf PORTB, 1

CHECK_2:
    btfss PORTC, 2
    goto DECREMENT_2
    movlw d'250'
    movwf 0x32
    bsf PORTB, 2
    goto CHECK_3

DECREMENT_2:
    movf 0x32, W
    btfsc STATUS, Z
    goto CHECK_3
    decfsz 0x32, F
    goto CHECK_3
    bcf PORTB, 2

CHECK_3:
    btfss PORTC, 3
    goto DECREMENT_3
    movlw d'250'
    movwf 0x33
    bsf PORTB, 3
    goto CHECK_4

DECREMENT_3:
    movf 0x33, W
    btfsc STATUS, Z
    goto CHECK_4
    decfsz 0x33, F
    goto CHECK_4
    bcf PORTB, 3

CHECK_4:
    btfss PORTC, 4
    goto DECREMENT_4
    movlw d'250'
    movwf 0x34
    bsf PORTB, 4
    goto CHECK_5

DECREMENT_4:
    movf 0x34, W
    btfsc STATUS, Z
    goto CHECK_5
    decfsz 0x34, F
    goto CHECK_5
    bcf PORTB, 4

CHECK_5:
    btfss PORTC, 5
    goto DECREMENT_5
    movlw d'250'
    movwf 0x35
    bsf PORTB, 5
    goto CHECK_6

DECREMENT_5:
    movf 0x35, W
    btfsc STATUS, Z
    goto CHECK_6
    decfsz 0x35, F
    goto CHECK_6
    bcf PORTB, 5

CHECK_6:
    btfss PORTC, 6
    goto DECREMENT_6
    movlw d'250'
    movwf 0x36
    bsf PORTB, 6
    goto CHECK_7

DECREMENT_6:
    movf 0x36, W
    btfsc STATUS, Z
    goto CHECK_7
    decfsz 0x36, F
    goto CHECK_7
    bcf PORTB, 6

CHECK_7:
    btfss PORTC, 7
    goto DECREMENT_7
    movlw d'250'
    movwf 0x37
    bsf PORTB, 7
    goto END_CYCLE

DECREMENT_7:
    movf 0x37, W
    btfsc STATUS, Z
    goto END_CYCLE
    decfsz 0x37, F
    goto END_CYCLE
    bcf PORTB, 7

END_CYCLE:
    goto LOOP

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
