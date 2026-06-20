; SAPER 5x5 dla PIC16F877A + ILI9341 SPI
;
; Rejestry:
;   0x20-0x23 : flagi (25 bitow, 4 bajty)
;   0x24-0x27 : miny
;   0x28-0x2B : odkryte
;   0x50/0x51 : X/Y biezacej pozycji (kursor lub rysowanie)
;   0x54/0x55 : X/Y nowej pozycji kursora
;   0x56/0x57 : indeks liniowy / licznik petli
;   0x58      : maska bitu
;   0x59      : adres bazowy warstwy
;   0x60-0x63 : x0,x1,y0,y1 okna LCD
;   0x64/0x65 : kolor RGB565 (H/L)
;   0x66/0x67 : liczniki petli rysowania
;   0x70/0x71 : liczniki opoznienia
;   0x72-0x74 : liczniki fill ekranu
;   0x75/0x76 : tymczasowe
;   0x78/0x79 : zachowanie X/Y w CHECK_MINE_AT
;   0x7A/0x7B : wspolrzedne sasiada
;   0x7C      : wynik COUNT_MINES
;   0x7D/0x7E : DY/DX w COUNT_MINES
;   0x7F      : wynik CHECK_MINE_AT
;   0x41      : stan gry (0=trwa, 1=wygrana, 2=przegrana)
;   0x42      : licznik odkrytych
;   0x43      : liczba min
;   0x44      : licznik petli sadzenia min
;   0x45      : losowy indeks kandydata
;
; Legenda kolorow odkrytych pol:
;   0 sas. = czarny  (0x0000)
;   1 sas. = zielony (0x07E0)
;   2 sas. = niebieski (0x001F)
;   3 sas. = pomaranczowy (0xFC00)
;   4 sas. = czerwony (0xF800)
;   5 sas. = rozowy (0xF81F)
;   6 sas. = cyan (0x07FF)
;   7 sas. = fioletowy (0x781F)
;   8 sas. = brazowy (0x8400)
;   flaga = zolty (0xFFE0)
;   kursor = bialy (0xFFFF)
;   zaslepka = szary (0x8410)
;
; Przyciski: RB0=UP RB1=LEFT RB2=DOWN RB3=RIGHT RB4=FLAGA RB6=ODKRYJ RB7=RESTART
;
; Kafelek 22x22px, stride 24:
;   x0 = X*24+60, y0 = Y*24+80

#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF

    ORG 0x0000
    goto START
    ORG 0x0005

START:
    ; Bank1: konfiguracja portow i SPI
    bcf   STATUS, RP1
    bsf   STATUS, RP0
    movlw 0x06
    movwf ADCON1
    movlw 0xFF
    movwf TRISB
    clrf  TRISD
    bcf   TRISC, 0
    bcf   TRISC, 1
    bcf   TRISC, 2
    bcf   TRISC, 3
    bcf   TRISC, 5
    movlw b'01000000'
    movwf SSPSTAT
    ; TMR0: wolny bieg jako zrodlo losowosci
    movlw b'00000111'       ; prescaler 1:256, wewnetrzny zegar
    movwf OPTION_REG
    bcf   STATUS, RP0

    ; Bank0: SPI master Fosc/4
    movlw b'00100000'
    movwf SSPCON

    ; Init LCD
    bsf   PORTC, 2
    bsf   PORTC, 0
    call  DELAY_20MS
    bcf   PORTC, 0
    call  DELAY_20MS
    bsf   PORTC, 0
    call  DELAY_20MS
    movlw 0x01
    call  SEND_CMD
    call  DELAY_20MS
    movlw 0x11
    call  SEND_CMD
    call  DELAY_20MS
    movlw 0x3A
    call  SEND_CMD
    movlw 0x55
    call  SEND_DAT
    movlw 0x29
    call  SEND_CMD

    ; Wypelnij czarnym (240x320)
    movlw 0x2A
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movlw 0xEF
    call  SEND_DAT
    movlw 0x2B
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    call  SEND_DAT
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movlw 0x01
    call  SEND_DAT
    movlw 0x3F
    call  SEND_DAT
    movlw 0x2C
    call  SEND_CMD
    bsf   PORTC, 1
    bcf   PORTC, 2
    movlw d'240'
    movwf 0x72
FILL_Y:
    movlw d'4'
    movwf 0x73
FILL_XO:
    movlw d'80'
    movwf 0x74
FILL_XI:
    bcf   PIR1, 3
    clrw
    movwf SSPBUF
WAIT_H:
    btfss PIR1, 3
    goto  WAIT_H
    bcf   PIR1, 3
    clrw
    movwf SSPBUF
WAIT_L:
    btfss PIR1, 3
    goto  WAIT_L
    decfsz 0x74, F
    goto  FILL_XI
    decfsz 0x73, F
    goto  FILL_XO
    decfsz 0x72, F
    goto  FILL_Y
    bsf   PORTC, 2

GAME_RESET:
    ; Wyczysc warstwy gry
    movlw 0x20
    movwf FSR
CLR_LOOP:
    clrf  INDF
    incf  FSR, F
    movlw 0x2C
    subwf FSR, W
    btfss STATUS, Z
    goto  CLR_LOOP

    clrf  0x41
    clrf  0x42
    movlw d'7'
    movwf 0x43

    call  PLANT_MINES
    call  DRAW_GRID

    ; Kursor startowy (2,2)
    movlw d'2'
    movwf 0x50
    movwf 0x51
    movwf 0x54
    movwf 0x55
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE

MAIN_LOOP:
    movf  0x41, F
    btfss STATUS, Z
    goto  $             ; gra skonczona - stoj
    call  CHECK_BUTTONS
    call  UPDATE_CURSOR
    goto  MAIN_LOOP

; =============================================================================
; PLANT_MINES - losuje 7 min uzywajac TMR0
; Algorytm: czytaj TMR0, mod 25 -> indeks kandydata
;           jesli juz mina, szukaj nastepnego wolnego (liniowo)
; =============================================================================
PLANT_MINES:
    movlw 0x24
    movwf 0x59
    movlw d'7'
    movwf 0x44          ; licznik min do posadzenia
PM_NEXT_MINE:
    ; Pobierz TMR0 jako baze losowosci
    movf  TMR0, W
    movwf 0x45          ; kandydat = TMR0
    ; mod 25: odejmuj 25 dopoki >= 25
PM_MOD25:
    movlw d'25'
    subwf 0x45, W       ; W = kandydat - 25
    btfss STATUS, C
    goto  PM_MOD_DONE   ; kandydat < 25
    movwf 0x45          ; kandydat = kandydat - 25
    goto  PM_MOD25
PM_MOD_DONE:
    ; Przelicz indeks na X/Y: X = idx%5, Y = idx/5
    movf  0x45, W
    movwf 0x56          ; 0x56 = idx
    clrf  0x51          ; Y = 0
PM_DIV5:
    movlw d'5'
    subwf 0x56, W
    btfss STATUS, C
    goto  PM_DIV_DONE   ; 0x56 < 5
    movwf 0x56
    incf  0x51, F       ; Y++
    goto  PM_DIV5
PM_DIV_DONE:
    movf  0x56, W
    movwf 0x50          ; X = pozostalosc
    ; Sprawdz czy to juz mina
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  PM_FIND_FREE  ; juz mina - znajdz wolne
    ; Posadz mine
    movf  0x58, W
    iorwf INDF, F
    ; Czekaj chwile na zmiane TMR0
    call  DELAY_20MS
    decfsz 0x44, F
    goto  PM_NEXT_MINE
    return

PM_FIND_FREE:
    ; Szukaj nastepnego wolnego pola liniowo od kandydata+1
    incf  0x45, F
    movlw d'25'
    subwf 0x45, W
    btfsc STATUS, C
    clrf  0x45          ; wraparound do 0
    goto  PM_MOD_DONE   ; przelicz X/Y i sprawdz ponownie

; =============================================================================
; GET_BITMASK_AND_PTR
; In: 0x50=X, 0x51=Y, 0x59=baza warstwy
; Out: FSR->bajt, 0x58=maska
; index = Y*5 + X
; =============================================================================
GET_BITMASK_AND_PTR:
    movf  0x51, W
    movwf 0x57
    movf  0x50, W
    movwf 0x56
    movf  0x57, F
    btfsc STATUS, Z
    goto  GTB_OFF
GTB_MUL:
    movlw d'5'
    addwf 0x56, F
    decfsz 0x57, F
    goto  GTB_MUL
GTB_OFF:
    movf  0x56, W
    movwf 0x57
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    movlw 0x07
    andwf 0x57, F
    movf  0x59, W
    addwf 0x57, W
    movwf FSR
    movlw 0x07
    andwf 0x56, W
    movwf 0x57
    movlw 0x01
    movwf 0x58
    movf  0x57, F
    btfsc STATUS, Z
    return
GTB_SHL:
    bcf   STATUS, C
    rlf   0x58, F
    decfsz 0x57, F
    goto  GTB_SHL
    return

; =============================================================================
; COUNT_MINES - liczy miny w 8 sasiadach (0x50=X, 0x51=Y)
; Out: W i 0x7C = liczba min
; =============================================================================
COUNT_MINES:
    clrf  0x7C
    movlw 0xFF
    movwf 0x7D          ; DY = -1
CM_Y:
    movlw 0xFF
    movwf 0x7E          ; DX = -1
CM_X:
    ; Pomin (0,0)
    movf  0x7E, F
    btfss STATUS, Z
    goto  CM_CHK
    movf  0x7D, F
    btfss STATUS, Z
    goto  CM_CHK
    goto  CM_NX
CM_CHK:
    ; sX = X+DX, sprawdz granice
    movf  0x50, W
    addwf 0x7E, W
    movwf 0x7A
    movlw d'5'
    subwf 0x7A, W
    btfsc STATUS, C
    goto  CM_NX
    ; sY = Y+DY
    movf  0x51, W
    addwf 0x7D, W
    movwf 0x7B
    movlw d'5'
    subwf 0x7B, W
    btfsc STATUS, C
    goto  CM_NX
    call  CHECK_MINE_AT
    addwf 0x7C, F
CM_NX:
    incf  0x7E, F
    movlw 0x02
    subwf 0x7E, W
    btfss STATUS, Z
    goto  CM_X
    incf  0x7D, F
    movlw 0x02
    subwf 0x7D, W
    btfss STATUS, Z
    goto  CM_Y
    movf  0x7C, W
    return

; =============================================================================
; CHECK_MINE_AT - sprawdza mine na (0x7A, 0x7B), zachowuje 0x50/0x51
; Out: W=1 jesli mina
; =============================================================================
CHECK_MINE_AT:
    movf  0x50, W
    movwf 0x78
    movf  0x51, W
    movwf 0x79
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    movlw 0x24
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    movlw 0x00
    btfss STATUS, Z
    movlw 0x01
    movwf 0x7F
    movf  0x78, W
    movwf 0x50
    movf  0x79, W
    movwf 0x51
    movf  0x7F, W
    return

; IS_REVEALED: Z=1 jesli NIE odkryte, Z=0 jesli odkryte
IS_REVEALED:
    movlw 0x28
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; IS_MINE: Z=1 jesli NIE mina, Z=0 jesli mina
IS_MINE:
    movlw 0x24
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; MARK_REVEALED: ustawia bit odkrycia
MARK_REVEALED:
    movlw 0x28
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    return

; =============================================================================
; REVEAL_CROSS - odkrywa pole (0x50/0x51) i 4 sasiadow (N/S/W/E)
;                Tylko pola bez min. Jesli wybrany kafelek to mina - przegrana.
; =============================================================================
REVEAL_CROSS:
    ; Odkryj centrum
    call  REVEAL_ONE
    ; Zachowaj centrum
    movf  0x50, W
    movwf 0x7A
    movf  0x51, W
    movwf 0x7B

    ; North: Y-1
    movf  0x7B, F
    btfsc STATUS, Z
    goto  RC_S
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    decf  0x51, F
    call  REVEAL_ONE

RC_S:
    ; South: Y+1
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfsc STATUS, C
    goto  RC_W
    call  REVEAL_ONE

RC_W:
    ; West: X-1
    movf  0x7A, F
    btfsc STATUS, Z
    goto  RC_E
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    decf  0x50, F
    call  REVEAL_ONE

RC_E:
    ; East: X+1
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfsc STATUS, C
    goto  RC_DONE
    call  REVEAL_ONE

RC_DONE:
    ; Przywroc centrum
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    return

; =============================================================================
; REVEAL_ONE - odkrywa jedno pole (0x50/0x51) jesli nie mina i nie odkryte
;              Pomija miny (nie wywoluje game over - to robi CHECK_REVEAL_BTN)
; =============================================================================
REVEAL_ONE:
    ; Pomin jesli juz odkryte
    call  IS_REVEALED
    btfss STATUS, Z
    return
    ; Pomin jesli flaga
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    return
    ; Pomin jesli mina
    call  IS_MINE
    btfss STATUS, Z
    return
    ; Odkryj
    call  MARK_REVEALED
    incf  0x42, F
    call  COUNT_MINES
    call  GET_REVEALED_COLOR
    call  DRAW_TILE
    return

; =============================================================================
; CHECK_WIN: odkryte == 25-miny (18)?
; =============================================================================
CHECK_WIN:
    movf  0x43, W
    sublw d'25'         ; W = 25 - miny
    subwf 0x42, W       ; W = odkryte - wymagane
    btfss STATUS, Z
    return
    movlw 0x01
    movwf 0x41
    ; Pomaluj wszystko na zielono
    movlw 0x00
    movwf 0x50
CW_X:
    movlw 0x00
    movwf 0x51
CW_Y:
    movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  CW_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  CW_X
    return

; =============================================================================
; DRAW_GRID - siatka 5x5 szarymi kafelkami
; =============================================================================
DRAW_GRID:
    movlw 0x00
    movwf 0x50
DG_X:
    movlw 0x00
    movwf 0x51
DG_Y:
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  DG_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  DG_X
    return

; =============================================================================
; UPDATE_CURSOR - odrysuj stare pole, narysuj kursor na nowym miejscu
; =============================================================================
UPDATE_CURSOR:
    ; Sprawdz czy kursor sie ruszyl
    movf  0x50, W
    xorwf 0x54, W
    btfss STATUS, Z
    goto  UC_DO
    movf  0x51, W
    xorwf 0x55, W
    btfss STATUS, Z
    goto  UC_DO
    return
UC_DO:
    ; Odrysuj stare pole (0x50/0x51)
    call  IS_REVEALED
    btfss STATUS, Z
    goto  UC_REVEALED
    ; Nie odkryte: sprawdz flage
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  UC_FLAG
    ; Szary
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
    goto  UC_DRAW
UC_FLAG:
    movlw 0xFF
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    goto  UC_DRAW
UC_REVEALED:
    call  COUNT_MINES
    call  GET_REVEALED_COLOR
UC_DRAW:
    call  DRAW_TILE
    ; Rysuj kursor na nowej pozycji
    movf  0x54, W
    movwf 0x50
    movf  0x55, W
    movwf 0x51
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    return

; =============================================================================
; DRAW_TILE - rysuje kafelek 22x22 na (0x50=X, 0x51=Y), kolor 0x64/0x65
; x0=X*24+60, y0=Y*24+80
; =============================================================================
DRAW_TILE:
    ; x0 = X*16 + X*8 + 60
    movf  0x50, W
    movwf 0x75
    swapf 0x75, W
    andlw 0xF0
    movwf 0x60
    movf  0x50, W
    movwf 0x75
    bcf   STATUS, C
    rlf   0x75, F
    bcf   STATUS, C
    rlf   0x75, F
    bcf   STATUS, C
    rlf   0x75, F
    movf  0x75, W
    addwf 0x60, F
    movlw d'60'
    addwf 0x60, F
    movf  0x60, W
    addlw d'21'
    movwf 0x61

    ; y0 = Y*16 + Y*8 + 80
    movf  0x51, W
    movwf 0x75
    swapf 0x75, W
    andlw 0xF0
    movwf 0x62
    movf  0x51, W
    movwf 0x75
    bcf   STATUS, C
    rlf   0x75, F
    bcf   STATUS, C
    rlf   0x75, F
    bcf   STATUS, C
    rlf   0x75, F
    movf  0x75, W
    addwf 0x62, F
    movlw d'80'
    addwf 0x62, F
    movf  0x62, W
    addlw d'21'
    movwf 0x63

    movlw 0x2A
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movf  0x60, W
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movf  0x61, W
    call  SEND_DAT
    movlw 0x2B
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movf  0x62, W
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movf  0x63, W
    call  SEND_DAT
    movlw 0x2C
    call  SEND_CMD
    bsf   PORTC, 1
    bcf   PORTC, 2
    movlw d'22'
    movwf 0x66
DT_Y:
    movlw d'22'
    movwf 0x67
DT_X:
    bcf   PIR1, 3
    movf  0x64, W
    movwf SSPBUF
DT_W1:
    btfss PIR1, 3
    goto  DT_W1
    bcf   PIR1, 3
    movf  0x65, W
    movwf SSPBUF
DT_W2:
    btfss PIR1, 3
    goto  DT_W2
    decfsz 0x67, F
    goto  DT_X
    decfsz 0x66, F
    goto  DT_Y
    bsf   PORTC, 2
    return

; =============================================================================
; GET_REVEALED_COLOR - In: W=liczba min, Out: 0x64/0x65=kolor
; =============================================================================
GET_REVEALED_COLOR:
    movwf 0x76
    movf  0x76, F
    btfsc STATUS, Z
    goto  GRC_0
    movlw 1
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_1
    movlw 2
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_2
    movlw 3
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_3
    movlw 4
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_4
    movlw 5
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_5
    movlw 6
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_6
    movlw 7
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_7
    goto  GRC_8
GRC_0: movlw 0x00
    movwf 0x64
    movwf 0x65
    return
GRC_1: movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    return
GRC_2: movlw 0x00           ; niebieski 0x001F
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_3: movlw 0xFC
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_4: movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_5: movlw 0xF8
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_6: movlw 0x07
    movwf 0x64
    movlw 0xFF
    movwf 0x65
    return
GRC_7: movlw 0x78
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_8: movlw 0x84
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return

; =============================================================================
; REVEAL_ALL_MINES - odkrywa miny na czerwono (game over)
; =============================================================================
REVEAL_ALL_MINES:
    movlw 0x00
    movwf 0x50
RA_X:
    movlw 0x00
    movwf 0x51
RA_Y:
    call  IS_MINE
    btfsc STATUS, Z
    goto  RA_NEXT
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
RA_NEXT:
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  RA_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  RA_X
    return

; =============================================================================
; SPI
; =============================================================================
SEND_CMD:
    bcf   PORTC, 1
    bcf   PORTC, 2
    goto  SPI_TX
SEND_DAT:
    bsf   PORTC, 1
    bcf   PORTC, 2
SPI_TX:
    bcf   PIR1, 3
    movwf SSPBUF
SPI_W:
    btfss PIR1, 3
    goto  SPI_W
    bsf   PORTC, 2
    return

; =============================================================================
; CHECK_BUTTONS
; RB0=UP RB1=LEFT RB2=DOWN RB3=RIGHT RB4=FLAGA RB6=ODKRYJ RB7=RESTART
; =============================================================================
CHECK_BUTTONS:
    btfss PORTB, 7
    goto  CB_SKIP_RST
    goto  CB_RESTART
CB_SKIP_RST:
    movf  0x41, F
    btfss STATUS, Z
    return              ; gra skonczona

    ; Skopiuj biezacy kursor do docelowego
    movf  0x50, W
    movwf 0x54
    movf  0x51, W
    movwf 0x55

CB_RIGHT:
    btfss PORTB, 3
    goto  CB_LEFT
    movlw d'4'
    subwf 0x54, W
    btfsc STATUS, Z
    goto  CB_WAIT
    incf  0x54, F
    goto  CB_WAIT

CB_LEFT:
    btfss PORTB, 1
    goto  CB_DOWN
    movf  0x54, W
    btfsc STATUS, Z
    goto  CB_WAIT
    decf  0x54, F
    goto  CB_WAIT

CB_DOWN:
    btfss PORTB, 2
    goto  CB_UP
    movlw d'4'
    subwf 0x55, W
    btfsc STATUS, Z
    goto  CB_WAIT
    incf  0x55, F
    goto  CB_WAIT

CB_UP:
    btfss PORTB, 0
    goto  CB_FLAG
    movf  0x55, W
    btfsc STATUS, Z
    goto  CB_WAIT
    decf  0x55, F
    goto  CB_WAIT

CB_FLAG:
    btfss PORTB, 4
    goto  CB_REVEAL
    ; Flaga dziala na biezacym kursorze (0x50/0x51)
    call  IS_REVEALED
    btfss STATUS, Z
    goto  CB_WAIT       ; juz odkryte, ignoruj
    ; XOR flagi
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    xorwf INDF, F
    ; Narysuj odpowiedni kolor (bez blinku - kursor zostaje bialy)
    ; Sprawdz czy flaga jest teraz ustawiona
    movf  0x58, W
    andwf INDF, W
    btfsc STATUS, Z
    goto  CF_UNFLAG
    ; Flaga ustawiona: zolty
    movlw 0xFF
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    goto  CF_DRAW
CF_UNFLAG:
    ; Flaga usunieta: szary
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
CF_DRAW:
    call  DRAW_TILE
    ; Przywroc kursor bialy (kursor jest na tym samym polu co 0x50/0x51)
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    goto  CB_WAIT

CB_REVEAL:
    btfss PORTB, 6
    goto  CB_WAIT_DONE

    ; Ignoruj jesli flaga
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  CB_WAIT

    ; Ignoruj jesli juz odkryte
    call  IS_REVEALED
    btfss STATUS, Z
    goto  CB_WAIT

    ; Mina?
    call  IS_MINE
    btfsc STATUS, Z
    goto  CB_REVEAL_SAFE

    ; --- PRZEGRANA ---
    movlw 0x02
    movwf 0x41
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
    call  REVEAL_ALL_MINES
    movlw d'3'
    movwf 0x75
CB_BLINK:
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
    decfsz 0x75, F
    goto  CB_BLINK
    goto  CB_WAIT_DONE

CB_REVEAL_SAFE:
    call  REVEAL_CROSS
    call  CHECK_WIN

CB_WAIT:
    call  DELAY_20MS
    movf  PORTB, W
    andlw 0xFF
    btfss STATUS, Z
    goto  CB_WAIT

CB_WAIT_DONE:
    return

CB_RESTART:
    ; Czekaj na puszczenie RB7
CB_RST_W:
    call  DELAY_20MS
    btfsc PORTB, 7
    goto  CB_RST_W
    goto  GAME_RESET

; =============================================================================
; DELAY_20MS
; =============================================================================
DELAY_20MS:
    movlw d'100'
    movwf 0x70
DL_O:
    movlw d'66'
    movwf 0x71
DL_I:
    decfsz 0x71, F
    goto  DL_I
    decfsz 0x70, F
    goto  DL_O
    return

    END
