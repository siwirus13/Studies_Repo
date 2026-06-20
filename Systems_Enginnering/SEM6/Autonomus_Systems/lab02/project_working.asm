; =============================================================================
; SAPER (Minesweeper) dla PIC16F877A + ILI9341 (SPI)
; =============================================================================
;
; PRZYCISKI (active-low, PORTB z wbudowanymi pull-upami):
;   RB0 = UP       (kursor w górę)
;   RB1 = DOWN     (kursor w dół)
;   RB2 = LEFT     (kursor w lewo)
;   RB3 = RIGHT    (kursor w prawo)
;   RB4 = REVEAL   (odkryj pole)
;   RB5 = FLAG     (postaw/zdejmij flagę)
;   RB6 = RESTART  (nowa gra)
;
; MAPA PAMIĘCI RAM:
;   0x20-0x2C : warstwa MIN      (10*10 = 100 bitów -> 13 bajtów)
;   0x2D-0x39 : warstwa FLAGI    (10*10 bitów)
;   0x3A-0x46 : warstwa ODKRYTE  (10*10 bitów)
;
;   0x50 : kursor X (0-9)
;   0x51 : kursor Y (0-9)
;
;   0x60 : roboczy X pola
;   0x61 : roboczy Y pola
;   0x62 : adres bazowy warstwy
;   0x63 : wynik GET_BIT (0 lub 1)
;   0x64 : maska bitu (1 bajt)
;   0x65 : tmp ogolny
;   0x66 : tmp ogolny
;   0x67 : tmp ogolny
;
;   0x68 : kolor H (RGB565 high byte)
;   0x69 : kolor L (RGB565 low byte)
;
;   0x70 : wynik COUNT_NEIGHBORS (0-8, 0xFF = błąd/skip)
;   0x71 : DY pętli sąsiadów
;   0x72 : DX pętli sąsiadów
;   0x73 : roboczy sX
;   0x74 : roboczy sY
;   0x75 : roboczy (różne)
;   0x76 : roboczy (różne)
;
;   0x78 : licznik delay outer
;   0x79 : licznik delay inner
;   0x7A-0x7C : liczniki wypełnienia ekranu
;
;   0x7D : LFSR high
;   0x7E : LFSR low
;   0x7F : licznik pętli plant mines
;
;   STOS BFS (flood-fill):
;   0x80-0x8F : X wpisów stosu (16 miejsc)
;   0x90-0x9F : Y wpisów stosu (16 miejsc)
;   0xA0      : wierzchołek stosu (0-16)
;
;   0xA1 : licznik odkrytych pól (max 80 = wygrana)
;   0xA2 : stan gry (0=trwa, 1=wygrana, 2=przegrana)
;
;   0xB0-0xB5 : scratch rysowania kafelka (x0,x1,y0,y1,cntY,cntX)
;
;   SCRATCH FF_PUSH_NEIGHBORS (izolowany):
;   0xC0 : centre X
;   0xC1 : centre Y
;   0xC2 : DX
;   0xC3 : DY
;   0xC4 : kandydat X
;   0xC5 : kandydat Y
;
; =============================================================================

#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF

LAYER_MINE  EQU 0x20
LAYER_FLAG  EQU 0x2D
LAYER_REV   EQU 0x3A
BOARD_W     EQU d'10'
BOARD_H     EQU d'10'
NUM_MINES   EQU d'20'
WIN_COUNT   EQU d'80'

    ORG 0x0000
    goto  START

    ORG 0x0005

; =============================================================================
; START
; =============================================================================
START:
    ; --- Bank 1: konfiguracja kierunków portów ---
    bcf   STATUS, RP1
    bsf   STATUS, RP0
    movlw 0x06
    movwf ADCON1            ; porty A/E jako cyfrowe
    movlw 0xFF
    movwf TRISB             ; PORTB = wejścia
    clrf  TRISD
    bcf   TRISC, 0          ; RC0 = RST
    bcf   TRISC, 1          ; RC1 = D/C
    bcf   TRISC, 2          ; RC2 = CS
    bcf   TRISC, 3          ; RC3 = SCK
    bcf   TRISC, 5          ; RC5 = SDO
    ; Włącz pull-upy PORTB
    bcf   OPTION_REG, 7
    ; SPI: SMP=0, CKE=1
    movlw b'01000000'
    movwf SSPSTAT
    bcf   STATUS, RP0
    ; SPI Master Fosc/4
    movlw b'00100000'
    movwf SSPCON

    ; --- Inicjalizacja LCD ---
    bsf   PORTC, 2          ; CS=1
    bsf   PORTC, 0          ; RST=1
    call  DELAY_20MS
    bcf   PORTC, 0          ; RST=0
    call  DELAY_20MS
    bsf   PORTC, 0          ; RST=1
    call  DELAY_20MS
    movlw 0x01
    call  LCD_CMD           ; Software Reset
    call  DELAY_20MS
    movlw 0x11
    call  LCD_CMD           ; Sleep Out
    call  DELAY_20MS
    movlw 0x3A
    call  LCD_CMD           ; Pixel Format
    movlw 0x55
    call  LCD_DAT           ; 16bpp
    movlw 0x29
    call  LCD_CMD           ; Display ON

    ; --- Wypełnij ekran czarnym ---
    call  LCD_FULL_BLACK

    ; --- Nowa gra ---
    call  GAME_INIT

; --- Główna pętla ---
MAIN_LOOP:
    call  READ_BUTTONS
    goto  MAIN_LOOP

; =============================================================================
; GAME_INIT - zeruje stan, generuje miny, rysuje planszę
; =============================================================================
GAME_INIT:
    ; Wyczyść warstwy 0x20..0x46
    movlw 0x20
    movwf FSR
GI_CLR:
    clrf  INDF
    incf  FSR, F
    movlw 0x47
    subwf FSR, W
    btfss STATUS, Z
    goto  GI_CLR

    ; Zeruj zmienne stanu
    clrf  0xA0
    clrf  0xA1
    clrf  0xA2

    ; Kursor na środek
    movlw d'4'
    movwf 0x50
    movwf 0x51

    ; Seed LFSR (nigdy nie może być 0x0000)
    movlw 0xAC
    movwf 0x7D
    movlw 0xE3
    movwf 0x7E

    call  PLANT_MINES
    call  DRAW_ALL_TILES
    call  DRAW_CURSOR
    return

; =============================================================================
; PLANT_MINES - losuje NUM_MINES unikalnych pozycji min
; LFSR 16-bit, poly x^16+x^14+x^13+x^11+1 (0xB400 Galois)
; =============================================================================
PLANT_MINES:
    movlw NUM_MINES
    movwf 0x7F              ; licznik

PM_NEXT:
    ; Krok LFSR -> nowy bajt w 0x7E
    call  LFSR_STEP

    ; Redukuj 0x7E mod 100
    movf  0x7E, W
    movwf 0x65              ; tmp = random
PM_MOD:
    movlw d'100'
    subwf 0x65, W
    btfss STATUS, C
    goto  PM_GOT            ; 0x65 < 100
    movwf 0x65
    goto  PM_MOD

PM_GOT:
    ; 0x65 = pozycja liniowa 0..99
    ; Zamień na X,Y: Y=pos/10, X=pos%10
    movf  0x65, W
    movwf 0x66              ; kopia
    clrf  0x61              ; Y = 0
PM_DIV:
    movlw d'10'
    subwf 0x66, W
    btfss STATUS, C
    goto  PM_DIV_DONE
    movwf 0x66
    incf  0x61, F
    goto  PM_DIV
PM_DIV_DONE:
    ; 0x61=Y, 0x66=X
    movf  0x66, W
    movwf 0x60              ; 0x60=X

    ; Czy już jest mina?
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  PM_NEXT           ; Jest - losuj ponownie

    ; Postaw minę
    movlw LAYER_MINE
    movwf 0x62
    call  SET_BIT
    decfsz 0x7F, F
    goto  PM_NEXT
    return

; =============================================================================
; LFSR_STEP - jeden krok LFSR 16-bit Galois
; Stan: [0x7D:0x7E], po kroku wynik widoczny w 0x7E
; =============================================================================
LFSR_STEP:
    ; LSB = bit wyjściowy
    bcf   STATUS, C
    btfsc 0x7E, 0
    bsf   STATUS, C         ; C = LSB(0x7E)
    ; Przesuń [0x7D:0x7E] w prawo
    rrf   0x7D, F
    rrf   0x7E, F
    ; Jeśli bit wyjściowy był 1, XOR z maską poly
    btfss STATUS, C
    return
    movlw 0xB4
    xorwf 0x7D, F
    ; Low byte maski = 0x00, nie trzeba XOR
    return

; =============================================================================
; CALC_BIT_ADDR
; Wejście: 0x60=X, 0x61=Y, 0x62=adres bazowy
; Wyjście: FSR -> bajt, 0x64 -> maska
; =============================================================================
CALC_BIT_ADDR:
    ; index = Y*10 + X
    movf  0x61, W
    movwf 0x65              ; tmp = Y
    movf  0x60, W
    movwf 0x66              ; index = X
    movf  0x65, F
    btfsc STATUS, Z
    goto  CBA_NO_Y
CBA_Y:
    movlw d'10'
    addwf 0x66, F
    decfsz 0x65, F
    goto  CBA_Y
CBA_NO_Y:
    ; 0x66 = index
    movf  0x66, W
    movwf 0x67              ; kopia indeksu dla bit%8

    ; bajt = index >> 3
    bcf   STATUS, C
    rrf   0x66, F
    bcf   STATUS, C
    rrf   0x66, F
    bcf   STATUS, C
    rrf   0x66, F
    movlw 0x0F
    andwf 0x66, F           ; max 13 bajtów

    ; FSR = baza + bajt
    movf  0x62, W
    addwf 0x66, W
    movwf FSR

    ; bit = index % 8 = 3 dolne bity kopii
    movlw 0x07
    andwf 0x67, W
    movwf 0x65              ; numer bitu

    ; maska = 1 << numer_bitu
    movlw b'00000001'
    movwf 0x64
    movf  0x65, F
    btfsc STATUS, Z
    goto  CBA_DONE
CBA_SHIFT:
    bcf   STATUS, C
    rlf   0x64, F
    decfsz 0x65, F
    goto  CBA_SHIFT
CBA_DONE:
    return

; =============================================================================
; GET_BIT: wynik w 0x63 (0 lub 1)
; =============================================================================
GET_BIT:
    call  CALC_BIT_ADDR
    movf  0x64, W
    andwf INDF, W
    movlw 0x00
    btfss STATUS, Z
    movlw 0x01
    movwf 0x63
    return

; =============================================================================
; SET_BIT
; =============================================================================
SET_BIT:
    call  CALC_BIT_ADDR
    movf  0x64, W
    iorwf INDF, F
    return

; =============================================================================
; XOR_BIT
; =============================================================================
XOR_BIT:
    call  CALC_BIT_ADDR
    movf  0x64, W
    xorwf INDF, F
    return

; =============================================================================
; COUNT_NEIGHBORS - liczy miny dookoła (0x60, 0x61)
; Wynik: 0x70
; =============================================================================
COUNT_NEIGHBORS:
    clrf  0x70

    movlw 0xFF
    movwf 0x71              ; DY = -1
CN_LOOP_Y:
    movlw 0xFF
    movwf 0x72              ; DX = -1
CN_LOOP_X:
    movf  0x72, F
    btfss STATUS, Z
    goto  CN_CALC
    movf  0x71, F
    btfss STATUS, Z
    goto  CN_CALC
    goto  CN_NEXT_X         ; DX=0, DY=0 -> centrum, pomiń

CN_CALC:
    ; sX = X + DX
    movf  0x60, W
    addwf 0x72, W
    movwf 0x73
    movlw BOARD_W
    subwf 0x73, W
    btfsc STATUS, C
    goto  CN_NEXT_X         ; sX >= 10 (też obsługuje wrap 0xFF+0=255)

    ; sY = Y + DY
    movf  0x61, W
    addwf 0x71, W
    movwf 0x74
    movlw BOARD_H
    subwf 0x74, W
    btfsc STATUS, C
    goto  CN_NEXT_X

    ; Zapisz bieżące X/Y i sprawdź minę na (sX,sY)
    movf  0x60, W
    movwf 0x75
    movf  0x61, W
    movwf 0x76
    movf  0x73, W
    movwf 0x60
    movf  0x74, W
    movwf 0x61
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0x75, W
    movwf 0x60
    movf  0x76, W
    movwf 0x61
    movf  0x63, W
    addwf 0x70, F

CN_NEXT_X:
    incf  0x72, F
    movlw 0x02
    subwf 0x72, W
    btfss STATUS, Z
    goto  CN_LOOP_X
CN_NEXT_Y:
    incf  0x71, F
    movlw 0x02
    subwf 0x71, W
    btfss STATUS, Z
    goto  CN_LOOP_Y
    return

; =============================================================================
; GET_COLOR - ustaw 0x68/0x69 na podstawie wartości w 0x70
; Kolory odkrytych pól (0-5+):
;   0 -> Czarny     0x0000
;   1 -> Zielony    0x07E0
;   2 -> Niebieski  0x001F
;   3 -> Cyjan      0x07FF
;   4 -> Czerwony   0xF800
;   5 -> Magenta    0xF81F
;   6+ -> Pomarańcz 0xFC00
; =============================================================================
GET_COLOR:
    movf  0x70, F
    btfsc STATUS, Z
    goto  GC_0
    movlw 0x01
    subwf 0x70, W
    btfsc STATUS, Z
    goto  GC_1
    movlw 0x02
    subwf 0x70, W
    btfsc STATUS, Z
    goto  GC_2
    movlw 0x03
    subwf 0x70, W
    btfsc STATUS, Z
    goto  GC_3
    movlw 0x04
    subwf 0x70, W
    btfsc STATUS, Z
    goto  GC_4
    movlw 0x05
    subwf 0x70, W
    btfsc STATUS, Z
    goto  GC_5
    goto  GC_6

GC_0:
    movlw 0x00
    movwf 0x68
    movlw 0x00
    movwf 0x69
    return
GC_1:
    movlw 0x07
    movwf 0x68
    movlw 0xE0
    movwf 0x69
    return
GC_2:
    movlw 0x00
    movwf 0x68
    movlw 0x1F
    movwf 0x69
    return
GC_3:
    movlw 0x07
    movwf 0x68
    movlw 0xFF
    movwf 0x69
    return
GC_4:
    movlw 0xF8
    movwf 0x68
    movlw 0x00
    movwf 0x69
    return
GC_5:
    movlw 0xF8
    movwf 0x68
    movlw 0x1F
    movwf 0x69
    return
GC_6:
    movlw 0xFC
    movwf 0x68
    movlw 0x00
    movwf 0x69
    return

; =============================================================================
; DRAW_ALL_TILES - rysuje całą planszę
; =============================================================================
DRAW_ALL_TILES:
    clrf  0x60
DAT_X:
    clrf  0x61
DAT_Y:
    call  DRAW_TILE_STATE
    incf  0x61, F
    movlw BOARD_H
    subwf 0x61, W
    btfss STATUS, Z
    goto  DAT_Y
    incf  0x60, F
    movlw BOARD_W
    subwf 0x60, W
    btfss STATUS, Z
    goto  DAT_X
    return

; =============================================================================
; DRAW_TILE_STATE - rysuje kafelek (0x60, 0x61) wg stanu gry
; =============================================================================
DRAW_TILE_STATE:
    ; Odkryte?
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  DTS_REVEALED

    ; Flagowane?
    movlw LAYER_FLAG
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  DTS_FLAG

    ; Zakryte - szare
    movlw 0x84
    movwf 0x68
    movlw 0x10
    movwf 0x69
    goto  DTS_DRAW

DTS_FLAG:
    ; Żółte
    movlw 0xFF
    movwf 0x68
    movlw 0xE0
    movwf 0x69
    goto  DTS_DRAW

DTS_REVEALED:
    call  COUNT_NEIGHBORS
    call  GET_COLOR

DTS_DRAW:
    call  DRAW_TILE
    return

; =============================================================================
; DRAW_TILE - rysuje kwadrat 18x18 px
; Piksel x0 = X*20+10, y0 = Y*20+10
; Wejście: 0x60=X, 0x61=Y, 0x68/0x69=kolor
; =============================================================================
DRAW_TILE:
    ; --- x0 = X*20 + 10 ---
    ; X*20 = X*16 + X*4
    movf  0x60, W
    movwf 0xB0              ; B0 = X
    swapf 0xB0, W
    andlw 0xF0
    movwf 0xB0              ; B0 = X*16

    movf  0x60, W
    movwf 0xB4              ; B4 = X
    bcf   STATUS, C
    rlf   0xB4, F
    bcf   STATUS, C
    rlf   0xB4, F           ; B4 = X*4

    movf  0xB4, W
    addwf 0xB0, F           ; B0 = X*20
    movlw d'10'
    addwf 0xB0, F           ; B0 = x0

    movf  0xB0, W
    addlw d'17'
    movwf 0xB1              ; B1 = x1

    ; --- y0 = Y*20 + 10 ---
    movf  0x61, W
    movwf 0xB2
    swapf 0xB2, W
    andlw 0xF0
    movwf 0xB2              ; B2 = Y*16

    movf  0x61, W
    movwf 0xB4
    bcf   STATUS, C
    rlf   0xB4, F
    bcf   STATUS, C
    rlf   0xB4, F           ; B4 = Y*4

    movf  0xB4, W
    addwf 0xB2, F
    movlw d'10'
    addwf 0xB2, F           ; B2 = y0

    movf  0xB2, W
    addlw d'17'
    movwf 0xB3              ; B3 = y1

    ; Ustaw okno LCD
    movlw 0x2A
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    movf  0xB0, W
    call  LCD_DAT
    movlw 0x00
    call  LCD_DAT
    movf  0xB1, W
    call  LCD_DAT

    movlw 0x2B
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    movf  0xB2, W
    call  LCD_DAT
    movlw 0x00
    call  LCD_DAT
    movf  0xB3, W
    call  LCD_DAT

    movlw 0x2C
    call  LCD_CMD
    bsf   PORTC, 1
    bcf   PORTC, 2

    movlw d'18'
    movwf 0xB4
DT_ROW:
    movlw d'18'
    movwf 0xB5
DT_COL:
    bcf   PIR1, 3
    movf  0x68, W
    movwf SSPBUF
DT_W1:
    btfss PIR1, 3
    goto  DT_W1
    bcf   PIR1, 3
    movf  0x69, W
    movwf SSPBUF
DT_W2:
    btfss PIR1, 3
    goto  DT_W2
    decfsz 0xB5, F
    goto  DT_COL
    decfsz 0xB4, F
    goto  DT_ROW

    bsf   PORTC, 2
    return

; =============================================================================
; DRAW_CURSOR - białe obramowanie 18x18, wnętrze 14x14 z właściwym kolorem
; Pozycja z 0x50/0x51
; =============================================================================
DRAW_CURSOR:
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61

    ; Narysuj biały kafelek (obramowanie)
    movlw 0xFF
    movwf 0x68
    movwf 0x69
    call  DRAW_TILE

    ; Oblicz kolor wnętrza
    call  GET_INNER_COLOR    ; wynik w 0x68/0x69

    ; Przelicz współrzędne wnętrza: x0+2, y0+2, x0+15, y0+15
    ; Ponownie oblicz x0 dla 0x50
    movf  0x60, W
    movwf 0xB0
    swapf 0xB0, W
    andlw 0xF0
    movwf 0xB0

    movf  0x60, W
    movwf 0xB4
    bcf   STATUS, C
    rlf   0xB4, F
    bcf   STATUS, C
    rlf   0xB4, F

    movf  0xB4, W
    addwf 0xB0, F
    movlw d'12'             ; 10+2
    addwf 0xB0, F           ; B0 = x0_inner

    movf  0xB0, W
    addlw d'13'             ; 14px szeroki
    movwf 0xB1

    movf  0x61, W
    movwf 0xB2
    swapf 0xB2, W
    andlw 0xF0
    movwf 0xB2

    movf  0x61, W
    movwf 0xB4
    bcf   STATUS, C
    rlf   0xB4, F
    bcf   STATUS, C
    rlf   0xB4, F

    movf  0xB4, W
    addwf 0xB2, F
    movlw d'12'
    addwf 0xB2, F           ; B2 = y0_inner

    movf  0xB2, W
    addlw d'13'
    movwf 0xB3

    ; Wyślij okno wewnętrzne
    movlw 0x2A
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    movf  0xB0, W
    call  LCD_DAT
    movlw 0x00
    call  LCD_DAT
    movf  0xB1, W
    call  LCD_DAT

    movlw 0x2B
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    movf  0xB2, W
    call  LCD_DAT
    movlw 0x00
    call  LCD_DAT
    movf  0xB3, W
    call  LCD_DAT

    movlw 0x2C
    call  LCD_CMD
    bsf   PORTC, 1
    bcf   PORTC, 2

    movlw d'14'
    movwf 0xB4
DC_ROW:
    movlw d'14'
    movwf 0xB5
DC_COL:
    bcf   PIR1, 3
    movf  0x68, W
    movwf SSPBUF
DC_W1:
    btfss PIR1, 3
    goto  DC_W1
    bcf   PIR1, 3
    movf  0x69, W
    movwf SSPBUF
DC_W2:
    btfss PIR1, 3
    goto  DC_W2
    decfsz 0xB5, F
    goto  DC_COL
    decfsz 0xB4, F
    goto  DC_ROW

    bsf   PORTC, 2
    return

; =============================================================================
; GET_INNER_COLOR - kolor wnętrza dla (0x60, 0x61), wynik w 0x68/0x69
; =============================================================================
GET_INNER_COLOR:
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  GIC_REV

    movlw LAYER_FLAG
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  GIC_FLAG

    ; Zakryte: ciemny szary (ciemniejszy niż normalny szary)
    movlw 0x52
    movwf 0x68
    movlw 0x8A
    movwf 0x69
    return

GIC_FLAG:
    movlw 0xFF
    movwf 0x68
    movlw 0xE0
    movwf 0x69
    return

GIC_REV:
    call  COUNT_NEIGHBORS
    call  GET_COLOR
    return

; =============================================================================
; READ_BUTTONS
; =============================================================================
READ_BUTTONS:
    ; Gra skończona? Reaguj tylko na RESTART
    movf  0xA2, F
    btfss STATUS, Z
    goto  RB_GAME_OVER

    movf  PORTB, W
    movwf 0x73

    ; RESTART (RB6)
    btfss 0x73, 6
    goto  DO_RESTART

    ; UP (RB0)
    btfsc 0x73, 0
    goto  RB_NOT_UP
    movf  0x51, F
    btfsc STATUS, Z
    goto  RB_NOT_UP
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    call  DRAW_TILE_STATE
    decf  0x51, F
    call  DRAW_CURSOR
    call  DEBOUNCE
    return
RB_NOT_UP:

    ; DOWN (RB1)
    btfsc 0x73, 1
    goto  RB_NOT_DOWN
    movlw d'9'
    subwf 0x51, W
    btfsc STATUS, Z
    goto  RB_NOT_DOWN
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    call  DRAW_TILE_STATE
    incf  0x51, F
    call  DRAW_CURSOR
    call  DEBOUNCE
    return
RB_NOT_DOWN:

    ; LEFT (RB2)
    btfsc 0x73, 2
    goto  RB_NOT_LEFT
    movf  0x50, F
    btfsc STATUS, Z
    goto  RB_NOT_LEFT
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    call  DRAW_TILE_STATE
    decf  0x50, F
    call  DRAW_CURSOR
    call  DEBOUNCE
    return
RB_NOT_LEFT:

    ; RIGHT (RB3)
    btfsc 0x73, 3
    goto  RB_NOT_RIGHT
    movlw d'9'
    subwf 0x50, W
    btfsc STATUS, Z
    goto  RB_NOT_RIGHT
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    call  DRAW_TILE_STATE
    incf  0x50, F
    call  DRAW_CURSOR
    call  DEBOUNCE
    return
RB_NOT_RIGHT:

    ; REVEAL (RB4)
    btfsc 0x73, 4
    goto  RB_NOT_REVEAL
    call  DO_REVEAL
    call  DEBOUNCE
    return
RB_NOT_REVEAL:

    ; FLAG (RB5)
    btfsc 0x73, 5
    goto  RB_NOT_FLAG
    call  DO_FLAG
    call  DEBOUNCE
    return
RB_NOT_FLAG:
    return

RB_GAME_OVER:
    btfss PORTB, 6
    call  DO_RESTART
    return

; =============================================================================
; DO_REVEAL - odkryj pole pod kursorem
; =============================================================================
DO_REVEAL:
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61

    ; Ma flagę? -> ignoruj
    movlw LAYER_FLAG
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    return

    ; Już odkryte? -> ignoruj
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    return

    ; Mina?
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  DR_MINE

    ; Bezpieczne - flood fill
    call  FLOOD_FILL
    call  CHECK_WIN
    ; Odśwież kursor po flood fill
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    call  DRAW_CURSOR
    return

DR_MINE:
    movlw 0x02
    movwf 0xA2              ; stan = przegrana
    call  REVEAL_ALL_MINES
    call  BLINK_MINE        ; mignij na polu miny
    return

; =============================================================================
; DO_FLAG
; =============================================================================
DO_FLAG:
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61

    ; Nie flaguj odkrytego
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    return

    movlw LAYER_FLAG
    movwf 0x62
    call  XOR_BIT

    ; Przerysuj z kursorem (kursor jest w tej samej pozycji)
    call  DRAW_CURSOR
    return

; =============================================================================
; FLOOD_FILL - iteracyjny BFS od (0x60, 0x61)
; =============================================================================
FLOOD_FILL:
    clrf  0xA0              ; wyczyść stos BFS

    ; Odkryj pole startowe
    call  FF_REVEAL_CELL
    ; 0x70 = liczba sąsiadów (lub 0xFF jeśli pominięto)
    movlw 0xFF
    subwf 0x70, W
    btfsc STATUS, Z
    return                  ; Pominięto (już odkryte/flaga/mina)
    movf  0x70, F
    btfss STATUS, Z
    return                  ; Ma sąsiadów-miny - nie rozszerzaj

    ; 0 sąsiadów: wrzuć sąsiadów startowego na stos
    call  FF_PUSH_NEIGHBORS

FF_LOOP:
    movf  0xA0, F
    btfsc STATUS, Z
    return                  ; Stos pusty - gotowe

    ; Pop
    decf  0xA0, F

    movlw 0x80
    addwf 0xA0, W
    movwf FSR
    movf  INDF, W
    movwf 0x60

    movlw 0x90
    addwf 0xA0, W
    movwf FSR
    movf  INDF, W
    movwf 0x61

    call  FF_REVEAL_CELL
    movlw 0xFF
    subwf 0x70, W
    btfsc STATUS, Z
    goto  FF_LOOP           ; Pominięto
    movf  0x70, F
    btfss STATUS, Z
    goto  FF_LOOP           ; Ma sąsiadów

    call  FF_PUSH_NEIGHBORS
    goto  FF_LOOP

; =============================================================================
; FF_REVEAL_CELL - odkrywa (0x60, 0x61)
; Wynik: 0x70 = liczba sąsiadów, lub 0xFF = pominięto
; =============================================================================
FF_REVEAL_CELL:
    ; Już odkryte?
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  FRC_SKIP

    ; Flagowane?
    movlw LAYER_FLAG
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  FRC_SKIP

    ; Mina?
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfss STATUS, Z
    goto  FRC_SKIP

    ; Odkryj
    movlw LAYER_REV
    movwf 0x62
    call  SET_BIT
    incf  0xA1, F

    call  COUNT_NEIGHBORS   ; wynik w 0x70
    call  GET_COLOR         ; wynik w 0x68/0x69
    call  DRAW_TILE
    return

FRC_SKIP:
    movlw 0xFF
    movwf 0x70
    return

; =============================================================================
; FF_PUSH_NEIGHBORS - wrzuca sąsiadów (0x60, 0x61) na stos
; Scratch izolowany: 0xC0-0xC5
; =============================================================================
FF_PUSH_NEIGHBORS:
    movf  0x60, W
    movwf 0xC0
    movf  0x61, W
    movwf 0xC1

    movlw 0xFF
    movwf 0xC3              ; DY = -1
FFPN_Y:
    movlw 0xFF
    movwf 0xC2              ; DX = -1
FFPN_X:
    ; Pomiń centrum
    movf  0xC2, F
    btfss STATUS, Z
    goto  FFPN_CALC
    movf  0xC3, F
    btfss STATUS, Z
    goto  FFPN_CALC
    goto  FFPN_NX

FFPN_CALC:
    ; candX = cX + DX
    movf  0xC0, W
    addwf 0xC2, W
    movwf 0xC4
    movlw BOARD_W
    subwf 0xC4, W
    btfsc STATUS, C
    goto  FFPN_NX           ; poza planszą

    ; candY = cY + DY
    movf  0xC1, W
    addwf 0xC3, W
    movwf 0xC5
    movlw BOARD_H
    subwf 0xC5, W
    btfsc STATUS, C
    goto  FFPN_NX

    ; Odkryte?
    movf  0xC4, W
    movwf 0x60
    movf  0xC5, W
    movwf 0x61
    movlw LAYER_REV
    movwf 0x62
    call  GET_BIT
    movf  0xC0, W
    movwf 0x60
    movf  0xC1, W
    movwf 0x61
    movf  0x63, F
    btfss STATUS, Z
    goto  FFPN_NX           ; Już odkryte

    ; Mina?
    movf  0xC4, W
    movwf 0x60
    movf  0xC5, W
    movwf 0x61
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0xC0, W
    movwf 0x60
    movf  0xC1, W
    movwf 0x61
    movf  0x63, F
    btfss STATUS, Z
    goto  FFPN_NX           ; Mina

    ; Stos pełny?
    movlw d'16'
    subwf 0xA0, W
    btfsc STATUS, C
    goto  FFPN_NX

    ; Push
    movlw 0x80
    addwf 0xA0, W
    movwf FSR
    movf  0xC4, W
    movwf INDF

    movlw 0x90
    addwf 0xA0, W
    movwf FSR
    movf  0xC5, W
    movwf INDF

    incf  0xA0, F

FFPN_NX:
    incf  0xC2, F
    movlw 0x02
    subwf 0xC2, W
    btfss STATUS, Z
    goto  FFPN_X
FFPN_NY:
    incf  0xC3, F
    movlw 0x02
    subwf 0xC3, W
    btfss STATUS, Z
    goto  FFPN_Y
    return

; =============================================================================
; CHECK_WIN
; =============================================================================
CHECK_WIN:
    movlw WIN_COUNT
    subwf 0xA1, W
    btfss STATUS, Z
    return
    ; Wygrana - zamaluj planszę zielono
    movlw 0x01
    movwf 0xA2
    clrf  0x60
CW_X:
    clrf  0x61
CW_Y:
    movlw 0x07
    movwf 0x68
    movlw 0xE0
    movwf 0x69
    call  DRAW_TILE
    incf  0x61, F
    movlw BOARD_H
    subwf 0x61, W
    btfss STATUS, Z
    goto  CW_Y
    incf  0x60, F
    movlw BOARD_W
    subwf 0x60, W
    btfss STATUS, Z
    goto  CW_X
    return

; =============================================================================
; REVEAL_ALL_MINES - pokaż wszystkie miny czerwono
; =============================================================================
REVEAL_ALL_MINES:
    clrf  0x60
RAM_X:
    clrf  0x61
RAM_Y:
    movlw LAYER_MINE
    movwf 0x62
    call  GET_BIT
    movf  0x63, F
    btfsc STATUS, Z
    goto  RAM_NEXT
    movlw 0xF8
    movwf 0x68
    movlw 0x00
    movwf 0x69
    call  DRAW_TILE
RAM_NEXT:
    incf  0x61, F
    movlw BOARD_H
    subwf 0x61, W
    btfss STATUS, Z
    goto  RAM_Y
    incf  0x60, F
    movlw BOARD_W
    subwf 0x60, W
    btfss STATUS, Z
    goto  RAM_X
    return

; =============================================================================
; BLINK_MINE - miga czerwono/biało na polu kursora 3 razy
; =============================================================================
BLINK_MINE:
    movlw d'3'
    movwf 0x75
BM_LOOP:
    movf  0x50, W
    movwf 0x60
    movf  0x51, W
    movwf 0x61
    movlw 0xFF
    movwf 0x68
    movwf 0x69
    call  DRAW_TILE
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xF8
    movwf 0x68
    movlw 0x00
    movwf 0x69
    call  DRAW_TILE
    call  DELAY_20MS
    call  DELAY_20MS
    decfsz 0x75, F
    goto  BM_LOOP
    return

; =============================================================================
; DO_RESTART
; =============================================================================
DO_RESTART:
    ; Czekaj aż zwolniony
DR_WAIT:
    btfss PORTB, 6
    goto  DR_WAIT
    call  DELAY_20MS
    btfss PORTB, 6
    goto  DR_WAIT
    call  GAME_INIT
    return

; =============================================================================
; LCD_FULL_BLACK
; =============================================================================
LCD_FULL_BLACK:
    movlw 0x2A
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    call  LCD_DAT
    movlw 0x00
    call  LCD_DAT
    movlw 0xEF
    call  LCD_DAT
    movlw 0x2B
    call  LCD_CMD
    movlw 0x00
    call  LCD_DAT
    call  LCD_DAT
    movlw 0x01
    call  LCD_DAT
    movlw 0x3F
    call  LCD_DAT
    movlw 0x2C
    call  LCD_CMD
    bsf   PORTC, 1
    bcf   PORTC, 2
    movlw d'240'
    movwf 0x7A
LFB_Y:
    movlw d'4'
    movwf 0x7B
LFB_XOUT:
    movlw d'80'
    movwf 0x7C
LFB_XIN:
    bcf   PIR1, 3
    movlw 0x00
    movwf SSPBUF
LFB_W1:
    btfss PIR1, 3
    goto  LFB_W1
    bcf   PIR1, 3
    movlw 0x00
    movwf SSPBUF
LFB_W2:
    btfss PIR1, 3
    goto  LFB_W2
    decfsz 0x7C, F
    goto  LFB_XIN
    decfsz 0x7B, F
    goto  LFB_XOUT
    decfsz 0x7A, F
    goto  LFB_Y
    bsf   PORTC, 2
    return

; =============================================================================
; LCD_CMD / LCD_DAT / SPI_SEND
; =============================================================================
LCD_CMD:
    bcf   PORTC, 1          ; D/C=0 komenda
    bcf   PORTC, 2          ; CS=0
    goto  SPI_SEND
LCD_DAT:
    bsf   PORTC, 1          ; D/C=1 dane
    bcf   PORTC, 2
SPI_SEND:
    bcf   PIR1, 3
    movwf SSPBUF
SPI_WAIT:
    btfss PIR1, 3
    goto  SPI_WAIT
    bsf   PORTC, 2
    return

; =============================================================================
; DEBOUNCE - czeka aż wszystkie przyciski zwolnione
; =============================================================================
DEBOUNCE:
    call  DELAY_20MS
DB_WAIT:
    movf  PORTB, W
    andlw b'01111111'       ; bity 0-6
    xorlw b'01111111'       ; active-low: zwolnione = 0x7F -> XOR = 0x00
    btfss STATUS, Z
    goto  DB_WAIT
    return

; =============================================================================
; DELAY_20MS (przy 4MHz Fosc)
; =============================================================================
DELAY_20MS:
    movlw d'100'
    movwf 0x78
DL_OUT:
    movlw d'66'
    movwf 0x79
DL_IN:
    decfsz 0x79, F
    goto  DL_IN
    decfsz 0x78, F
    goto  DL_OUT
    return

    END
