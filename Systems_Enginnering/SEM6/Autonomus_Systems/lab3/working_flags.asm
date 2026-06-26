; =============================================================================
; SAPER (Minesweeper) dla PIC16F877A + ILI9341 (SPI)
; =============================================================================
; MAPA REJESTRÓW:
;   0x20-0x2C : Warstwa flag    (10x10 = 100 bitów -> 13 bajtów)
;   0x2D-0x39 : Warstwa min     (10x10 = 100 bitów -> 13 bajtów)
;   0x3A-0x46 : Warstwa odkrytych (10x10 = 100 bitów -> 13 bajtów)
;
;   0x50 : Aktualny X kursora (bieżące pole przy rysowaniu)
;   0x51 : Aktualny Y kursora
;   0x54 : Docelowy X (po wciśnięciu przycisku)
;   0x55 : Docelowy Y
;   0x56 : Tymczasowy indeks liniowy
;   0x57 : Tymczasowy (przesunięcie bajtowe / licznik pętli)
;   0x58 : Wynikowa maska bitu
;   0x59 : Adres bazowy warstwy (0x20/0x2D/0x3A)
;   0x60-0x63 : x0, x1, y0, y1 okna LCD
;   0x64-0x65 : Kolor (RGB565, High/Low)
;   0x66-0x67 : Liczniki pętli rysowania kafelka
;   0x70-0x71 : Liczniki opóźnienia
;   0x72-0x74 : Liczniki wypełniania ekranu
;   0x75 : Tymczasowy przy obliczaniu współrzędnych LCD
;   0x76 : Tymczasowy (kopia W)
;   0x78-0x79 : Kopia X/Y przy CHECK_MINE_AT
;   0x7A-0x7B : Współrzędne sąsiada przy licz sąsiadów
;   0x7C : Wynik COUNT_MINES (liczba sąsiadów-min)
;   0x7D : Delta Y sąsiada (-1,0,+1 jako 0xFF,0x00,0x01)
;   0x7E : Delta X sąsiada
;   0x7F : Tymczasowy wynik CHECK_MINE_AT
;
; STOS FLOOD-FILL (0x80-0x9E, 16 pozycji X/Y)
;   0x80-0x8F : Bufor X stosu (16 wpisów)
;   0x90-0x9F : Bufor Y stosu (16 wpisów)
;   0xA0 : Wskaźnik wierzchołka stosu (0-16)
;
; KONTROLA GRY
;   0xA1 : Stan gry (0=trwa, 1=wygrana, 2=przegrana)
;   0xA2 : Licznik odkrytych pól
;   0xA3 : Łączna liczba min
; =============================================================================

#include <p16f877a.inc>
    __CONFIG _FOSC_EXTRC & _WDTE_OFF & _PWRTE_OFF & _BOREN_OFF & _LVP_ON & _CPD_OFF & _WRT_OFF & _CP_OFF

    ORG 0x0000
    goto START

    ORG 0x0005

START:
    ; --- Konfiguracja portów (Bank 1) ---
    bcf  STATUS, RP1
    bsf  STATUS, RP0
    movlw 0x06
    movwf ADCON1            ; Wyłącz ADC na PORTA/E
    movlw 0xFF
    movwf TRISB             ; PORTB jako wejścia (przyciski RB0-RB5)
    clrf  TRISD
    bcf   TRISC, 0          ; RC0 = RESET LCD (wyjście)
    bcf   TRISC, 1          ; RC1 = D/C LCD  (wyjście)
    bcf   TRISC, 2          ; RC2 = CS LCD   (wyjście)
    bcf   TRISC, 3          ; RC3 = SCK SPI  (wyjście)
    bcf   TRISC, 5          ; RC5 = SDO SPI  (wyjście)
    ; Konfiguracja SPI Master (Bank 1)
    movlw b'01000000'       ; SMP=0, CKE=1
    movwf SSPSTAT
    bcf   STATUS, RP0

    ; Konfiguracja SPI (Bank 0)
    movlw b'00100000'       ; SSPEN, Master, Fosc/4
    movwf SSPCON

    ; --- Inicjalizacja LCD ILI9341 ---
    bsf   PORTC, 2          ; CS=1 (nieaktywny)
    bsf   PORTC, 0          ; RESET=1
    call  DELAY_20MS
    bcf   PORTC, 0          ; RESET=0
    call  DELAY_20MS
    bsf   PORTC, 0          ; RESET=1
    call  DELAY_20MS
    movlw 0x01              ; Software Reset
    call  SEND_CMD
    call  DELAY_20MS
    movlw 0x11              ; Sleep Out
    call  SEND_CMD
    call  DELAY_20MS
    movlw 0x3A              ; Pixel Format
    call  SEND_CMD
    movlw 0x55              ; 16-bit (RGB565)
    call  SEND_DAT
    movlw 0x29              ; Display ON
    call  SEND_CMD

    ; --- Wypełnij ekran czarnym ---
    movlw 0x2A              ; Column Address Set
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movlw 0xEF              ; x=239
    call  SEND_DAT
    movlw 0x2B              ; Row Address Set
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movlw 0x01
    call  SEND_DAT
    movlw 0x3F              ; y=319
    call  SEND_DAT
    movlw 0x2C              ; Memory Write
    call  SEND_CMD
    bsf   PORTC, 1          ; D/C=1 (dane)
    bcf   PORTC, 2          ; CS=0 (aktywny)
    movlw d'240'
    movwf 0x72
FILL_Y:
    movlw d'4'
    movwf 0x73
FILL_X_OUTER:
    movlw d'80'
    movwf 0x74
FILL_X_INNER:
    bcf  PIR1, 3
    movlw 0x00
    movwf SSPBUF
WAIT_H:
    btfss PIR1, 3
    goto  WAIT_H
    bcf   PIR1, 3
    movlw 0x00
    movwf SSPBUF
WAIT_L:
    btfss PIR1, 3
    goto  WAIT_L
    decfsz 0x74, F
    goto  FILL_X_INNER
    decfsz 0x73, F
    goto  FILL_X_OUTER
    decfsz 0x72, F
    goto  FILL_Y
    bsf   PORTC, 2          ; CS=1

    ; --- Wyczyść RAM gry ---
    movlw 0x20
    movwf FSR
CLEAR_RAM_LOOP:
    clrf  INDF
    incf  FSR, F
    movlw 0x47
    subwf FSR, W
    btfss STATUS, Z
    goto  CLEAR_RAM_LOOP

    ; --- Inicjalizacja zmiennych gry ---
    clrf  0xA0              ; Stack pointer = 0
    clrf  0xA1              ; Stan gry = trwa
    clrf  0xA2              ; Odkryte = 0
    movlw d'10'             ; 10 min (możesz zmienić)
    movwf 0xA3

    call  PLANT_DUMMY_MINES
    call  DRAW_GRID
    call  DRAW_LEGEND

GAME_INIT:
    movlw d'4'
    movwf 0x50
    movwf 0x51
    movwf 0x54
    movwf 0x55
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE         ; Rysuj kursor startowy

MAIN_LOOP:
    ; Jeśli gra skończona, czekaj w pętli
    movf  0xA1, F
    btfss STATUS, Z
    goto  GAME_OVER_LOOP
    call  CHECK_BUTTONS
    call  UPDATE_CURSOR
    goto  MAIN_LOOP

GAME_OVER_LOOP:
    goto  GAME_OVER_LOOP    ; Nieskończona pętla po końcu gry

; =============================================================================
; ROZMIESZCZENIE MIN (hardcoded do testów - zastąp losowaniem)
; =============================================================================
PLANT_DUMMY_MINES:
    movlw 0x2D
    movwf 0x59
    ; X=1, Y=1
    movlw d'1'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=5, Y=5
    movlw d'5'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=8, Y=2
    movlw d'8'
    movwf 0x50
    movlw d'2'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=3, Y=7
    movlw d'3'
    movwf 0x50
    movlw d'7'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=0, Y=0
    movlw d'0'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=9, Y=9
    movlw d'9'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=2, Y=4
    movlw d'2'
    movwf 0x50
    movlw d'4'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=6, Y=1
    movlw d'6'
    movwf 0x50
    movlw d'1'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=4, Y=8
    movlw d'4'
    movwf 0x50
    movlw d'8'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=7, Y=3
    movlw d'7'
    movwf 0x50
    movlw d'3'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    return

; =============================================================================
; GET_BITMASK_AND_PTR
; Wejście: 0x50=X, 0x51=Y, 0x59=adres bazowy warstwy
; Wyjście: FSR wskazuje na właściwy bajt, 0x58=maska bitu, 0x56=index liniowy
; =============================================================================
GET_BITMASK_AND_PTR:
    ; Index = Y*10 + X
    movf  0x51, W
    movwf 0x57          ; Licznik Y
    movf  0x50, W
    movwf 0x56          ; Index = X
    movf  0x57, F
    btfsc STATUS, Z
    goto  GTB_CALC_OFFSET
GTB_LOOP_Y:
    movlw d'10'
    addwf 0x56, F       ; Index += 10
    decfsz 0x57, F
    goto  GTB_LOOP_Y

GTB_CALC_OFFSET:
    ; Bajt = Index / 8 (przez 3x przesunięcie w prawo)
    movf  0x56, W
    movwf 0x57
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    movlw 0x1F          ; Max 13 bajtów
    andwf 0x57, F
    movf  0x59, W
    addwf 0x57, W
    movwf FSR           ; FSR = adres bazowy + bajt

    ; Bit = Index % 8
    movlw 0x07
    andwf 0x56, W
    movwf 0x57          ; Numer bitu (0-7)

    movlw b'00000001'
    movwf 0x58
    movf  0x57, F
    btfsc STATUS, Z
    goto  GTB_END
GTB_MASK_LOOP:
    bcf   STATUS, C
    rlf   0x58, F
    decfsz 0x57, F
    goto  GTB_MASK_LOOP
GTB_END:
    return

; =============================================================================
; COUNT_MINES
; Wejście: 0x50=X, 0x51=Y
; Wyjście: 0x7C = liczba min w sąsiadujących 8 polach
; Niszczy: 0x7A-0x7F
; =============================================================================
COUNT_MINES:
    clrf  0x7C              ; Wynik = 0

    ; Pętla po DY = -1, 0, +1 (zapisane jako 0xFF, 0x00, 0x01)
    movlw 0xFF
    movwf 0x7D              ; DY = -1
CM_LOOP_Y:
    movlw 0xFF
    movwf 0x7E              ; DX = -1
CM_LOOP_X:
    ; Pomiń (DX==0 AND DY==0) - to jest samo pole
    movf  0x7E, F
    btfss STATUS, Z
    goto  CM_CALC_COORD     ; DX != 0, oblicz
    movf  0x7D, F
    btfss STATUS, Z
    goto  CM_CALC_COORD     ; DY != 0, oblicz
    goto  CM_NEXT_X         ; Oba zero - pomiń

CM_CALC_COORD:
    ; sX = X + DX
    movf  0x50, W
    addwf 0x7E, W
    movwf 0x7A              ; sX

    ; Sprawdź sX >= 0 (DX=0xFF to -1, X+0xFF < X gdy X>0, ale może wraparound)
    ; Rzeczywisty test: jeśli DX=0xFF i X=0, wynik=0xFF -> odrzuć (C=0 po addwf unsigned)
    ; Użyj: jeśli wynik >= 10, odrzuć
    movlw d'10'
    subwf 0x7A, W
    btfsc STATUS, C
    goto  CM_NEXT_X         ; sX >= 10, poza planszą

    ; sY = Y + DY
    movf  0x51, W
    addwf 0x7D, W
    movwf 0x7B              ; sY

    movlw d'10'
    subwf 0x7B, W
    btfsc STATUS, C
    goto  CM_NEXT_X         ; sY >= 10, poza planszą

    ; Sprawdź czy na (sX, sY) jest mina
    call  CHECK_MINE_AT
    addwf 0x7C, F           ; Wynik += 0 lub 1

CM_NEXT_X:
    incf  0x7E, F           ; DX++
    movlw 0x02
    subwf 0x7E, W
    btfss STATUS, Z
    goto  CM_LOOP_X         ; Kontynuuj dopóki DX <= 1

CM_NEXT_Y:
    incf  0x7D, F           ; DY++
    movlw 0x02
    subwf 0x7D, W
    btfss STATUS, Z
    goto  CM_LOOP_Y         ; Kontynuuj dopóki DY <= 1

    movf  0x7C, W
    return

; =============================================================================
; CHECK_MINE_AT
; Wejście: 0x7A=X sąsiada, 0x7B=Y sąsiada
; Wyjście: W = 1 jeśli mina, 0 jeśli nie
; Zachowuje: 0x50, 0x51
; =============================================================================
CHECK_MINE_AT:
    ; Zachowaj bieżące X/Y
    movf  0x50, W
    movwf 0x78
    movf  0x51, W
    movwf 0x79
    ; Ustaw X/Y na sąsiada
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51
    ; Wskaż warstwę min
    movlw 0x2D
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    ; Test bitu
    movf  0x58, W
    andwf INDF, W
    movlw 0x00              ; Zakładaj: brak miny
    btfss STATUS, Z
    movlw 0x01              ; Jest mina
    movwf 0x7F
    ; Przywróć X/Y
    movf  0x78, W
    movwf 0x50
    movf  0x79, W
    movwf 0x51
    movf  0x7F, W
    return

; =============================================================================
; IS_REVEALED - sprawdza czy pole (0x50, 0x51) jest odkryte
; Wyjście: STATUS,Z=1 jeśli NIE odkryte, Z=0 jeśli odkryte
; =============================================================================
IS_REVEALED:
    movlw 0x3A
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; =============================================================================
; IS_MINE - sprawdza czy pole (0x50, 0x51) jest miną
; Wyjście: STATUS,Z=1 jeśli NIE mina, Z=0 jeśli mina
; =============================================================================
IS_MINE:
    movlw 0x2D
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; =============================================================================
; REVEAL_TILE - odkrywa pole (0x50, 0x51)
; Obsługuje flood-fill dla pól z 0 sąsiadami
; Aktualizuje licznik odkrytych (0xA2) i sprawdza wygraną
; =============================================================================
REVEAL_TILE:
    ; 1. Jeśli już odkryte, wyjdź
    call  IS_REVEALED
    btfss STATUS, Z
    return                  ; Już odkryte

    ; 2. Jeśli ma flagę, zignoruj (flaga blokuje odkrycie)
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    return                  ; Zablokowane flagą

    ; 3. Oznacz jako odkryte
    movlw 0x3A
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F

    ; 4. Zwiększ licznik odkrytych
    incf  0xA2, F

    ; 5. Policz sąsiednie miny
    call  COUNT_MINES       ; Wynik w W i 0x7C

    ; 6. Narysuj kafelek z odpowiednim kolorem
    call  GET_REVEALED_COLOR
    call  DRAW_TILE

    ; 7. Jeśli 0 min w sąsiedztwie - flood fill (stos)
    movf  0x7C, F
    btfss STATUS, Z
    return                  ; Są sąsiednie miny, nie rozszerzaj

    ; Push sąsiadów na stos
    call  FLOOD_REVEAL_NEIGHBORS
    return

; =============================================================================
; =============================================================================
; FLOOD_REVEAL_NEIGHBORS
; Simple 4-direction (N/S/E/W) flood reveal — no stack, no register collision.
; Called when a revealed tile has 0 mine neighbours.
; Each of the 4 neighbours is checked and revealed with REVEAL_NEIGHBOR.
; REVEAL_NEIGHBOR itself checks bounds, already-revealed, mine status,
; and calls COUNT_MINES + DRAW_TILE inline (no subroutine calls that
; clobber 0x50/0x51 while we still need the centre).
;
; Registers used (all local, not shared with COUNT_MINES / IS_REVEALED):
;   0xB0 = centre X,  0xB1 = centre Y
;   0xB4 = candidate X (neighbour),  0xB5 = candidate Y
; =============================================================================
FLOOD_REVEAL_NEIGHBORS:
    movf  0x50, W
    movwf 0xB0          ; save centre X
    movf  0x51, W
    movwf 0xB1          ; save centre Y

    ; --- North: same X, Y-1 ---
    movf  0xB0, W
    movwf 0xB4
    movf  0xB1, W
    movwf 0xB5
    movf  0xB5, F
    btfsc STATUS, Z
    goto  FRN_SOUTH     ; Y==0, skip north
    decf  0xB5, F
    call  REVEAL_NEIGHBOR

FRN_SOUTH:
    ; --- South: same X, Y+1 ---
    movf  0xB0, W
    movwf 0xB4
    movf  0xB1, W
    movwf 0xB5
    incf  0xB5, F
    movlw d'10'
    subwf 0xB5, W
    btfsc STATUS, C
    goto  FRN_WEST      ; Y+1 >= 10, skip south
    call  REVEAL_NEIGHBOR

FRN_WEST:
    ; --- West: X-1, same Y ---
    movf  0xB0, W
    movwf 0xB4
    movf  0xB1, W
    movwf 0xB5
    movf  0xB4, F
    btfsc STATUS, Z
    goto  FRN_EAST      ; X==0, skip west
    decf  0xB4, F
    call  REVEAL_NEIGHBOR

FRN_EAST:
    ; --- East: X+1, same Y ---
    movf  0xB0, W
    movwf 0xB4
    movf  0xB1, W
    movwf 0xB5
    incf  0xB4, F
    movlw d'10'
    subwf 0xB4, W
    btfsc STATUS, C
    goto  FRN_DONE      ; X+1 >= 10, skip east
    call  REVEAL_NEIGHBOR

FRN_DONE:
    ; Restore centre (REVEAL_NEIGHBOR leaves 0x50/0x51 pointing at neighbour)
    movf  0xB0, W
    movwf 0x50
    movf  0xB1, W
    movwf 0x51
    return

; =============================================================================
; REVEAL_NEIGHBOR
; Reveals candidate tile at (0xB4, 0xB5) if: not revealed, not a mine.
; If neighbour also has 0 mine-neighbours, recurses one level via
; FLOOD_REVEAL_NEIGHBORS (gives ~3-tile chain on open areas).
; Preserves 0xB0/0xB1 (centre) — loads candidate into 0x50/0x51, checks,
; then draws.  0xB0/0xB1 are only written by FLOOD_REVEAL_NEIGHBORS caller.
; =============================================================================
REVEAL_NEIGHBOR:
    ; Load candidate into working position
    movf  0xB4, W
    movwf 0x50
    movf  0xB5, W
    movwf 0x51

    ; Skip if already revealed
    movlw 0x3A
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    return              ; already revealed

    ; Skip if flagged
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    return              ; flagged

    ; Skip if mine
    movlw 0x2D
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  RN_IS_MINE
    ; Not a mine — mark revealed
    movlw 0x3A
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    incf  0xA2, F       ; increment revealed counter

    ; Count its mine neighbours and draw
    call  COUNT_MINES   ; result in W and 0x7C
    call  GET_REVEALED_COLOR
    call  DRAW_TILE

    ; If this neighbour also has 0 neighbours, expand one more level
    movf  0x7C, F
    btfss STATUS, Z
    return              ; has neighbours, stop here

    ; Save B4/B5 across recursive call (they'll be overwritten)
    movf  0xB4, W
    movwf 0xB2          ; borrow 0xB2 as temp save for candX
    movf  0xB5, W
    movwf 0xB3          ; borrow 0xB3 as temp save for candY
    call  FLOOD_REVEAL_NEIGHBORS  ; recurse one level (0xB0/0xB1 set inside)
    ; Restore B4/B5
    movf  0xB2, W
    movwf 0xB4
    movf  0xB3, W
    movwf 0xB5
    return

RN_IS_MINE:
    return

; CHECK_WIN
; Sprawdza czy gracz wygrał (odkryte = 100 - liczba min)
; Jeśli tak, ustawia 0xA1=1 i rysuje zielone pole
; =============================================================================
CHECK_WIN:
    movf  0xA3, W           ; Liczba min
    movwf 0x76
    movlw d'100'
    subwf 0x76, W
    ; W = 100 - miny = wymagana liczba odkrytych
    subwf 0xA2, W           ; (odkryte) - (wymagane)
    btfss STATUS, Z
    return                  ; Jeszcze nie wygrał

    ; --- WYGRANA ---
    movlw 0x01
    movwf 0xA1              ; Stan = wygrana

    ; Rysuj wszystkie pola zielono (zwycięstwo)
    movlw 0x00
    movwf 0x50
WIN_REVEAL_X:
    movlw 0x00
    movwf 0x51
WIN_REVEAL_Y:
    movlw 0x07             ; Zielony RGB565 high
    movwf 0x64
    movlw 0xE0             ; Zielony RGB565 low
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'10'
    subwf 0x51, W
    btfss STATUS, Z
    goto  WIN_REVEAL_Y
    incf  0x50, F
    movlw d'10'
    subwf 0x50, W
    btfss STATUS, Z
    goto  WIN_REVEAL_X
    return

; =============================================================================
; DRAW_LEGEND
; Rysuje legendę kolorów poniżej siatki (Y >= 220 pikseli)
; 9 wpisów: 0 sąsiadów -> kolor tła, 1-8 -> kolory liczb, + min
; Każdy wpis: mały kwadrat 10x10 + etykieta tekstowa (nieobsługiwana przez LCD
; bez czcionki, więc rysujemy same kwadraty z kolorami w rzędzie)
; Legenda: 10 kwadratów 14x14 px, wiersz Y=210..224, X od 8 co 24
; Kolejność: 0=czarny,1=zielony,2=żółty,3=pomarańcz,4=czerwony,
;            5=różowy,6=niebieski,7=fioletowy,8=brązowy,MINE=jasnoczerw
; =============================================================================
DRAW_LEGEND:
    ; Rysuj 9 kolorowych kwadratów (0-8 sąsiadów) + 1 kwadrat miny
    ; Współrzędne: y0=220, y1=233 (14px), x0=8+i*24, x1=x0+13

    ; Wpis 0 - Czarny (0 sąsiadów = bezpieczne puste)
    movlw 0x00
    movwf 0x64
    movwf 0x65
    movlw d'0'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 1 - Zielony
    movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    movlw d'1'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 2 - Żółty
    movlw 0xFF
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    movlw d'2'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 3 - Pomarańczowy
    movlw 0xFC
    movwf 0x64
    movlw 0x00
    movwf 0x65
    movlw d'3'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 4 - Czerwony
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    movlw d'4'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 5 - Różowy
    movlw 0xF8
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    movlw d'5'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 6 - Niebieski
    movlw 0x00
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    movlw d'6'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 7 - Fioletowy
    movlw 0x78
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    movlw d'7'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis 8 - Brązowy
    movlw 0x84
    movwf 0x64
    movlw 0x00
    movwf 0x65
    movlw d'8'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Wpis MINE - Jasnoczerwony (wyróżnik miny)
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    movlw d'9'
    movwf 0x50
    call  DL_DRAW_BOX

    ; Przywróć 0x50/0x51 do bezpiecznych wartości
    clrf  0x50
    clrf  0x51
    return

; DL_DRAW_BOX: rysuje kwadrat 14x14 px legendy
; Wejście: 0x50=indeks (0-9), 0x64/0x65=kolor
; x0 = 8 + indeks*16, x1=x0+13, y0=220, y1=233
DL_DRAW_BOX:
    ; x0 = index*16 + 8
    movf  0x50, W
    movwf 0x75
    swapf 0x75, W           ; W = index * 16
    andlw 0xF0
    addlw d'8'
    movwf 0x60              ; x0
    addlw d'13'
    movwf 0x61              ; x1

    movlw d'220'
    movwf 0x62              ; y0
    movlw d'233'
    movwf 0x63              ; y1

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
    movlw d'14'
    movwf 0x66
DL_LOOP_Y:
    movlw d'14'
    movwf 0x67
DL_LOOP_X:
    bcf   PIR1, 3
    movf  0x64, W
    movwf SSPBUF
DL_WAIT1:
    btfss PIR1, 3
    goto  DL_WAIT1
    bcf   PIR1, 3
    movf  0x65, W
    movwf SSPBUF
DL_WAIT2:
    btfss PIR1, 3
    goto  DL_WAIT2
    decfsz 0x67, F
    goto  DL_LOOP_X
    decfsz 0x66, F
    goto  DL_LOOP_Y
    bsf   PORTC, 2
    return

; =============================================================================
; RYSOWANIE SIATKI I KAFELKÓW
; =============================================================================
DRAW_GRID:
    movlw 0x00
    movwf 0x50
GRID_LOOP_X:
    movlw 0x00
    movwf 0x51
GRID_LOOP_Y:
    movlw 0x84             ; Szary (szary RGB565 high)
    movwf 0x64
    movlw 0x10             ; Szary RGB565 low (0x8410)
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'10'
    subwf 0x51, W
    btfss STATUS, Z
    goto  GRID_LOOP_Y
    incf  0x50, F
    movlw d'10'
    subwf 0x50, W
    btfss STATUS, Z
    goto  GRID_LOOP_X
    return

; =============================================================================
; UPDATE_CURSOR
; Sprawdza czy kursor się przesunął, odrysowuje stare pole i rysuje kursor
; =============================================================================
UPDATE_CURSOR:
    movf  0x50, W
    xorwf 0x54, W
    btfss STATUS, Z
    goto  DO_UPDATE
    movf  0x51, W
    xorwf 0x55, W
    btfss STATUS, Z
    goto  DO_UPDATE
    return

DO_UPDATE:
    ; --- Odrysuj stare pole ---
    call  IS_REVEALED       ; Czy stare pole odkryte?
    btfss STATUS, Z
    goto  DU_WAS_REVEALED

    movlw 0x20              ; Czy flaga?
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  DU_DRAW_YELLOW

DU_DRAW_GREY:
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
    goto  DU_DO_DRAW

DU_DRAW_YELLOW:
    movlw 0xFF
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    goto  DU_DO_DRAW

DU_WAS_REVEALED:
    call  COUNT_MINES
    call  GET_REVEALED_COLOR

DU_DO_DRAW:
    call  DRAW_TILE

    ; --- Rysuj nowy kursor ---
    movf  0x54, W
    movwf 0x50
    movf  0x55, W
    movwf 0x51
    movlw 0xFF             ; Biały kursor
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    return

; =============================================================================
; DRAW_TILE
; Wejście: 0x50=X, 0x51=Y, 0x64/0x65=kolor (RGB565)
; Rysuje kafelek 14x14 pikseli w odpowiednim miejscu ekranu
; =============================================================================
DRAW_TILE:
    ; X_start = X * 24 + 8 (uproszczone: X*16+X*8 ale używamy X*16+przesunięcie)
    ; Dla uproszczenia: piksel = X * 24, Y * 32 (lub podobnie)
    ; ORYGINALNE: swapf daje X*16, +40 offset
    movf  0x50, W
    movwf 0x75
    swapf 0x75, W          ; W = X * 16
    andlw 0xF0
    addlw d'40'            ; Offset od lewej
    movwf 0x60             ; x0
    addlw d'13'
    movwf 0x61             ; x1

    movf  0x51, W
    movwf 0x75
    swapf 0x75, W          ; W = Y * 16
    andlw 0xF0
    addlw d'40'
    movwf 0x62             ; y0
    addlw d'13'
    movwf 0x63             ; y1

    movlw 0x2A             ; Column Address Set
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movf  0x60, W
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movf  0x61, W
    call  SEND_DAT

    movlw 0x2B             ; Row Address Set
    call  SEND_CMD
    movlw 0x00
    call  SEND_DAT
    movf  0x62, W
    call  SEND_DAT
    movlw 0x00
    call  SEND_DAT
    movf  0x63, W
    call  SEND_DAT

    movlw 0x2C             ; Memory Write
    call  SEND_CMD
    bsf   PORTC, 1         ; D/C=1
    bcf   PORTC, 2         ; CS=0

    movlw d'14'
    movwf 0x66
LOOP_RECT_Y:
    movlw d'14'
    movwf 0x67
LOOP_RECT_X:
    bcf   PIR1, 3
    movf  0x64, W
    movwf SSPBUF
WAIT_C1:
    btfss PIR1, 3
    goto  WAIT_C1
    bcf   PIR1, 3
    movf  0x65, W
    movwf SSPBUF
WAIT_C2:
    btfss PIR1, 3
    goto  WAIT_C2
    decfsz 0x67, F
    goto  LOOP_RECT_X
    decfsz 0x66, F
    goto  LOOP_RECT_Y
    bsf   PORTC, 2
    return

; =============================================================================
; GET_REVEALED_COLOR
; Wejście: W = liczba sąsiednich min (0-8), też w 0x7C
; Wyjście: 0x64/0x65 = kolor RGB565
; Paleta:
;   0 -> Czarny  (0x0000) - pole bezpieczne, brak sąsiadów
;   1 -> Zielony (0x07E0)
;   2 -> Żółty   (0xFFE0)
;   3 -> Pomarańczowy (0xFC00)
;   4 -> Czerwony (0xF800)
;   5 -> Różowy  (0xF81F)
;   6 -> Niebieski (0x001F)
;   7 -> Fioletowy (0x781F)
;   8 -> Brązowy (0x8400)
; =============================================================================
GET_REVEALED_COLOR:
    movwf 0x76
    movf  0x76, F
    btfsc STATUS, Z
    goto  GRC_0
    movlw 0x01
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_1
    movlw 0x02
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_2
    movlw 0x03
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_3
    movlw 0x04
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_4
    movlw 0x05
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_5
    movlw 0x06
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_6
    movlw 0x07
    subwf 0x76, W
    btfsc STATUS, Z
    goto  GRC_7
    goto  GRC_8            ; 8 lub więcej

GRC_0:                     ; Czarny
    movlw 0x00
    movwf 0x64
    movwf 0x65
    return
GRC_1:                     ; Zielony
    movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    return
GRC_2:                     ; Żółty
    movlw 0xFF
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_3:                     ; Pomarańczowy
    movlw 0xFC
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_4:                     ; Czerwony
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_5:                     ; Różowy
    movlw 0xF8
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_6:                     ; Niebieski
    movlw 0x00
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_7:                     ; Fioletowy
    movlw 0x78
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_8:                     ; Brązowy
    movlw 0x84
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return

; =============================================================================
; REVEAL_ALL_MINES - odkrywa wszystkie miny (Game Over)
; =============================================================================
REVEAL_ALL_MINES:
    movlw 0x00
    movwf 0x50
RAM_LOOP_X:
    movlw 0x00
    movwf 0x51
RAM_LOOP_Y:
    call  IS_MINE
    btfsc STATUS, Z
    goto  RAM_NEXT          ; Nie mina
    ; Rysuj minę jako czerwony
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
RAM_NEXT:
    incf  0x51, F
    movlw d'10'
    subwf 0x51, W
    btfss STATUS, Z
    goto  RAM_LOOP_Y
    incf  0x50, F
    movlw d'10'
    subwf 0x50, W
    btfss STATUS, Z
    goto  RAM_LOOP_X
    return

; =============================================================================
; SPI I LCD
; =============================================================================
SEND_CMD:
    bcf   PORTC, 1          ; D/C=0 (komenda)
    bcf   PORTC, 2          ; CS=0
    goto  SPI_TX
SEND_DAT:
    bsf   PORTC, 1          ; D/C=1 (dane)
    bcf   PORTC, 2          ; CS=0
SPI_TX:
    bcf   PIR1, 3
    movwf SSPBUF
SPI_WAIT:
    btfss PIR1, 3
    goto  SPI_WAIT
    bsf   PORTC, 2          ; CS=1
    return

; =============================================================================
; CHECK_BUTTONS
; RB0=UP, RB1=R(right), RB2=DOWN, RB3=L(left), RB4=FLAG, RB6=CHECK, RB7=RESTART
; =============================================================================
CHECK_BUTTONS:
    ; RB7=RESTART działa zawsze (nawet po game over)
    btfss PORTB, 7
    goto  CB_SKIP_RESTART
    call  DO_RESTART
    return
CB_SKIP_RESTART:

    ; Jeśli gra skończona, nie reaguj na pozostałe przyciski
    movf  0xA1, F
    btfss STATUS, Z
    return

    movf  0x50, W
    movwf 0x54
    movf  0x51, W
    movwf 0x55

CHECK_RIGHT:
    btfss PORTB, 3
    goto  CHECK_LEFT
    movlw d'9'
    subwf 0x54, W
    btfsc STATUS, Z
    goto  WAIT_RELEASE
    incf  0x54, F
    goto  WAIT_RELEASE

CHECK_LEFT:
    btfss PORTB, 1
    goto  CHECK_DOWN
    movf  0x54, W
    btfsc STATUS, Z
    goto  WAIT_RELEASE
    decf  0x54, F
    goto  WAIT_RELEASE

CHECK_DOWN:
    btfss PORTB, 2
    goto  CHECK_UP
    movlw d'9'
    subwf 0x55, W
    btfsc STATUS, Z
    goto  WAIT_RELEASE
    incf  0x55, F
    goto  WAIT_RELEASE

CHECK_UP:
    btfss PORTB, 0
    goto  CHECK_FLAG
    movf  0x55, W
    btfsc STATUS, Z
    goto  WAIT_RELEASE
    decf  0x55, F
    goto  WAIT_RELEASE

CHECK_FLAG:
    btfss PORTB, 4
    goto  CHECK_REVEAL

    ; Nie można flagować odkrytego pola
    call  IS_REVEALED
    btfss STATUS, Z
    goto  WAIT_RELEASE      ; Odkryte, ignoruj

    ; Przełącz flagę (XOR)
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    xorwf INDF, F

    ; Rysuj odpowiedni kolor (żółty=flaga, szary=brak flagi)
    movf  0x58, W
    andwf INDF, W
    btfsc STATUS, Z
    goto  CF_DRAW_UNFLAGGED
CF_DRAW_FLAGGED:
    movlw 0xFF
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    goto  CF_DO_BLINK
CF_DRAW_UNFLAGGED:
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
CF_DO_BLINK:
    call  DRAW_TILE
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xFF             ; Pokaż kursor
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    goto  WAIT_RELEASE

CHECK_REVEAL:
    btfss PORTB, 6
    goto  END_CHECK

    ; 1. Ignoruj jeśli ma flagę
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  WAIT_RELEASE      ; Flaga blokuje!

    ; 2. Ignoruj jeśli już odkryte
    call  IS_REVEALED
    btfss STATUS, Z
    goto  WAIT_RELEASE      ; Już odkryte

    ; 3. Czy to mina?
    call  IS_MINE
    btfsc STATUS, Z
    goto  DO_REVEAL_SAFE    ; Nie mina

    ; --- MINA - Przegrana ---
    movlw 0x02
    movwf 0xA1              ; Stan = przegrana

    ; Narysuj trafioną minę
    movlw 0xF8             ; Czerwony
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE

    ; Odkryj wszystkie miny
    call  REVEAL_ALL_MINES

    ; Efekt blink na przegranej minie (3 razy)
    movlw d'3'
    movwf 0x75
BLINK_LOOP:
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xFF             ; Biały
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    call  DELAY_20MS
    call  DELAY_20MS
    movlw 0xF8             ; Czerwony
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
    decfsz 0x75, F
    goto  BLINK_LOOP
    goto  END_CHECK

DO_REVEAL_SAFE:
    ; Odkryj pole (z flood-fill jeśli 0 sąsiadów)
    call  REVEAL_TILE

    ; Sprawdź wygraną
    call  CHECK_WIN

WAIT_RELEASE:
    call  DELAY_20MS
    movf  PORTB, W
    andlw b'11111111'      ; RB0-RB7 — all buttons
    btfss STATUS, Z
    goto  WAIT_RELEASE
END_CHECK:
    return

; =============================================================================
; DO_RESTART - pełny reset gry bez reinicjalizacji LCD
; =============================================================================
DO_RESTART:
    ; Czekaj na zwolnienie RB7
DR_WAIT:
    call  DELAY_20MS
    btfsc PORTB, 7
    goto  DR_WAIT

    ; Wyczyść RAM gry
    movlw 0x20
    movwf FSR
DR_CLEAR:
    clrf  INDF
    incf  FSR, F
    movlw 0x47
    subwf FSR, W
    btfss STATUS, Z
    goto  DR_CLEAR

    ; Reset zmiennych stanu
    clrf  0xA0             ; Stack pointer
    clrf  0xA1             ; Stan gry = trwa
    clrf  0xA2             ; Odkryte = 0
    movlw d'10'
    movwf 0xA3             ; Liczba min

    ; Przywróć kursor do (4,4)
    movlw d'4'
    movwf 0x50
    movwf 0x51
    movwf 0x54
    movwf 0x55

    call  PLANT_DUMMY_MINES
    call  DRAW_GRID
    call  DRAW_LEGEND

    ; Narysuj kursor startowy
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    return

; =============================================================================
; DELAY
; =============================================================================
DELAY_20MS:
    movlw d'100'
    movwf 0x70
DELAY_OUTER:
    movlw d'66'
    movwf 0x71
DELAY_INNER:
    decfsz 0x71, F
    goto  DELAY_INNER
    decfsz 0x70, F
    goto  DELAY_OUTER
    return

    END
