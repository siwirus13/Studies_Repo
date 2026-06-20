; =============================================================================
; SAPER (Minesweeper) dla PIC16F877A + ILI9341 (SPI)
; WERSJA: 5x5, 7 min, poprawiony flood-fill (BFS na stosie software),
;         poprawiony kolor dla 2 sąsiadów (niebieski 0x001F)
; =============================================================================
; MAPA REJESTRÓW:
;   0x20-0x23 : Warstwa flag    (5x5 = 25 bitów -> 4 bajty)
;   0x24-0x27 : Warstwa min     (5x5 = 25 bitów -> 4 bajty)
;   0x28-0x2B : Warstwa odkrytych (5x5 = 25 bitów -> 4 bajty)
;
;   0x50 : Aktualny X kursora (bieżące pole przy rysowaniu)
;   0x51 : Aktualny Y kursora
;   0x54 : Docelowy X (po wciśnięciu przycisku)
;   0x55 : Docelowy Y
;   0x56 : Tymczasowy indeks liniowy
;   0x57 : Tymczasowy (przesunięcie bajtowe / licznik pętli)
;   0x58 : Wynikowa maska bitu
;   0x59 : Adres bazowy warstwy (0x20/0x24/0x28)
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
; STOS BFS FLOOD-FILL (software stack, 16 pozycji)
;   0x80-0x8F : Bufor X stosu (16 wpisów)
;   0x90-0x9F : Bufor Y stosu (16 wpisów)
;   0xA0 : Wskaźnik wierzchołka stosu (0-16)
;
; KONTROLA GRY
;   0xA1 : Stan gry (0=trwa, 1=wygrana, 2=przegrana)
;   0xA2 : Licznik odkrytych pól
;   0xA3 : Łączna liczba min
;
; ROZMIAR KAFELKA / UKŁAD EKRANU (5x5):
;   Stride = 24px, rozmiar kafelka = 22x22px
;   x0 = X*24 + 60,  x1 = x0+21
;   y0 = Y*24 + 80,  y1 = y0+21
;   Legenda: y=260..273
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
    movwf TRISB             ; PORTB jako wejścia (przyciski RB0-RB7)
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

    ; --- Wyczyść RAM gry (0x20-0x2B, 4 bajty × 3 warstwy = 12 bajtów) ---
    movlw 0x20
    movwf FSR
CLEAR_RAM_LOOP:
    clrf  INDF
    incf  FSR, F
    movlw 0x2C
    subwf FSR, W
    btfss STATUS, Z
    goto  CLEAR_RAM_LOOP

    ; --- Inicjalizacja zmiennych gry ---
    clrf  0xA0              ; Stack pointer BFS = 0
    clrf  0xA1              ; Stan gry = trwa
    clrf  0xA2              ; Odkryte = 0
    movlw d'7'              ; 7 min
    movwf 0xA3

    call  PLANT_DUMMY_MINES
    call  DRAW_GRID
    call  DRAW_LEGEND

GAME_INIT:
    movlw d'2'
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
; ROZMIESZCZENIE MIN (7 min na siatce 5x5, hardcoded do testów)
; Miny na: (0,0),(4,4),(1,3),(3,1),(0,4),(2,2),(4,1)
; =============================================================================
PLANT_DUMMY_MINES:
    movlw 0x24              ; Warstwa min
    movwf 0x59
    ; X=0, Y=0
    movlw d'0'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=4, Y=4
    movlw d'4'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=1, Y=3
    movlw d'1'
    movwf 0x50
    movlw d'3'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=3, Y=1
    movlw d'3'
    movwf 0x50
    movlw d'1'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=0, Y=4
    movlw d'0'
    movwf 0x50
    movlw d'4'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=2, Y=2
    movlw d'2'
    movwf 0x50
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    ; X=4, Y=1
    movlw d'4'
    movwf 0x50
    movlw d'1'
    movwf 0x51
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    return

; =============================================================================
; GET_BITMASK_AND_PTR
; Wejście: 0x50=X, 0x51=Y, 0x59=adres bazowy warstwy
; Wyjście: FSR wskazuje na właściwy bajt, 0x58=maska bitu
; Formuła: index = Y*5 + X
; =============================================================================
GET_BITMASK_AND_PTR:
    ; Index = Y*5 + X
    movf  0x51, W
    movwf 0x57          ; Licznik Y (iteracje mnożenia)
    movf  0x50, W
    movwf 0x56          ; Index = X na start
    movf  0x57, F
    btfsc STATUS, Z
    goto  GTB_CALC_OFFSET
GTB_LOOP_Y:
    movlw d'5'
    addwf 0x56, F       ; Index += 5 (bo siatka 5 szeroka)
    decfsz 0x57, F
    goto  GTB_LOOP_Y

GTB_CALC_OFFSET:
    ; Bajt = Index / 8 (3x rrf)
    movf  0x56, W
    movwf 0x57
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    bcf   STATUS, C
    rrf   0x57, F
    movlw 0x07          ; Max 4 bajty dla 5x5 (25 bitów)
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
; Wyjście: W i 0x7C = liczba min w sąsiadujących 8 polach
; Granica planszy: 5 (siatka 5x5)
; =============================================================================
COUNT_MINES:
    clrf  0x7C

    movlw 0xFF
    movwf 0x7D              ; DY = -1
CM_LOOP_Y:
    movlw 0xFF
    movwf 0x7E              ; DX = -1
CM_LOOP_X:
    movf  0x7E, F
    btfss STATUS, Z
    goto  CM_CALC_COORD
    movf  0x7D, F
    btfss STATUS, Z
    goto  CM_CALC_COORD
    goto  CM_NEXT_X         ; (0,0) - samo pole, pomiń

CM_CALC_COORD:
    ; sX = X + DX (DX jest 0xFF dla -1, 0x00, 0x01)
    movf  0x50, W
    addwf 0x7E, W
    movwf 0x7A              ; sX

    ; Odrzuć jeśli sX >= 5 (obsługuje też wraparound przy X=0,DX=-1)
    movlw d'5'
    subwf 0x7A, W           ; W = sX - 5; C=1 jeśli sX>=5
    btfsc STATUS, C
    goto  CM_NEXT_X

    ; sY = Y + DY
    movf  0x51, W
    addwf 0x7D, W
    movwf 0x7B              ; sY

    movlw d'5'
    subwf 0x7B, W           ; W = sY - 5; C=1 jeśli sY>=5
    btfsc STATUS, C
    goto  CM_NEXT_X

    call  CHECK_MINE_AT
    addwf 0x7C, F

CM_NEXT_X:
    incf  0x7E, F
    movlw 0x02
    subwf 0x7E, W
    btfss STATUS, Z
    goto  CM_LOOP_X

CM_NEXT_Y:
    incf  0x7D, F
    movlw 0x02
    subwf 0x7D, W
    btfss STATUS, Z
    goto  CM_LOOP_Y

    movf  0x7C, W
    return

; =============================================================================
; CHECK_MINE_AT
; Wejście: 0x7A=X sąsiada, 0x7B=Y sąsiada
; Wyjście: W = 1 jeśli mina, 0 jeśli nie
; Zachowuje: 0x50, 0x51
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

; =============================================================================
; IS_REVEALED
; Wejście: 0x50=X, 0x51=Y
; Wyjście: STATUS,Z=1 jeśli NIE odkryte, Z=0 jeśli odkryte
; =============================================================================
IS_REVEALED:
    movlw 0x28
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; =============================================================================
; IS_MINE
; Wejście: 0x50=X, 0x51=Y
; Wyjście: STATUS,Z=1 jeśli NIE mina, Z=0 jeśli mina
; =============================================================================
IS_MINE:
    movlw 0x24
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    return

; =============================================================================
; MARK_REVEALED - ustawia bit odkrycia dla pola (0x50, 0x51)
; =============================================================================
MARK_REVEALED:
    movlw 0x28
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    iorwf INDF, F
    return

; =============================================================================
; REVEAL_TILE
; Wejście: 0x50=X, 0x51=Y
; Odkrywa pole; jeśli 0 sąsiadów-min, inicjuje BFS flood-fill (stos iteracyjny)
; =============================================================================
REVEAL_TILE:
    ; 1. Jeśli już odkryte, wyjdź
    call  IS_REVEALED
    btfss STATUS, Z
    return

    ; 2. Jeśli ma flagę, zignoruj
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    return

    ; 3. Oznacz jako odkryte
    call  MARK_REVEALED
    incf  0xA2, F

    ; 4. Policz sąsiednie miny i narysuj
    call  COUNT_MINES       ; W = 0x7C = liczba
    call  GET_REVEALED_COLOR
    call  DRAW_TILE

    ; 5. Jeśli 0 min w sąsiedztwie, użyj BFS flood-fill
    movf  0x7C, F
    btfss STATUS, Z
    return                  ; Ma sąsiadów-min, stop

    ; --- Inicjalizacja stosu BFS ---
    clrf  0xA0              ; Reset wskaźnika stosu

    ; Push bieżącego pola
    movlw 0x80
    addwf 0xA0, W
    movwf FSR
    movf  0x50, W
    movwf INDF              ; Stos X [0] = X

    movlw 0x90
    addwf 0xA0, W
    movwf FSR
    movf  0x51, W
    movwf INDF              ; Stos Y [0] = Y

    incf  0xA0, F           ; SP = 1

    ; --- BFS główna pętla ---
BFS_LOOP:
    movf  0xA0, F
    btfsc STATUS, Z
    return                  ; Stos pusty, koniec

    ; Pop ze stosu (LIFO: SP-1)
    decf  0xA0, F

    movlw 0x80
    addwf 0xA0, W
    movwf FSR
    movf  INDF, W
    movwf 0x50              ; X = Stos X [SP]

    movlw 0x90
    addwf 0xA0, W
    movwf FSR
    movf  INDF, W
    movwf 0x51              ; Y = Stos Y [SP]

    ; Sprawdź 4 sąsiadów (N/S/W/E)
    ; --- North: Y-1 ---
    movf  0x51, F
    btfsc STATUS, Z
    goto  BFS_SOUTH         ; Y==0, pomiń
    movf  0x51, W
    movwf 0x7B
    decf  0x7B, F           ; NY = Y-1
    movf  0x50, W
    movwf 0x7A              ; NX = X
    call  BFS_TRY_PUSH

BFS_SOUTH:
    ; --- South: Y+1 ---
    movf  0x51, W
    movwf 0x7B
    incf  0x7B, F           ; NY = Y+1
    movlw d'5'
    subwf 0x7B, W
    btfsc STATUS, C
    goto  BFS_WEST          ; Y+1 >= 5, pomiń
    movf  0x50, W
    movwf 0x7A
    call  BFS_TRY_PUSH

BFS_WEST:
    ; --- West: X-1 ---
    movf  0x50, F
    btfsc STATUS, Z
    goto  BFS_EAST          ; X==0, pomiń
    movf  0x50, W
    movwf 0x7A
    decf  0x7A, F           ; NX = X-1
    movf  0x51, W
    movwf 0x7B              ; NY = Y
    call  BFS_TRY_PUSH

BFS_EAST:
    ; --- East: X+1 ---
    movf  0x50, W
    movwf 0x7A
    incf  0x7A, F           ; NX = X+1
    movlw d'5'
    subwf 0x7A, W
    btfsc STATUS, C
    goto  BFS_LOOP          ; X+1 >= 5, pomiń; kontynuuj pętlę BFS
    movf  0x51, W
    movwf 0x7B
    call  BFS_TRY_PUSH
    goto  BFS_LOOP

; =============================================================================
; BFS_TRY_PUSH
; Wejście: 0x7A=NX, 0x7B=NY (kandydat)
; Jeśli kandydat nie odkryty, nie flagowany, nie mina:
;   odkrywa go, rysuje, i jeśli ma 0 sąsiadów – wrzuca na stos
; Zachowuje 0x50/0x51 (bieżące centrum BFS)
; =============================================================================
BFS_TRY_PUSH:
    ; Zachowaj centrum
    movf  0x50, W
    movwf 0xB0
    movf  0x51, W
    movwf 0xB1

    ; Ustaw kandydata jako bieżące pole
    movf  0x7A, W
    movwf 0x50
    movf  0x7B, W
    movwf 0x51

    ; Pomiń jeśli już odkryte
    movlw 0x28
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  BTP_RESTORE       ; Już odkryte

    ; Pomiń jeśli flagowane
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  BTP_RESTORE       ; Flagowane

    ; Pomiń jeśli mina
    movlw 0x24
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  BTP_RESTORE       ; To mina – nie odkrywaj

    ; Odkryj kandydata
    call  MARK_REVEALED
    incf  0xA2, F

    ; Policz sąsiadów i narysuj
    call  COUNT_MINES       ; W = 0x7C = liczba
    call  GET_REVEALED_COLOR
    call  DRAW_TILE

    ; Jeśli 0 sąsiadów i stos nie pełny (max 16) → push na stos
    movf  0x7C, F
    btfss STATUS, Z
    goto  BTP_RESTORE       ; Ma sąsiadów, nie rozszerzaj

    ; Sprawdź czy stos pełny
    movlw d'16'
    subwf 0xA0, W
    btfsc STATUS, C
    goto  BTP_RESTORE       ; Stos pełny

    ; Push: Stos X [SP] = NX (= 0x50 teraz), Stos Y [SP] = NY (= 0x51)
    movlw 0x80
    addwf 0xA0, W
    movwf FSR
    movf  0x50, W
    movwf INDF

    movlw 0x90
    addwf 0xA0, W
    movwf FSR
    movf  0x51, W
    movwf INDF

    incf  0xA0, F

BTP_RESTORE:
    ; Przywróć centrum
    movf  0xB0, W
    movwf 0x50
    movf  0xB1, W
    movwf 0x51
    return

; =============================================================================
; CHECK_WIN
; Sprawdza czy odkryte = 25 - 7 = 18 pól
; =============================================================================
CHECK_WIN:
    ; Wymagana liczba odkrytych = 25 - liczba_min
    movf  0xA3, W           ; W = liczba min (7)
    movwf 0x76
    movlw d'25'             ; Rozmiar siatki 5x5
    subwf 0x76, W           ; Hmm: subwf robi f-W; chcemy 25-miny
    ; Poprawka: subwf 0x76, W = 0x76 - W = miny - 25 ... niedobrze
    ; Użyj: W = 25 - miny
    movf  0xA3, W           ; W = miny
    sublw d'25'             ; W = 25 - W (sublw: W = k - W)
    ; Teraz W = 25 - miny = wymagana liczba odkrytych
    subwf 0xA2, W           ; W = odkryte - wymagane (jeśli =0 → wygrana)
    btfss STATUS, Z
    return                  ; Jeszcze nie wygrał

    ; --- WYGRANA ---
    movlw 0x01
    movwf 0xA1

    movlw 0x00
    movwf 0x50
WIN_REVEAL_X:
    movlw 0x00
    movwf 0x51
WIN_REVEAL_Y:
    movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  WIN_REVEAL_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  WIN_REVEAL_X
    return

; =============================================================================
; DRAW_LEGEND
; 8 kolorowych kwadratów (0-7 sąsiadów) + kwadrat miny
; y0=260, y1=273 (14px), x0=8+i*24, x1=x0+13
; Kolory:
;   0=Czarny, 1=Zielony, 2=Niebieski(fix!), 3=Pomarańcz, 4=Czerwony,
;   5=Różowy,  6=Jasnoniebieski, 7=Fioletowy, MINE=Czerwony(wyróżnik)
; =============================================================================
DRAW_LEGEND:
    ; Wpis 0 - Czarny
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

    ; Wpis 2 - Niebieski (0x001F) ← POPRAWKA: było żółte/białe
    movlw 0x00
    movwf 0x64
    movlw 0x1F
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

    ; Wpis 6 - Jasnoniebieski (0x07FF cyan)
    movlw 0x07
    movwf 0x64
    movlw 0xFF
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

    ; Wpis MINE - Jasnoczerw (wyróżnik)
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    movlw d'8'
    movwf 0x50
    call  DL_DRAW_BOX

    clrf  0x50
    clrf  0x51
    return

; DL_DRAW_BOX: kwadrat 14x14 legendy
; Wejście: 0x50=indeks (0-8), 0x64/0x65=kolor
; x0=8+idx*24, x1=x0+13, y0=260, y1=273
DL_DRAW_BOX:
    ; x0 = index*24 + 8
    ; index*24 = index*16 + index*8
    movf  0x50, W
    movwf 0x75
    swapf 0x75, W
    andlw 0xF0              ; W = index*16
    movwf 0x60

    movf  0x50, W
    movwf 0x75
    bcf   STATUS, C
    rlf   0x75, F           ; *2
    bcf   STATUS, C
    rlf   0x75, F           ; *4
    bcf   STATUS, C
    rlf   0x75, F           ; *8
    movf  0x75, W
    addwf 0x60, F           ; 0x60 = index*24
    movlw d'8'
    addwf 0x60, F           ; 0x60 = index*24 + 8

    movf  0x60, W
    addlw d'13'
    movwf 0x61              ; x1 = x0+13

    movlw d'260'            ; y0 (poniżej siatki 5x5)
    movwf 0x62
    movlw d'0'              ; Uwaga: 260 > 255, wysyłamy high=0x01, low=0x04
    ; y0=260: high=0x01, low=0x04; y1=273: high=0x01, low=0x11
    ; Obsłuż to bezpośrednio w wysyłaniu

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
    movlw 0x01              ; y0 high byte (260 = 0x0104)
    call  SEND_DAT
    movlw 0x04              ; y0 low byte
    call  SEND_DAT
    movlw 0x01              ; y1 high byte (273 = 0x0111)
    call  SEND_DAT
    movlw 0x11              ; y1 low byte
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
; DRAW_GRID - rysuje siatkę 5x5 szarymi kafelkami
; =============================================================================
DRAW_GRID:
    movlw 0x00
    movwf 0x50
GRID_LOOP_X:
    movlw 0x00
    movwf 0x51
GRID_LOOP_Y:
    movlw 0x84
    movwf 0x64
    movlw 0x10
    movwf 0x65
    call  DRAW_TILE
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  GRID_LOOP_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  GRID_LOOP_X
    return

; =============================================================================
; UPDATE_CURSOR - odrysowuje stare pole, rysuje kursor na nowej pozycji
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
    ; Odrysuj stare pole (0x50/0x51 = stara pozycja)
    call  IS_REVEALED
    btfss STATUS, Z
    goto  DU_WAS_REVEALED

    ; Nie odkryte – sprawdź flagę
    movlw 0x20
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
    call  COUNT_MINES       ; Wynik w W i 0x7C (0x50/0x51 = stara poz.)
    call  GET_REVEALED_COLOR

DU_DO_DRAW:
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
; DRAW_TILE
; Wejście: 0x50=X (0-4), 0x51=Y (0-4), 0x64/0x65=kolor RGB565
; Rysuje kafelek 22x22 pikseli.
; x0 = X*24 + 60,  x1 = x0+21
; y0 = Y*24 + 80,  y1 = y0+21
; =============================================================================
DRAW_TILE:
    ; --- Oblicz x0 = X*24 + 60 ---
    ; X*24 = X*16 + X*8
    movf  0x50, W
    movwf 0x75
    swapf 0x75, W
    andlw 0xF0              ; W = X*16
    movwf 0x60              ; 0x60 = X*16

    movf  0x50, W
    movwf 0x75
    bcf   STATUS, C
    rlf   0x75, F           ; 0x75 = X*2
    bcf   STATUS, C
    rlf   0x75, F           ; 0x75 = X*4
    bcf   STATUS, C
    rlf   0x75, F           ; 0x75 = X*8
    movf  0x75, W
    addwf 0x60, F           ; 0x60 = X*24

    movlw d'60'
    addwf 0x60, F           ; 0x60 = X*24 + 60 = x0
    movf  0x60, W
    addlw d'21'
    movwf 0x61              ; x1 = x0+21

    ; --- Oblicz y0 = Y*24 + 80 ---
    movf  0x51, W
    movwf 0x75
    swapf 0x75, W
    andlw 0xF0              ; W = Y*16
    movwf 0x62

    movf  0x51, W
    movwf 0x75
    bcf   STATUS, C
    rlf   0x75, F           ; Y*2
    bcf   STATUS, C
    rlf   0x75, F           ; Y*4
    bcf   STATUS, C
    rlf   0x75, F           ; Y*8
    movf  0x75, W
    addwf 0x62, F           ; 0x62 = Y*24

    movlw d'80'
    addwf 0x62, F           ; 0x62 = Y*24 + 80 = y0
    movf  0x62, W
    addlw d'21'
    movwf 0x63              ; y1 = y0+21

    ; --- Wyślij okno do LCD ---
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
LOOP_RECT_Y:
    movlw d'22'
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
; Wejście: W = liczba sąsiednich min (0-8)
; Wyjście: 0x64/0x65 = kolor RGB565
;   0 -> Czarny    (0x0000)
;   1 -> Zielony   (0x07E0)
;   2 -> Niebieski (0x001F)  ← POPRAWKA
;   3 -> Pomarańcz (0xFC00)
;   4 -> Czerwony  (0xF800)
;   5 -> Różowy    (0xF81F)
;   6 -> Cyan      (0x07FF)
;   7 -> Fioletowy (0x781F)
;   8 -> Brązowy   (0x8400)
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
    goto  GRC_8

GRC_0:
    movlw 0x00
    movwf 0x64
    movwf 0x65
    return
GRC_1:
    movlw 0x07
    movwf 0x64
    movlw 0xE0
    movwf 0x65
    return
GRC_2:                      ; Niebieski 0x001F ← POPRAWKA
    movlw 0x00
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_3:
    movlw 0xFC
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_4:
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    return
GRC_5:
    movlw 0xF8
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_6:                      ; Cyan 0x07FF
    movlw 0x07
    movwf 0x64
    movlw 0xFF
    movwf 0x65
    return
GRC_7:
    movlw 0x78
    movwf 0x64
    movlw 0x1F
    movwf 0x65
    return
GRC_8:
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
    goto  RAM_NEXT
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
RAM_NEXT:
    incf  0x51, F
    movlw d'5'
    subwf 0x51, W
    btfss STATUS, Z
    goto  RAM_LOOP_Y
    incf  0x50, F
    movlw d'5'
    subwf 0x50, W
    btfss STATUS, Z
    goto  RAM_LOOP_X
    return

; =============================================================================
; SPI I LCD
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
SPI_WAIT:
    btfss PIR1, 3
    goto  SPI_WAIT
    bsf   PORTC, 2
    return

; =============================================================================
; CHECK_BUTTONS
; RB0=UP, RB1=RIGHT, RB2=DOWN, RB3=LEFT, RB4=FLAG, RB6=REVEAL, RB7=RESTART
; =============================================================================
CHECK_BUTTONS:
    ; RB7=RESTART działa zawsze
    btfss PORTB, 7
    goto  CB_SKIP_RESTART
    call  DO_RESTART
    return
CB_SKIP_RESTART:

    movf  0xA1, F
    btfss STATUS, Z
    return                  ; Gra skończona, ignoruj

    movf  0x50, W
    movwf 0x54
    movf  0x51, W
    movwf 0x55

CHECK_RIGHT:
    btfss PORTB, 3
    goto  CHECK_LEFT
    movlw d'4'              ; Max X = 4 (siatka 5x5)
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
    movlw d'4'              ; Max Y = 4
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
    goto  CHECK_REVEAL_BTN

    ; Nie można flagować odkrytego
    call  IS_REVEALED
    btfss STATUS, Z
    goto  WAIT_RELEASE

    ; Przełącz flagę
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    xorwf INDF, F

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
    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    goto  WAIT_RELEASE

CHECK_REVEAL_BTN:
    btfss PORTB, 6
    goto  END_CHECK

    ; Ignoruj jeśli flaga
    movlw 0x20
    movwf 0x59
    call  GET_BITMASK_AND_PTR
    movf  0x58, W
    andwf INDF, W
    btfss STATUS, Z
    goto  WAIT_RELEASE

    ; Ignoruj jeśli już odkryte
    call  IS_REVEALED
    btfss STATUS, Z
    goto  WAIT_RELEASE

    ; Mina?
    call  IS_MINE
    btfsc STATUS, Z
    goto  DO_REVEAL_SAFE

    ; --- PRZEGRANA ---
    movlw 0x02
    movwf 0xA1
    movlw 0xF8
    movwf 0x64
    movlw 0x00
    movwf 0x65
    call  DRAW_TILE
    call  REVEAL_ALL_MINES

    movlw d'3'
    movwf 0x75
BLINK_LOOP:
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
    goto  BLINK_LOOP
    goto  END_CHECK

DO_REVEAL_SAFE:
    call  REVEAL_TILE
    call  CHECK_WIN

WAIT_RELEASE:
    call  DELAY_20MS
    movf  PORTB, W
    andlw b'11111111'
    btfss STATUS, Z
    goto  WAIT_RELEASE
END_CHECK:
    return

; =============================================================================
; DO_RESTART - pełny reset gry bez reinicjalizacji LCD
; =============================================================================
DO_RESTART:
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
    movlw 0x2C
    subwf FSR, W
    btfss STATUS, Z
    goto  DR_CLEAR

    clrf  0xA0
    clrf  0xA1
    clrf  0xA2
    movlw d'7'
    movwf 0xA3

    movlw d'2'
    movwf 0x50
    movwf 0x51
    movwf 0x54
    movwf 0x55

    call  PLANT_DUMMY_MINES
    call  DRAW_GRID
    call  DRAW_LEGEND

    movlw 0xFF
    movwf 0x64
    movwf 0x65
    call  DRAW_TILE
    return

; =============================================================================
; DELAY_20MS
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
